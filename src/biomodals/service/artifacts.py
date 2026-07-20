"""Verified local cache for immutable final result archives."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import re
import stat
from collections.abc import AsyncIterable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from shutil import disk_usage
from threading import Lock
from typing import TypeVar
from uuid import UUID, uuid4

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW
_WRITE_FLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
_T = TypeVar("_T")


class ArtifactIntegrityError(RuntimeError):
    """Raised when Modal bytes do not match the recorded final artifact."""


class ArtifactSourceMissingError(RuntimeError):
    """Raised when an authoritative remote Result source is missing."""


@dataclass(frozen=True, slots=True)
class CacheUsage:
    """Filesystem-backed Result cache metrics."""

    cached_entries: int
    cached_bytes: int
    staging_entries: int
    staging_bytes: int
    free_bytes: int
    reclaimable_entries: int
    reclaimable_bytes: int


@dataclass(frozen=True, slots=True)
class CacheCleanup:
    """Actual unleased cache entries removed by one cleanup."""

    entries: int
    bytes: int
    job_ids: tuple[str, ...]


class ArtifactLease:
    """A verified archive descriptor held for exactly one response."""

    def __init__(
        self,
        descriptor: int,
        *,
        path: Path | None,
        cache: ArtifactCache | None = None,
        job_id: str | None = None,
    ) -> None:
        """Hold an open descriptor and its optional cache reference."""
        self._descriptor: int | None = descriptor
        self.path = path
        self._cache = cache
        self._job_id = job_id

    def read(self, size: int) -> bytes:
        """Read bytes from the verified descriptor."""
        if self._descriptor is None:
            raise ValueError("Artifact lease is closed")
        return os.read(self._descriptor, size)

    def seek(self, offset: int) -> None:
        """Move the verified descriptor to an absolute byte offset."""
        if self._descriptor is None:
            raise ValueError("Artifact lease is closed")
        os.lseek(self._descriptor, offset, os.SEEK_SET)

    def close(self) -> None:
        """Close the descriptor and make its cache entry evictable."""
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            os.close(descriptor)
        finally:
            if self._cache is not None and self._job_id is not None:
                self._cache._release(self._job_id)


class ArtifactCache:
    """Explicitly managed local cache; Modal Volume storage is authoritative."""

    def __init__(self, directory: Path) -> None:
        """Configure a cache directory without automatic size eviction."""
        self.directory = directory
        self._active: dict[str, int] = {}
        self._verified: dict[str, tuple[int, str, int, int, int, int]] = {}
        self._fill_tasks: dict[str, asyncio.Task[None]] = {}
        self._state_lock = Lock()
        self._closed = False
        self._worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="biomodals-artifact",
        )
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(
                self.directory,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
        except OSError as exc:
            raise RuntimeError(
                "Artifact cache directory must be a real directory"
            ) from exc
        try:
            os.fchmod(descriptor, 0o700)
        finally:
            os.close(descriptor)
        for path, _stat in self._archives():
            descriptor = self._open(path)
            if descriptor is None:
                continue
            try:
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
        with os.scandir(self.directory) as entries:
            for entry in entries:
                if not entry.name.endswith(".part"):
                    continue
                file_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(file_stat.st_mode):
                    os.unlink(entry.path)

    def check_ready(self) -> None:
        """Confirm the configured cache directory remains locally usable."""
        descriptor = os.open(
            self.directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        os.close(descriptor)
        if not os.access(self.directory, os.R_OK | os.W_OK | os.X_OK):
            raise RuntimeError("Artifact cache directory is not usable")

    async def run_bounded(
        self,
        operation: Callable[..., _T],
        /,
        *args: object,
        **kwargs: object,
    ) -> _T:
        """Run whole-file or directory work on the single artifact worker."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._worker,
            partial(operation, *args, **kwargs),
        )

    async def check_ready_async(self) -> None:
        """Verify both cache storage and the bounded worker."""
        await self.run_bounded(self.check_ready)

    async def acquire_async(
        self,
        job_id: str,
        *,
        size_bytes: int,
        sha256: str,
    ) -> ArtifactLease | None:
        """Verify a cache hit outside the FastAPI event loop."""
        known, lease = self._acquire_verified(
            job_id,
            size_bytes=size_bytes,
            sha256=sha256,
        )
        if known:
            return lease
        return await self.run_bounded(
            self.acquire,
            job_id,
            size_bytes=size_bytes,
            sha256=sha256,
        )

    def _acquire_verified(
        self,
        job_id: str,
        *,
        size_bytes: int,
        sha256: str,
    ) -> tuple[bool, ArtifactLease | None]:
        """Lease a process-verified hit without queueing behind another fill."""
        self._validate_metadata(size_bytes=size_bytes, sha256=sha256)
        path = self._path(job_id)
        descriptor = self._open(path)
        if descriptor is None:
            return True, None
        try:
            with self._state_lock:
                fingerprint = self._fingerprint(
                    os.fstat(descriptor),
                    size_bytes,
                    sha256,
                )
                if self._verified.get(job_id) != fingerprint:
                    os.close(descriptor)
                    return False, None
                os.utime(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                self._verified[job_id] = self._fingerprint(
                    os.fstat(descriptor),
                    size_bytes,
                    sha256,
                )
                self._active[job_id] = self._active.get(job_id, 0) + 1
            return True, ArtifactLease(
                descriptor,
                path=path,
                cache=self,
                job_id=job_id,
            )
        except BaseException:
            os.close(descriptor)
            raise

    def _path(self, job_id: str) -> Path:
        normalized = str(UUID(job_id))
        if normalized != job_id:
            raise ValueError("job_id must be a canonical UUID")
        return self.directory / f"{job_id}.zip"

    def acquire(
        self,
        job_id: str,
        *,
        size_bytes: int,
        sha256: str,
    ) -> ArtifactLease | None:
        """Hold the same verified descriptor that the response will stream."""
        self._validate_metadata(size_bytes=size_bytes, sha256=sha256)
        path = self._path(job_id)
        descriptor = self._open(path)
        if descriptor is None:
            return None
        try:
            file_stat = os.fstat(descriptor)
            fingerprint = self._fingerprint(file_stat, size_bytes, sha256)
            with self._state_lock:
                already_verified = self._verified.get(job_id) == fingerprint
            if not already_verified and not self._matches(
                descriptor,
                size_bytes=size_bytes,
                sha256=sha256,
            ):
                self._unlink_if_same(path, file_stat)
                with self._state_lock:
                    self._verified.pop(job_id, None)
                os.close(descriptor)
                return None
            os.utime(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            with self._state_lock:
                self._verified[job_id] = self._fingerprint(
                    os.fstat(descriptor), size_bytes, sha256
                )
                self._active[job_id] = self._active.get(job_id, 0) + 1
            return ArtifactLease(
                descriptor,
                path=path,
                cache=self,
                job_id=job_id,
            )
        except BaseException:
            os.close(descriptor)
            raise

    async def store(
        self,
        job_id: str,
        *,
        size_bytes: int,
        sha256: str,
        chunks: AsyncIterable[bytes],
    ) -> ArtifactLease:
        """Join one cancellation-safe, per-Job cache fill and lease its result."""
        self._validate_metadata(size_bytes=size_bytes, sha256=sha256)
        existing = await self.acquire_async(
            job_id,
            size_bytes=size_bytes,
            sha256=sha256,
        )
        if existing is not None:
            return existing

        with self._state_lock:
            if self._closed:
                raise RuntimeError("Artifact cache is closed")
            task = self._fill_tasks.get(job_id)
            if task is not None and task.done():
                self._fill_tasks.pop(job_id, None)
                task = None
            if task is None:
                task = asyncio.create_task(
                    self._fill(
                        job_id,
                        size_bytes=size_bytes,
                        sha256=sha256,
                        chunks=chunks,
                    ),
                    name=f"biomodals-artifact-fill-{job_id}",
                )
                self._fill_tasks[job_id] = task
                task.add_done_callback(
                    lambda completed, fill_job_id=job_id: self._fill_finished(
                        fill_job_id,
                        completed,
                    )
                )
        await asyncio.shield(task)
        lease = await self.acquire_async(
            job_id,
            size_bytes=size_bytes,
            sha256=sha256,
        )
        if lease is None:  # pragma: no cover - protected fill invariant
            raise ArtifactIntegrityError("Published cache entry disappeared")
        return lease

    async def _fill(
        self,
        job_id: str,
        *,
        size_bytes: int,
        sha256: str,
        chunks: AsyncIterable[bytes],
    ) -> None:
        """Stream one remote artifact into a private staging file."""
        temporary = self.staging_path(job_id)
        descriptor: int | None = None
        written = 0
        try:
            descriptor = os.open(temporary, _WRITE_FLAGS, 0o600)
            async for chunk in chunks:
                written += len(chunk)
                if written > size_bytes:
                    raise ArtifactIntegrityError(
                        "Downloaded artifact exceeded its recorded size"
                    )
                await self.run_bounded(self._write_all, descriptor, chunk)
            if written != size_bytes:
                raise ArtifactIntegrityError(
                    "Downloaded artifact failed its integrity check"
                )
            await self.run_bounded(
                self._publish_descriptor,
                job_id,
                temporary,
                descriptor,
                size_bytes,
                sha256,
                False,
            )
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def staging_path(self, job_id: str) -> Path:
        """Allocate a unique path counted as active Result staging."""
        self._path(job_id)
        return self.directory / f".{job_id}.{uuid4().hex}.part"

    async def publish_staged(
        self,
        job_id: str,
        path: Path,
        *,
        size_bytes: int,
        sha256: str,
    ) -> ArtifactLease:
        """Verify and atomically adopt a locally built Result archive."""
        self._validate_metadata(size_bytes=size_bytes, sha256=sha256)
        descriptor = self._open_staging(path)
        try:
            await self.run_bounded(
                self._publish_descriptor,
                job_id,
                path,
                descriptor,
                size_bytes,
                sha256,
                True,
            )
            return ArtifactLease(
                descriptor,
                path=self._path(job_id),
                cache=self,
                job_id=job_id,
            )
        except BaseException:
            os.close(descriptor)
            raise

    def _publish_descriptor(
        self,
        job_id: str,
        source: Path,
        descriptor: int,
        size_bytes: int,
        sha256: str,
        lease: bool,
    ) -> None:
        """Verify one staged descriptor and publish it without a path race."""
        if not self._matches(
            descriptor,
            size_bytes=size_bytes,
            sha256=sha256,
        ):
            raise ArtifactIntegrityError(
                "Downloaded artifact failed its integrity check"
            )
        os.fchmod(descriptor, 0o600)
        destination = self._path(job_id)
        with self._state_lock:
            os.replace(source, destination)
            os.utime(descriptor)
            self._verified[job_id] = self._fingerprint(
                os.fstat(descriptor),
                size_bytes,
                sha256,
            )
            if lease:
                self._active[job_id] = self._active.get(job_id, 0) + 1

    def _open_staging(self, path: Path) -> int:
        if path.parent != self.directory or not path.name.endswith(".part"):
            raise ValueError("Staged artifact is outside the Result cache")
        descriptor = self._open(path)
        if descriptor is None:
            raise ArtifactIntegrityError("Staged artifact is unavailable")
        return descriptor

    def _fill_finished(self, job_id: str, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()
        with self._state_lock:
            if self._fill_tasks.get(job_id) is task:
                self._fill_tasks.pop(job_id, None)

    def _open(self, path: Path) -> int | None:
        try:
            descriptor = os.open(path, _READ_FLAGS)
        except OSError as exc:
            if exc.errno in {errno.ENOENT, errno.ELOOP, errno.ENOTDIR}:
                return None
            raise
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            return None
        return descriptor

    def _matches(
        self,
        descriptor: int,
        *,
        size_bytes: int,
        sha256: str,
    ) -> bool:
        if os.fstat(descriptor).st_size != size_bytes:
            return False
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest() == sha256

    def _release(self, job_id: str) -> None:
        with self._state_lock:
            references = self._active.get(job_id, 0)
            if references <= 1:
                self._active.pop(job_id, None)
            else:
                self._active[job_id] = references - 1

    def _archives(self) -> list[tuple[Path, os.stat_result]]:
        archives: list[tuple[Path, os.stat_result]] = []
        with os.scandir(self.directory) as entries:
            for entry in entries:
                if not entry.name.endswith(".zip"):
                    continue
                file_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(file_stat.st_mode):
                    os.unlink(entry.path)
                elif stat.S_ISREG(file_stat.st_mode):
                    archives.append((Path(entry.path), file_stat))
        return archives

    def usage(self) -> CacheUsage:
        """Measure completed, staging, free, and currently reclaimable bytes."""
        archives = self._archives()
        staging: list[os.stat_result] = []
        with os.scandir(self.directory) as entries:
            for entry in entries:
                if not entry.name.endswith(".part"):
                    continue
                file_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(file_stat.st_mode):
                    staging.append(file_stat)
        with self._state_lock:
            protected = set(self._active) | set(self._fill_tasks)
        reclaimable = [
            file_stat for path, file_stat in archives if path.stem not in protected
        ]
        return CacheUsage(
            cached_entries=len(archives),
            cached_bytes=sum(item.st_size for _path, item in archives),
            staging_entries=len(staging),
            staging_bytes=sum(item.st_size for item in staging),
            free_bytes=disk_usage(self.directory).free,
            reclaimable_entries=len(reclaimable),
            reclaimable_bytes=sum(item.st_size for item in reclaimable),
        )

    async def usage_async(self) -> CacheUsage:
        """Measure cache storage on the bounded artifact worker."""
        return await self.run_bounded(self.usage)

    def clear(self) -> CacheCleanup:
        """Remove every unleased completed archive and report actual recovery."""
        entries = 0
        reclaimed = 0
        job_ids: list[str] = []
        for path, file_stat in self._archives():
            job_id = path.stem
            with self._state_lock:
                if self._active.get(job_id, 0) or job_id in self._fill_tasks:
                    continue
                if self._unlink_if_same(path, file_stat):
                    self._verified.pop(job_id, None)
                    entries += 1
                    reclaimed += file_stat.st_size
                    job_ids.append(job_id)
        return CacheCleanup(
            entries=entries,
            bytes=reclaimed,
            job_ids=tuple(job_ids),
        )

    async def clear_async(self) -> CacheCleanup:
        """Clear unleased entries on the bounded artifact worker."""
        return await self.run_bounded(self.clear)

    async def shutdown(self) -> None:
        """Finish queued artifact work and stop the single worker thread."""
        with self._state_lock:
            self._closed = True
            fills = tuple(self._fill_tasks.values())
        if fills:
            await asyncio.gather(*fills, return_exceptions=True)
        self._worker.shutdown(wait=True, cancel_futures=True)

    def cached_job_ids(self) -> set[str]:
        """Return canonical Job identifiers represented by local ZIP files."""
        return {path.stem for path, _file_stat in self._archives()}

    async def cached_job_ids_async(self) -> set[str]:
        """Scan cache membership on the bounded artifact worker."""
        return await self.run_bounded(self.cached_job_ids)

    @staticmethod
    def _write_all(descriptor: int, content: bytes) -> None:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]

    @staticmethod
    def _validate_metadata(*, size_bytes: int, sha256: str) -> None:
        if (
            type(size_bytes) is not int
            or size_bytes <= 0
            or not isinstance(sha256, str)
            or _SHA256_PATTERN.fullmatch(sha256) is None
        ):
            raise ArtifactIntegrityError("Invalid artifact metadata")

    @staticmethod
    def _fingerprint(
        file_stat: os.stat_result,
        size_bytes: int,
        sha256: str,
    ) -> tuple[int, str, int, int, int, int]:
        return (
            size_bytes,
            sha256,
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )

    @staticmethod
    def _unlink_if_same(path: Path, expected: os.stat_result) -> bool:
        try:
            current = path.lstat()
        except FileNotFoundError:
            return False
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            return False
        path.unlink()
        return True
