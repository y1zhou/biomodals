"""Verified local cache for immutable final result archives."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import re
import stat
from collections.abc import AsyncIterable
from pathlib import Path
from uuid import UUID, uuid4

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW
_WRITE_FLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW


class ArtifactIntegrityError(RuntimeError):
    """Raised when Modal bytes do not match the recorded final artifact."""


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
    """Size-bounded LRU cache; Modal Volume storage remains authoritative."""

    def __init__(self, directory: Path, *, max_bytes: int) -> None:
        """Configure a cache directory and byte target."""
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        self.directory = directory
        self.max_bytes = max_bytes
        self._active: dict[str, int] = {}
        self._verified: dict[str, tuple[int, str, int, int, int, int]] = {}
        self._lock = asyncio.Lock()
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
        self._evict(exclude=set())

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
            if self._verified.get(job_id) != fingerprint and not self._matches(
                descriptor,
                size_bytes=size_bytes,
                sha256=sha256,
            ):
                self._unlink_if_same(path, file_stat)
                self._verified.pop(job_id, None)
                os.close(descriptor)
                return None
            os.utime(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
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
        """Download and verify an archive, caching it only when it fits."""
        self._validate_metadata(size_bytes=size_bytes, sha256=sha256)
        destination = self._path(job_id)
        async with self._lock:
            existing = self.acquire(
                job_id,
                size_bytes=size_bytes,
                sha256=sha256,
            )
            if existing is not None:
                return existing

            temporary = self.directory / f".{job_id}.{uuid4().hex}.part"
            digest = hashlib.sha256()
            written = 0
            descriptor: int | None = None
            try:
                descriptor = os.open(temporary, _WRITE_FLAGS, 0o600)
                with os.fdopen(descriptor, "wb", closefd=False) as output:
                    async for chunk in chunks:
                        written += len(chunk)
                        digest.update(chunk)
                        output.write(chunk)
                if written != size_bytes or digest.hexdigest() != sha256:
                    raise ArtifactIntegrityError(
                        "Downloaded artifact failed its integrity check"
                    )
                os.lseek(descriptor, 0, os.SEEK_SET)

                if size_bytes > self.max_bytes:
                    temporary.unlink()
                    lease = ArtifactLease(descriptor, path=None)
                else:
                    os.replace(temporary, destination)
                    os.utime(descriptor)
                    self._verified[job_id] = self._fingerprint(
                        os.fstat(descriptor), size_bytes, sha256
                    )
                    self._active[job_id] = self._active.get(job_id, 0) + 1
                    lease = ArtifactLease(
                        descriptor,
                        path=destination,
                        cache=self,
                        job_id=job_id,
                    )
                    self._evict(exclude={job_id})
                descriptor = None
                return lease
            except BaseException:
                if descriptor is not None:
                    os.close(descriptor)
                temporary.unlink(missing_ok=True)
                raise

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
        references = self._active.get(job_id, 0)
        if references <= 1:
            self._active.pop(job_id, None)
        else:
            self._active[job_id] = references - 1
        self._evict(exclude=set())

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

    def _evict(self, *, exclude: set[str]) -> None:
        archives = self._archives()
        total = sum(file_stat.st_size for _path, file_stat in archives)
        for path, file_stat in sorted(
            archives,
            key=lambda item: item[1].st_mtime,
        ):
            job_id = path.stem
            if total <= self.max_bytes:
                return
            if job_id in exclude or self._active.get(job_id, 0):
                continue
            if self._unlink_if_same(path, file_stat):
                self._verified.pop(job_id, None)
                total -= file_stat.st_size

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
