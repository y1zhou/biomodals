"""Verified local cache for immutable final result archives."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncIterable
from pathlib import Path
from uuid import UUID, uuid4


class ArtifactIntegrityError(RuntimeError):
    """Raised when Modal bytes do not match the recorded final artifact."""


class ArtifactCache:
    """Size-bounded LRU cache; Modal Volume storage remains authoritative."""

    def __init__(self, directory: Path, *, max_bytes: int) -> None:
        """Configure a cache directory and byte target."""
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        self.directory = directory
        self.max_bytes = max_bytes
        self._active: dict[str, int] = {}
        self._verified: dict[str, tuple[int, str, int]] = {}
        self._lock = asyncio.Lock()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink():
            raise RuntimeError("Artifact cache directory must not be a symbolic link")
        self.directory.chmod(0o700)
        for archive in self.directory.glob("*.zip"):
            if archive.is_symlink():
                archive.unlink()
            else:
                archive.chmod(0o600)
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
        size_bytes: int | None = None,
        sha256: str | None = None,
    ) -> Path | None:
        """Hold an existing archive so eviction cannot remove it mid-response."""
        path = self._path(job_id)
        if not path.is_file() or path.is_symlink():
            return None
        if size_bytes is not None and sha256 is not None:
            stat = path.stat()
            expected = (size_bytes, sha256, stat.st_mtime_ns)
            if self._verified.get(job_id) != expected:
                if not self._matches(path, size_bytes=size_bytes, sha256=sha256):
                    path.unlink(missing_ok=True)
                    self._verified.pop(job_id, None)
                    return None
                self._verified[job_id] = expected
        self._active[job_id] = self._active.get(job_id, 0) + 1
        path.touch()
        if size_bytes is not None and sha256 is not None:
            self._verified[job_id] = (size_bytes, sha256, path.stat().st_mtime_ns)
        return path

    def release(self, job_id: str) -> None:
        """Release one active response reference."""
        references = self._active.get(job_id, 0)
        if references <= 1:
            self._active.pop(job_id, None)
        else:
            self._active[job_id] = references - 1
        self._evict(exclude=set())

    async def store(
        self,
        job_id: str,
        *,
        size_bytes: int,
        sha256: str,
        chunks: AsyncIterable[bytes],
    ) -> Path | None:
        """Verify and atomically cache one archive, or bypass an oversized one."""
        if size_bytes > self.max_bytes:
            return None
        if size_bytes < 0 or len(sha256) != 64:
            raise ArtifactIntegrityError("Invalid artifact metadata")

        destination = self._path(job_id)
        async with self._lock:
            if destination.is_symlink():
                destination.unlink()
            if destination.is_file() and self._matches(
                destination, size_bytes=size_bytes, sha256=sha256
            ):
                destination.touch()
                self._verified[job_id] = (
                    size_bytes,
                    sha256,
                    destination.stat().st_mtime_ns,
                )
                return destination
            destination.unlink(missing_ok=True)

            temporary = self.directory / f".{job_id}.{uuid4().hex}.part"
            digest = hashlib.sha256()
            written = 0
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as output:
                    async for chunk in chunks:
                        written += len(chunk)
                        digest.update(chunk)
                        output.write(chunk)
                if written != size_bytes or digest.hexdigest() != sha256:
                    raise ArtifactIntegrityError(
                        "Downloaded artifact failed its integrity check"
                    )
                os.replace(temporary, destination)
                self._verified[job_id] = (
                    size_bytes,
                    sha256,
                    destination.stat().st_mtime_ns,
                )
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise

            self._evict(exclude={job_id})
            return destination

    def _matches(
        self,
        path: Path,
        *,
        size_bytes: int,
        sha256: str,
    ) -> bool:
        if path.stat().st_size != size_bytes:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as cached:
            while chunk := cached.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest() == sha256

    def _evict(self, *, exclude: set[str]) -> None:
        archives = list(self.directory.glob("*.zip"))
        total = sum(path.stat().st_size for path in archives)
        for path in sorted(archives, key=lambda item: item.stat().st_mtime):
            job_id = path.stem
            if total <= self.max_bytes:
                return
            if job_id in exclude or self._active.get(job_id, 0):
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            self._verified.pop(job_id, None)
            total -= size
