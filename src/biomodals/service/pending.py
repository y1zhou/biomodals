"""Small filesystem staging for requests admitted before provider launch."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

from biomodals.helper.artifacts import replace_bytes_atomic


class PendingRequestStore:
    """Retain immutable request bytes until remote staging is verified."""

    def __init__(self, state_directory: Path) -> None:
        """Place pending requests below the service state directory."""
        self.directory = state_directory / "pending-inputs"

    def initialize(self) -> None:
        """Create private pending request storage."""
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)

    def put(self, job_id: UUID, content: bytes) -> Path:
        """Create or verify one immutable pending request."""
        if not content:
            raise ValueError("Pending request cannot be empty")
        directory = self.directory / str(job_id)
        directory.mkdir(mode=0o700)
        path = directory / "request.bin"
        if path.exists():
            if path.read_bytes() != content:
                raise RuntimeError("Pending request conflicts with this Job")
        else:
            replace_bytes_atomic(path, content)
        return path

    def get(self, job_id: UUID) -> bytes | None:
        """Load one pending request when it remains locally available."""
        try:
            return (self.directory / str(job_id) / "request.bin").read_bytes()
        except FileNotFoundError:
            return None

    def delete(self, job_id: UUID) -> None:
        """Delete reconstructable local staging after verified remote upload."""
        shutil.rmtree(self.directory / str(job_id), ignore_errors=True)

    def cleanup_orphans(self, *, retained: set[UUID]) -> int:
        """Remove staging directories that no admitted Job still references."""
        removed = 0
        for path in self.directory.iterdir():
            try:
                job_id = UUID(path.name)
            except ValueError:
                job_id = None
            if job_id not in retained:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    def usage(self) -> tuple[int, int]:
        """Return pending request count and byte size."""
        files = list(self.directory.glob("*/request.bin"))
        return len(files), sum(path.stat().st_size for path in files)
