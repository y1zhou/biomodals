"""Final-result cache contracts."""

# ruff: noqa: D101,D102,D103

import asyncio
import hashlib
import os
import stat
from pathlib import Path

import pytest

from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError


async def chunks(content: bytes):
    midpoint = len(content) // 2
    yield content[:midpoint]
    yield content[midpoint:]


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_cache_verifies_download_before_atomic_publish(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=100)
    content = b"valid zip bytes"

    lease = asyncio.run(
        cache.store(
            "11111111-1111-4111-8111-111111111111",
            size_bytes=len(content),
            sha256=digest(content),
            chunks=chunks(content),
        )
    )

    path = lease.path
    assert path is not None
    assert lease.read(len(content)) == content
    lease.close()
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(tmp_path.glob("*.part")) == []


def test_cache_discards_corrupt_download(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=100)

    with pytest.raises(ArtifactIntegrityError):
        asyncio.run(
            cache.store(
                "11111111-1111-4111-8111-111111111111",
                size_bytes=5,
                sha256=digest(b"other"),
                chunks=chunks(b"wrong"),
            )
        )

    assert list(tmp_path.iterdir()) == []


def test_cache_revalidates_an_existing_file_before_serving(tmp_path: Path) -> None:
    job_id = "11111111-1111-4111-8111-111111111111"
    path = tmp_path / f"{job_id}.zip"
    path.write_bytes(b"corrupt")
    cache = ArtifactCache(tmp_path, max_bytes=100)

    assert (
        cache.acquire(
            job_id,
            size_bytes=len(b"expected"),
            sha256=digest(b"expected"),
        )
        is None
    )
    assert not path.exists()


def test_oversized_result_is_verified_without_becoming_cached(
    tmp_path: Path,
) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=3)
    consumed = False

    async def source():
        nonlocal consumed
        consumed = True
        yield b"large"

    lease = asyncio.run(
        cache.store(
            "11111111-1111-4111-8111-111111111111",
            size_bytes=5,
            sha256=digest(b"large"),
            chunks=source(),
        )
    )

    assert consumed is True
    assert lease.path is None
    assert lease.read(5) == b"large"
    assert list(tmp_path.iterdir()) == []
    lease.close()


def test_oversized_corrupt_result_is_rejected(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=3)

    with pytest.raises(ArtifactIntegrityError):
        asyncio.run(
            cache.store(
                "11111111-1111-4111-8111-111111111111",
                size_bytes=5,
                sha256=digest(b"other"),
                chunks=chunks(b"large"),
            )
        )

    assert list(tmp_path.iterdir()) == []


def test_cache_evicts_least_recently_used_inactive_result(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=8)
    first_content = b"first"
    second_content = b"next"
    first_lease = asyncio.run(
        cache.store(
            "11111111-1111-4111-8111-111111111111",
            size_bytes=len(first_content),
            sha256=digest(first_content),
            chunks=chunks(first_content),
        )
    )
    first = first_lease.path
    assert first is not None
    first_lease.close()
    os.utime(first, (1, 1))

    second_lease = asyncio.run(
        cache.store(
            "22222222-2222-4222-8222-222222222222",
            size_bytes=len(second_content),
            sha256=digest(second_content),
            chunks=chunks(second_content),
        )
    )

    second = second_lease.path
    assert second is not None and second.exists()
    assert not first.exists()
    second_lease.close()


def test_cache_does_not_evict_an_active_download(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=8)
    first_content = b"first"
    first_lease = asyncio.run(
        cache.store(
            "11111111-1111-4111-8111-111111111111",
            size_bytes=len(first_content),
            sha256=digest(first_content),
            chunks=chunks(first_content),
        )
    )
    first = first_lease.path
    assert first is not None

    second_content = b"next"
    second_lease = asyncio.run(
        cache.store(
            "22222222-2222-4222-8222-222222222222",
            size_bytes=len(second_content),
            sha256=digest(second_content),
            chunks=chunks(second_content),
        )
    )

    second = second_lease.path
    assert second is not None
    assert first.exists()
    first_lease.close()
    assert not first.exists()
    second_lease.close()


def test_cache_refuses_symlinks_without_touching_their_target(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=100)
    job_id = "11111111-1111-4111-8111-111111111111"
    target = tmp_path / "secret"
    target.write_bytes(b"expected")
    (tmp_path / f"{job_id}.zip").symlink_to(target)

    assert (
        cache.acquire(
            job_id,
            size_bytes=len(b"expected"),
            sha256=digest(b"expected"),
        )
        is None
    )
    assert target.read_bytes() == b"expected"


def test_lease_streams_verified_descriptor_after_path_replacement(
    tmp_path: Path,
) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=100)
    job_id = "11111111-1111-4111-8111-111111111111"
    content = b"verified"
    lease = asyncio.run(
        cache.store(
            job_id,
            size_bytes=len(content),
            sha256=digest(content),
            chunks=chunks(content),
        )
    )
    path = lease.path
    assert path is not None
    target = tmp_path / "secret"
    target.write_bytes(b"not verified")
    path.unlink()
    path.symlink_to(target)

    assert lease.read(len(content)) == content
    lease.close()
    assert target.read_bytes() == b"not verified"


@pytest.mark.parametrize(
    ("size_bytes", "sha256"),
    [
        (0, digest(b"x")),
        (-1, digest(b"x")),
        (True, digest(b"x")),
        (1, digest(b"x").upper()),
        (1, "g" * 64),
        (1, "0" * 63),
    ],
)
def test_cache_rejects_invalid_metadata(
    tmp_path: Path,
    size_bytes: int,
    sha256: str,
) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=100)

    with pytest.raises(ArtifactIntegrityError, match="Invalid artifact metadata"):
        asyncio.run(
            cache.store(
                "11111111-1111-4111-8111-111111111111",
                size_bytes=size_bytes,
                sha256=sha256,
                chunks=chunks(b"x"),
            )
        )
