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

    path = asyncio.run(
        cache.store(
            "11111111-1111-4111-8111-111111111111",
            size_bytes=len(content),
            sha256=digest(content),
            chunks=chunks(content),
        )
    )

    assert path is not None
    assert path.read_bytes() == content
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


def test_oversized_result_bypasses_cache_without_consuming_source(
    tmp_path: Path,
) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=3)
    consumed = False

    async def source():
        nonlocal consumed
        consumed = True
        yield b"large"

    result = asyncio.run(
        cache.store(
            "11111111-1111-4111-8111-111111111111",
            size_bytes=5,
            sha256=digest(b"large"),
            chunks=source(),
        )
    )

    assert result is None
    assert consumed is False


def test_cache_evicts_least_recently_used_inactive_result(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=8)
    first_content = b"first"
    second_content = b"next"
    first = asyncio.run(
        cache.store(
            "11111111-1111-4111-8111-111111111111",
            size_bytes=len(first_content),
            sha256=digest(first_content),
            chunks=chunks(first_content),
        )
    )
    assert first is not None
    os.utime(first, (1, 1))

    second = asyncio.run(
        cache.store(
            "22222222-2222-4222-8222-222222222222",
            size_bytes=len(second_content),
            sha256=digest(second_content),
            chunks=chunks(second_content),
        )
    )

    assert second is not None and second.exists()
    assert not first.exists()


def test_cache_does_not_evict_an_active_download(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path, max_bytes=8)
    first_content = b"first"
    first = asyncio.run(
        cache.store(
            "11111111-1111-4111-8111-111111111111",
            size_bytes=len(first_content),
            sha256=digest(first_content),
            chunks=chunks(first_content),
        )
    )
    assert first is not None
    assert cache.acquire("11111111-1111-4111-8111-111111111111") == first

    second_content = b"next"
    second = asyncio.run(
        cache.store(
            "22222222-2222-4222-8222-222222222222",
            size_bytes=len(second_content),
            sha256=digest(second_content),
            chunks=chunks(second_content),
        )
    )

    assert second is not None
    assert first.exists()
    cache.release("11111111-1111-4111-8111-111111111111")
