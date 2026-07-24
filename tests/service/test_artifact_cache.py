"""Final-result cache contracts."""

# ruff: noqa: D101,D102,D103

import asyncio
import hashlib
import os
import stat
from pathlib import Path
from threading import Event

import pytest

from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError


async def chunks(content: bytes):
    midpoint = len(content) // 2
    yield content[:midpoint]
    yield content[midpoint:]


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_bounded_io_finishes_before_cancellation_unwinds(tmp_path: Path) -> None:
    async def scenario() -> None:
        cache = ArtifactCache(tmp_path)
        started = Event()
        release = Event()
        finished = Event()

        def blocking_operation() -> None:
            started.set()
            release.wait(timeout=5)
            finished.set()

        task = asyncio.create_task(cache.run_bounded(blocking_operation))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0)
        cancellation_waits_for_io = not task.done()
        release.set()
        [outcome] = await asyncio.gather(task, return_exceptions=True)
        await cache.shutdown()

        assert cancellation_waits_for_io is True
        assert isinstance(outcome, asyncio.CancelledError)
        assert finished.is_set()

    asyncio.run(scenario())


def test_cache_verifies_download_before_atomic_publish(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
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
    cache = ArtifactCache(tmp_path)

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
    cache = ArtifactCache(tmp_path)

    assert (
        cache.acquire(
            job_id,
            size_bytes=len(b"expected"),
            sha256=digest(b"expected"),
        )
        is None
    )
    assert not path.exists()


def test_result_size_does_not_trigger_automatic_eviction(
    tmp_path: Path,
) -> None:
    cache = ArtifactCache(tmp_path)
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
    assert lease.path is not None
    assert lease.read(5) == b"large"
    assert [path.name for path in tmp_path.iterdir()] == [
        "11111111-1111-4111-8111-111111111111.zip"
    ]
    lease.close()


def test_oversized_corrupt_result_is_rejected(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)

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


def test_cache_keeps_completed_results_until_explicit_cleanup(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
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
    assert first.exists()
    second_lease.close()


def test_explicit_cleanup_protects_an_active_download(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
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
    second_lease.close()
    assert first.exists()
    cleanup = cache.clear()
    assert first.exists()
    assert not second.exists()
    assert cleanup.entries == 1
    first_lease.close()
    assert first.exists()


def test_explicit_cleanup_protects_a_prepared_download_until_acquired(
    tmp_path: Path,
) -> None:
    cache = ArtifactCache(tmp_path)
    job_id = "11111111-1111-4111-8111-111111111111"
    content = b"prepared"
    lease = asyncio.run(
        cache.store(
            job_id,
            size_bytes=len(content),
            sha256=digest(content),
            chunks=chunks(content),
        )
    )
    lease.close()

    cache.protect_prepared(job_id)
    assert cache.clear().entries == 0

    download = cache.acquire(
        job_id,
        size_bytes=len(content),
        sha256=digest(content),
    )
    assert download is not None
    download.close()
    assert cache.clear().entries == 1


def test_cleanup_cannot_remove_completed_fill_before_first_lease(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        cache = ArtifactCache(tmp_path)
        job_id = "11111111-1111-4111-8111-111111111111"
        content = b"published"
        second_acquire_started = asyncio.Event()
        release_second_acquire = asyncio.Event()
        original_acquire = cache.acquire_async
        acquire_calls = 0

        async def delayed_acquire(*args, **kwargs):
            nonlocal acquire_calls
            acquire_calls += 1
            if acquire_calls == 2:
                second_acquire_started.set()
                await release_second_acquire.wait()
            return await original_acquire(*args, **kwargs)

        monkeypatch.setattr(cache, "acquire_async", delayed_acquire)
        stored = asyncio.create_task(
            cache.store(
                job_id,
                size_bytes=len(content),
                sha256=digest(content),
                chunks=chunks(content),
            )
        )
        await second_acquire_started.wait()

        assert (await cache.clear_async()).entries == 0

        release_second_acquire.set()
        lease = await stored
        assert lease.read(len(content)) == content
        lease.close()
        assert (await cache.clear_async()).entries == 1
        await cache.shutdown()

    asyncio.run(scenario())


def test_cancelled_waiter_does_not_cancel_shared_cache_fill(tmp_path: Path) -> None:
    async def scenario() -> None:
        cache = ArtifactCache(tmp_path)
        content = b"first-second"
        started = asyncio.Event()
        release = asyncio.Event()
        fallback_consumed = False

        async def source():
            yield b"first-"
            started.set()
            await release.wait()
            yield b"second"

        async def fallback():
            nonlocal fallback_consumed
            fallback_consumed = True
            yield content

        first = asyncio.create_task(
            cache.store(
                "11111111-1111-4111-8111-111111111111",
                size_bytes=len(content),
                sha256=digest(content),
                chunks=source(),
            )
        )
        await started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(
            cache.store(
                "11111111-1111-4111-8111-111111111111",
                size_bytes=len(content),
                sha256=digest(content),
                chunks=fallback(),
            )
        )
        await asyncio.sleep(0)
        release.set()
        lease = await second

        assert lease.read(len(content)) == content
        assert fallback_consumed is False
        lease.close()
        await cache.shutdown()

    asyncio.run(scenario())


def test_failed_shared_fill_can_be_rebuilt_while_other_waiters_unwind(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        cache = ArtifactCache(tmp_path)
        job_id = "11111111-1111-4111-8111-111111111111"
        expected = b"recovered"
        started = asyncio.Event()
        release = asyncio.Event()

        async def corrupt_source():
            started.set()
            await release.wait()
            yield b"corrupt"

        first = asyncio.create_task(
            cache.store(
                job_id,
                size_bytes=len(expected),
                sha256=digest(expected),
                chunks=corrupt_source(),
            )
        )
        await started.wait()
        second = asyncio.create_task(
            cache.store(
                job_id,
                size_bytes=len(expected),
                sha256=digest(expected),
                chunks=chunks(b"unused"),
            )
        )
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        assert all(isinstance(result, ArtifactIntegrityError) for result in results)

        lease = await cache.store(
            job_id,
            size_bytes=len(expected),
            sha256=digest(expected),
            chunks=chunks(expected),
        )
        assert lease.read(len(expected)) == expected
        lease.close()
        await cache.shutdown()

    asyncio.run(scenario())


def test_cache_refuses_symlinks_without_touching_their_target(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
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
    cache = ArtifactCache(tmp_path)
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
    cache = ArtifactCache(tmp_path)

    with pytest.raises(ArtifactIntegrityError, match="Invalid artifact metadata"):
        asyncio.run(
            cache.store(
                "11111111-1111-4111-8111-111111111111",
                size_bytes=size_bytes,
                sha256=sha256,
                chunks=chunks(b"x"),
            )
        )
