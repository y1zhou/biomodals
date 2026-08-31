"""Shared Modal Volume download contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path

from biomodals.helper.modal_volume import download_modal_volume_files


def test_download_modal_volume_files_limits_parallel_files(tmp_path: Path) -> None:
    """Use the configured limit across files and create parent directories."""
    active = 0
    maximum_active = 0

    class DownloadMethod:
        async def aio(self, path, handle, **kwargs):
            nonlocal active, maximum_active
            assert kwargs["download_semaphore"] is not None
            assert kwargs["rpc_semaphore"] is not None
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            written = handle.write(path.encode())
            active -= 1
            return written

    class Volume:
        _read_file_into_fileobj = DownloadMethod()

    downloads = [
        (f"remote/{index}", tmp_path / "nested" / str(index)) for index in range(5)
    ]

    download_modal_volume_files(Volume(), downloads, concurrency=2)

    assert maximum_active == 2
    assert [path.read_text() for _, path in downloads] == [
        f"remote/{index}" for index in range(5)
    ]


def test_download_modal_volume_files_bounds_live_tasks(tmp_path: Path) -> None:
    """Large lazy inputs create only the configured worker tasks."""
    maximum_tasks = 0

    class DownloadMethod:
        async def aio(self, path, handle, **_kwargs):
            nonlocal maximum_tasks
            maximum_tasks = max(maximum_tasks, len(asyncio.all_tasks()))
            await asyncio.sleep(0)
            return handle.write(path.encode())

    class Volume:
        _read_file_into_fileobj = DownloadMethod()

    downloads = ((f"remote/{index}", tmp_path / str(index)) for index in range(500))
    download_modal_volume_files(Volume(), downloads, concurrency=4)

    assert maximum_tasks <= 5
