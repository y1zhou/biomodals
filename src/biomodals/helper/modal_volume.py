"""Shared Modal Volume transfer helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast


def download_modal_volume_files(
    volume: object,
    downloads: Iterable[tuple[str, Path]],
    *,
    concurrency: int,
) -> None:
    """Download selected Volume files with one shared concurrency limit."""
    if isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    asyncio.run(_download_modal_volume_files(volume, downloads, concurrency))


async def _download_modal_volume_files(
    volume: object,
    downloads: Iterable[tuple[str, Path]],
    concurrency: int,
) -> None:
    download_semaphore = asyncio.Semaphore(concurrency)
    rpc_semaphore = asyncio.Semaphore(concurrency)
    read_file = cast(Any, volume)._read_file_into_fileobj.aio
    pending = iter(downloads)

    async def download(remote_path: str, destination: Path) -> None:
        complete = False
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                await read_file(
                    remote_path,
                    handle,
                    download_semaphore=download_semaphore,
                    rpc_semaphore=rpc_semaphore,
                )
            complete = True
        finally:
            if not complete:
                destination.unlink(missing_ok=True)

    async def worker() -> None:
        while True:
            try:
                remote_path, destination = next(pending)
            except StopIteration:
                return
            await download(remote_path, destination)

    async with asyncio.TaskGroup() as tasks:
        for _ in range(concurrency):
            tasks.create_task(worker())
