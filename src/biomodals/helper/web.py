"""Helper functions for web-related operations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import niquests


async def _download_files(
    urls: Mapping[str, str | Path],
    force: bool = False,
    max_connected_hosts: int = 10,
    max_connections: int = 20,
    num_retries: int = 1,
    progress_bar_desc: str | None = None,
):
    """Download multiple files concurrently.

    Args:
        urls: Keys are URLs, and values are local file paths.
        force: Whether to overwrite existing files.
        max_connected_hosts: Concurrent hosts to be kept alive by a session.
        max_connections: Limit concurrent downloads per host to be civil.
        num_retries: Number of times to retry failed downloads.
        progress_bar_desc: Optional description for the progress bar.

    """
    from tqdm.asyncio import tqdm_asyncio

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.3"
    }

    # launch downloads concurrently
    # https://niquests.readthedocs.io/en/latest/user/quickstart.html#scale-your-session-pool
    async with niquests.AsyncSession(
        headers=headers,
        retries=num_retries,
        pool_connections=max_connected_hosts,
        pool_maxsize=max_connections,
    ) as session:
        tasks = []
        for url, local_file in urls.items():
            local_path = Path(local_file)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append(_download_file(session, url, local_path, force))

        # run all of the downloads and await their completion
        await tqdm_asyncio.gather(*tasks, desc=progress_bar_desc)


async def _download_file(
    session: niquests.AsyncSession, url: str, local_path: Path, force: bool
):
    """Download a file asynchronously."""
    import aiofiles

    try:
        if not await _should_download(session, url, local_path, force):
            return

        response = None
        try:
            response = await session.get(url, stream=True)
            response.raise_for_status()
            async with aiofiles.open(local_path, "wb") as f:
                async for chunk in await response.iter_content():
                    await f.write(chunk)
        finally:
            if response is not None:
                await response.close()
    except Exception as e:
        raise RuntimeError(f"Download for {url} to {local_path} failed.") from e


async def _should_download(
    session: niquests.AsyncSession, url: str, local_path: Path, force: bool
) -> bool:
    """Return whether a remote URL should be downloaded."""
    if force or not local_path.exists():
        return True
    try:
        remote_size = await _remote_content_length(session, url)
    except Exception:
        return False
    return remote_size is not None and remote_size != local_path.stat().st_size


async def _remote_content_length(
    session: niquests.AsyncSession, url: str
) -> int | None:
    """Return a URL's content length from HEAD metadata when available."""
    response = None
    try:
        response = cast(
            niquests.AsyncResponse,
            await session.head(url, allow_redirects=True),
        )
        response.raise_for_status()
        if "content-length" not in response.headers:
            return None
        return int(response.headers["content-length"])
    finally:
        if response is not None:
            await response.close()


def download_files(
    urls: Mapping[str, str | Path],
    force: bool = False,
    max_connected_hosts: int = 10,
    max_connections: int = 20,
    num_retries: int = 1,
    progress_bar_desc: str | None = None,
):
    """Download files synchronously via _download_files."""
    import asyncio

    asyncio.run(
        _download_files(
            urls=urls,
            force=force,
            max_connected_hosts=max_connected_hosts,
            max_connections=max_connections,
            num_retries=num_retries,
            progress_bar_desc=progress_bar_desc,
        )
    )
