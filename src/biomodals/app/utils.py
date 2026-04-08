"""Utility scripts for Biomodals apps."""

from pathlib import Path

import niquests


def run_command(cmd: list[str], **kwargs) -> None:
    """Run a shell command and stream output to stdout."""
    import subprocess as sp

    print(f"Running command: {' '.join(cmd)}")
    # Set default kwargs for sp.Popen
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    with sp.Popen(cmd, **kwargs) as p:  # noqa: S603
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        buffered_output = None
        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            print(buffered_output, end="", flush=True)

        if p.returncode != 0:
            raise sp.CalledProcessError(p.returncode, cmd, buffered_output)


def softlink_dir(src: str | Path, dst: str | Path) -> None:
    """Create a soft link from src to dst if dst does not exist."""
    src_path = Path(src)
    dst_path = Path(dst)
    if dst_path.exists():
        print(f"Destination path {dst} already exists. Skipping link creation.")
        return

    src_path.mkdir(parents=True, exist_ok=True)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.symlink_to(src_path, target_is_directory=True)


async def _download_file(session: niquests.AsyncSession, url: str, local_path: Path):
    """Download a file asynchronously using aiohttp."""
    import aiofiles

    try:
        response = await session.get(url, stream=True)
        response.raise_for_status()
        if not local_path.exists() or (
            int(response.headers["content-length"]) != local_path.stat().st_size
        ):
            async with aiofiles.open(local_path, "wb") as f:
                async for chunk in await response.iter_content():
                    await f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Download for {url} to {local_path} failed.") from e


async def _download_files(
    urls: dict[str, str | Path],
    force: bool = False,
    max_connected_hosts: int = 10,
    max_connections: int = 20,
    num_retries: int = 1,
    progress_bar_desc: str | None = None,
):
    """Download multiple files concurrently using aiohttp.

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
            if force or not local_path.exists():
                tasks.append(_download_file(session, url, local_path))

        # run all of the downloads and await their completion
        await tqdm_asyncio.gather(*tasks, desc=progress_bar_desc)


def download_files(
    urls: dict[str, str | Path],
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
