"""Modal CLI-backed live log streaming for attached function calls."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable

LOGGER = logging.getLogger(__name__)
_READ_CHUNK_BYTES = 64 * 1024
_PROCESS_STOP_TIMEOUT_SECONDS = 5
ProcessFactory = Callable[..., Awaitable[asyncio.subprocess.Process]]


class ModalCLILogStreamer:
    """Open one supported ``modal app logs`` stream filtered by call ID."""

    def __init__(
        self,
        *,
        process_factory: ProcessFactory = asyncio.create_subprocess_exec,
    ) -> None:
        """Allow process creation to be replaced in focused tests."""
        self._process_factory = process_factory

    async def open(
        self,
        *,
        app_name: str,
        environment_name: str,
        function_call_id: str,
    ) -> AsyncIterable[bytes]:
        """Start following one FunctionCall without exposing its ID over HTTP."""
        if not app_name.strip() or not environment_name.strip():
            raise ValueError("Modal App and Environment names must not be empty")
        if not function_call_id.startswith("fc-"):
            raise ValueError("Modal FunctionCall ID must start with fc-")
        environment = dict(os.environ)
        environment["NO_COLOR"] = "1"
        process = await self._process_factory(
            sys.executable,
            "-m",
            "modal",
            "app",
            "logs",
            app_name,
            "--follow",
            "--function-call",
            function_call_id,
            "--timestamps",
            "--env",
            environment_name,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        stdout = process.stdout
        if stdout is None:  # pragma: no cover - PIPE guarantees stdout
            await _stop_process(process)
            raise OSError("Modal log process did not expose stdout")
        return _read_process(process, stdout)


async def _read_process(
    process: asyncio.subprocess.Process,
    stdout: asyncio.StreamReader,
) -> AsyncIterator[bytes]:
    """Yield bounded chunks and stop the CLI when the HTTP consumer leaves."""
    try:
        while chunk := await stdout.read(_READ_CHUNK_BYTES):
            yield chunk
        return_code = await process.wait()
        if return_code != 0:
            LOGGER.warning("Modal log process exited with status %s", return_code)
    finally:
        await _stop_process(process)


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=_PROCESS_STOP_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
