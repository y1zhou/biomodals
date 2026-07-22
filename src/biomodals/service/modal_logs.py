"""Modal CLI-backed live log streaming for attached function calls."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from datetime import datetime

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
        follow: bool,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> AsyncIterable[bytes]:
        """Open live or complete time-bounded logs for one FunctionCall."""
        if not app_name.strip() or not environment_name.strip():
            raise ValueError("Modal App and Environment names must not be empty")
        if not function_call_id.startswith("fc-"):
            raise ValueError("Modal FunctionCall ID must start with fc-")
        if follow and (since is not None or until is not None):
            raise ValueError("Live Modal logs cannot use a time range")
        if not follow and (
            since is None
            or until is None
            or since.tzinfo is None
            or until.tzinfo is None
            or since >= until
        ):
            raise ValueError("Historical Modal logs require a valid aware time range")
        environment = dict(os.environ)
        environment["NO_COLOR"] = "1"
        command = [
            sys.executable,
            "-m",
            "modal",
            "app",
            "logs",
            app_name,
        ]
        if follow:
            command.append("--follow")
        else:
            if since is None or until is None:  # pragma: no cover - validated above
                raise ValueError("Historical Modal logs require a time range")
            command.extend(("--since", since.isoformat(), "--until", until.isoformat()))
        command.extend((
            "--function-call",
            function_call_id,
            "--timestamps",
            "--env",
            environment_name,
        ))
        process = await self._process_factory(
            *command,
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
