"""Focused contract tests for the Modal CLI log-stream boundary."""

# ruff: noqa: D101,D102,D103,D107,S101

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest

from biomodals.service.modal_logs import ModalCLILogStreamer


class FakeProcess:
    def __init__(self) -> None:
        self.stdout = asyncio.StreamReader()
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._finished = asyncio.Event()

    async def wait(self) -> int:
        await self._finished.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0
        self._finished.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._finished.set()


def test_modal_cli_stream_filters_one_call_and_stops_on_disconnect() -> None:
    async def scenario() -> None:
        process = FakeProcess()
        command: tuple[str, ...] = ()
        options: dict[str, Any] = {}

        async def create_process(*args: str, **kwargs: Any) -> Any:
            nonlocal command, options
            command = args
            options = kwargs
            return process

        streamer = ModalCLILogStreamer(process_factory=create_process)
        opened = await streamer.open(
            app_name="GromacsApp",
            environment_name="production",
            function_call_id="fc-example",
        )
        stream = cast(AsyncGenerator[bytes, None], opened)
        process.stdout.feed_data(b"remote output\n")

        assert await anext(stream) == b"remote output\n"
        assert command[1:] == (
            "-m",
            "modal",
            "app",
            "logs",
            "GromacsApp",
            "--follow",
            "--function-call",
            "fc-example",
            "--timestamps",
            "--env",
            "production",
        )
        assert options["env"]["NO_COLOR"] == "1"
        assert options["stderr"] == asyncio.subprocess.STDOUT

        await stream.aclose()

        assert process.terminated
        assert not process.killed

    asyncio.run(scenario())


def test_modal_cli_stream_rejects_non_function_call_ids() -> None:
    async def scenario() -> None:
        streamer = ModalCLILogStreamer()
        with pytest.raises(ValueError, match="must start with fc-"):
            await streamer.open(
                app_name="GromacsApp",
                environment_name="production",
                function_call_id="not-a-call",
            )

    asyncio.run(scenario())
