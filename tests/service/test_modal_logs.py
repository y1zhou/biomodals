"""Focused contract tests for the Modal CLI log-stream boundary."""

# ruff: noqa: D101,D102,D103,D107,S101

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, MutableMapping
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from starlette.requests import ClientDisconnect

from biomodals.service.admin_jobs_api import (
    _ClosingStreamingResponse,
    _redact_provider_call_id,
)
from biomodals.service.modal_logs import ModalCLILogSource


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

        streamer = ModalCLILogSource(process_factory=create_process)
        opened = await streamer.open(
            app_name="GromacsApp",
            environment_name="production",
            function_call_id="fc-example",
            follow=True,
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
        streamer = ModalCLILogSource()
        with pytest.raises(ValueError, match="must start with fc-"):
            await streamer.open(
                app_name="GromacsApp",
                environment_name="production",
                function_call_id="not-a-call",
                follow=True,
            )

    asyncio.run(scenario())


def test_modal_cli_fetches_complete_historical_call_range() -> None:
    async def scenario() -> None:
        process = FakeProcess()
        command: tuple[str, ...] = ()

        async def create_process(*args: str, **_kwargs: Any) -> Any:
            nonlocal command
            command = args
            process.stdout.feed_eof()
            process.returncode = 0
            process._finished.set()
            return process

        streamer = ModalCLILogSource(process_factory=create_process)
        stream = await streamer.open(
            app_name="GromacsApp",
            environment_name="production",
            function_call_id="fc-example",
            follow=False,
            since=datetime(2026, 7, 22, 1, 2, 3, tzinfo=UTC),
            until=datetime(2026, 7, 22, 1, 4, 5, tzinfo=UTC),
        )

        assert [chunk async for chunk in stream] == []
        assert command[1:] == (
            "-m",
            "modal",
            "app",
            "logs",
            "GromacsApp",
            "--since",
            "2026-07-22T01:02:03+00:00",
            "--until",
            "2026-07-22T01:04:05+00:00",
            "--function-call",
            "fc-example",
            "--timestamps",
            "--env",
            "production",
        )

    asyncio.run(scenario())


def test_modal_cli_reports_nonzero_process_exit() -> None:
    async def scenario() -> None:
        process = FakeProcess()

        async def create_process(*_args: str, **_kwargs: Any) -> Any:
            process.stdout.feed_data(b"Modal could not read this Function Call\n")
            process.stdout.feed_eof()
            process.returncode = 1
            process._finished.set()
            return process

        streamer = ModalCLILogSource(process_factory=create_process)
        opened = await streamer.open(
            app_name="GromacsApp",
            environment_name="production",
            function_call_id="fc-example",
            follow=True,
        )
        stream = cast(AsyncGenerator[bytes, None], opened)

        assert await anext(stream) == b"Modal could not read this Function Call\n"
        with pytest.raises(OSError, match="exited with status 1"):
            await anext(stream)

    asyncio.run(scenario())


def test_log_redaction_closes_source_when_consumer_disconnects() -> None:
    async def scenario() -> None:
        source_closed = False

        async def source() -> AsyncGenerator[bytes, None]:
            nonlocal source_closed
            try:
                yield b"visible output before a long-running call\n"
                await asyncio.Event().wait()
            finally:
                source_closed = True

        redacted = cast(
            AsyncGenerator[bytes, None],
            _redact_provider_call_id(source(), "fc-secret"),
        )

        assert await anext(redacted) == b"visible output before a long-running call\n"
        await redacted.aclose()

        assert source_closed

    asyncio.run(scenario())


def test_streaming_response_closes_body_after_client_disconnect() -> None:
    async def scenario() -> None:
        source_closed = False

        async def source() -> AsyncGenerator[bytes, None]:
            nonlocal source_closed
            try:
                yield b"one log chunk\n"
                await asyncio.Event().wait()
            finally:
                source_closed = True

        response = _ClosingStreamingResponse(source(), media_type="text/plain")

        async def receive() -> dict[str, str]:
            return {"type": "http.disconnect"}

        async def send(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        with pytest.raises(ClientDisconnect):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}},
                receive,
                send,
            )

        assert source_closed

    asyncio.run(scenario())
