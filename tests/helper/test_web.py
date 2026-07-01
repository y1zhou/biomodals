"""Tests for web download helpers."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import niquests

from biomodals.helper import web


class FakeResponse:
    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        chunks: tuple[bytes, ...] = (),
        status_error: Exception | None = None,
    ) -> None:
        self.headers = headers or {}
        self.chunks = chunks
        self.status_error = status_error
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    async def iter_content(self):
        async def chunks():
            for chunk in self.chunks:
                yield chunk

        return chunks()

    async def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(
        self,
        *,
        head_response: FakeResponse | None = None,
        head_error: Exception | None = None,
        get_response: FakeResponse | None = None,
    ) -> None:
        self.head_response = head_response
        self.head_error = head_error
        self.get_response = get_response
        self.calls: list[tuple[str, str, dict]] = []

    async def head(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append(("HEAD", url, kwargs))
        if self.head_error is not None:
            raise self.head_error
        if self.head_response is None:
            raise AssertionError("unexpected HEAD request")
        return self.head_response

    async def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        if self.get_response is None:
            raise AssertionError("unexpected GET request")
        return self.get_response


def test_download_file_uses_head_size_check_for_cached_file(tmp_path: Path) -> None:
    output = tmp_path / "model.bin"
    output.write_bytes(b"cached")
    head_response = FakeResponse(headers={"content-length": "6"})
    session = FakeSession(head_response=head_response)

    asyncio.run(
        web._download_file(
            cast(niquests.AsyncSession, session),
            "https://example.test/model.bin",
            output,
            force=False,
        )
    )

    assert output.read_bytes() == b"cached"
    assert session.calls == [
        ("HEAD", "https://example.test/model.bin", {"allow_redirects": True})
    ]
    assert head_response.closed is True


def test_download_file_trusts_cached_file_when_head_fails(tmp_path: Path) -> None:
    output = tmp_path / "model.bin"
    output.write_bytes(b"cached")
    session = FakeSession(head_error=RuntimeError("HEAD failed"))

    asyncio.run(
        web._download_file(
            cast(niquests.AsyncSession, session),
            "https://example.test/model.bin",
            output,
            force=False,
        )
    )

    assert output.read_bytes() == b"cached"
    assert session.calls == [
        ("HEAD", "https://example.test/model.bin", {"allow_redirects": True})
    ]


def test_download_file_closes_head_and_get_when_cached_size_differs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "model.bin"
    output.write_bytes(b"old")
    head_response = FakeResponse(headers={"content-length": "6"})
    get_response = FakeResponse(chunks=(b"new", b"bin"))
    session = FakeSession(head_response=head_response, get_response=get_response)

    asyncio.run(
        web._download_file(
            cast(niquests.AsyncSession, session),
            "https://example.test/model.bin",
            output,
            force=False,
        )
    )

    assert output.read_bytes() == b"newbin"
    assert [call[0] for call in session.calls] == ["HEAD", "GET"]
    assert session.calls[1] == (
        "GET",
        "https://example.test/model.bin",
        {"stream": True},
    )
    assert head_response.closed is True
    assert get_response.closed is True


def test_download_file_force_skips_head_and_refreshes_existing_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "model.bin"
    output.write_bytes(b"old")
    get_response = FakeResponse(chunks=(b"new",))
    session = FakeSession(get_response=get_response)

    asyncio.run(
        web._download_file(
            cast(niquests.AsyncSession, session),
            "https://example.test/model.bin",
            output,
            force=True,
        )
    )

    assert output.read_bytes() == b"new"
    assert session.calls == [
        ("GET", "https://example.test/model.bin", {"stream": True})
    ]
    assert get_response.closed is True


def test_download_file_closes_get_for_missing_file(tmp_path: Path) -> None:
    output = tmp_path / "model.bin"
    get_response = FakeResponse(chunks=(b"downloaded",))
    session = FakeSession(get_response=get_response)

    asyncio.run(
        web._download_file(
            cast(niquests.AsyncSession, session),
            "https://example.test/model.bin",
            output,
            force=False,
        )
    )

    assert output.read_bytes() == b"downloaded"
    assert session.calls == [
        ("GET", "https://example.test/model.bin", {"stream": True})
    ]
    assert get_response.closed is True
