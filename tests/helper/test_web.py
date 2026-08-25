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
        status_code: int = 200,
    ) -> None:
        self.headers = headers or {}
        self.chunks = chunks
        self.status_error = status_error
        self.status_code = status_code
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
        get_responses: FakeResponse | list[FakeResponse] | None = None,
    ) -> None:
        self.head_response = head_response
        self.head_error = head_error
        self.get_responses = (
            [get_responses]
            if isinstance(get_responses, FakeResponse)
            else list(get_responses or [])
        )
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
        if not self.get_responses:
            raise AssertionError("unexpected GET request")
        return self.get_responses.pop(0)


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
    session = FakeSession(head_response=head_response, get_responses=get_response)

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
    session = FakeSession(get_responses=get_response)

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
    session = FakeSession(get_responses=get_response)

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


def test_download_file_resumes_partial_file_with_range(tmp_path: Path) -> None:
    output = tmp_path / "archive.zst.part"
    output.write_bytes(b"partial")
    head_response = FakeResponse(headers={"content-length": "11"})
    get_response = FakeResponse(
        headers={"Content-Range": "bytes 7-10/11"},
        chunks=(b"rest",),
        status_code=206,
    )
    session = FakeSession(head_response=head_response, get_responses=get_response)

    asyncio.run(
        web._download_file(
            cast(niquests.AsyncSession, session),
            "https://example.test/archive.zst",
            output,
            force=False,
            resume=True,
        )
    )

    assert output.read_bytes() == b"partialrest"
    assert session.calls[1] == (
        "GET",
        "https://example.test/archive.zst",
        {"stream": True, "headers": {"Range": "bytes=7-"}},
    )


def test_resumable_download_tries_get_when_head_fails(tmp_path: Path) -> None:
    output = tmp_path / "archive.zst.part"
    output.write_bytes(b"partial")
    get_response = FakeResponse(
        headers={"content-range": "bytes 7-10/11"},
        chunks=(b"rest",),
        status_code=206,
    )
    session = FakeSession(
        head_error=RuntimeError("HEAD unsupported"),
        get_responses=get_response,
    )

    asyncio.run(
        web._download_file(
            cast(niquests.AsyncSession, session),
            "https://example.test/archive.zst",
            output,
            force=False,
            resume=True,
        )
    )

    assert output.read_bytes() == b"partialrest"


def test_resumable_download_restarts_after_range_not_satisfiable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archive.zst.part"
    output.write_bytes(b"stale-partial")
    head_response = FakeResponse(headers={"content-length": "10"})
    rejected = FakeResponse(status_code=416)
    complete = FakeResponse(chunks=(b"fresh-data",))
    session = FakeSession(
        head_response=head_response,
        get_responses=[rejected, complete],
    )

    asyncio.run(
        web._download_file(
            cast(niquests.AsyncSession, session),
            "https://example.test/archive.zst",
            output,
            force=False,
            resume=True,
        )
    )

    assert output.read_bytes() == b"fresh-data"
    assert session.calls[1][2]["headers"] == {"Range": "bytes=13-"}
    assert session.calls[2] == (
        "GET",
        "https://example.test/archive.zst",
        {"stream": True},
    )
    assert rejected.closed is True


def test_resumable_download_restarts_after_mismatched_content_range(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archive.zst.part"
    output.write_bytes(b"partial")
    head_response = FakeResponse(headers={"content-length": "11"})
    mismatched = FakeResponse(
        headers={"content-range": "bytes 3-10/11"},
        chunks=(b"ignored",),
        status_code=206,
    )
    complete = FakeResponse(chunks=(b"replacement",))
    session = FakeSession(
        head_response=head_response,
        get_responses=[mismatched, complete],
    )

    asyncio.run(
        web._download_file(
            cast(niquests.AsyncSession, session),
            "https://example.test/archive.zst",
            output,
            force=False,
            resume=True,
        )
    )

    assert output.read_bytes() == b"replacement"
    assert mismatched.closed is True
