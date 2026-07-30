"""Tests for AlphaFold3 coordinator result references."""

# ruff: noqa: D101,D102,D107

from pathlib import Path
from uuid import UUID

from biomodals.app.fold.alphafold3.execution_publications import (
    execution_result_path,
    load_execution_result,
    publish_execution_result,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        pass


def test_execution_result_is_published_outside_kernel_state(tmp_path: Path) -> None:
    """Large decoded results use app-owned files and small envelope references."""
    volume = FakeVolume()
    path = execution_result_path(RUN_ID, "combined-msa-publications", "a" * 64)

    reference = publish_execution_result(
        tmp_path,
        volume,
        path.as_posix(),
        {"status": "request-local", "fields": {"unpairedMsa": ">query\nACDE\n"}},
    )

    assert path.parts[0] == "execution-publications"
    assert ".biomodals" not in path.parts
    assert load_execution_result(
        tmp_path,
        reference,
        expected_path=path,
    ) == {
        "status": "request-local",
        "fields": {"unpairedMsa": ">query\nACDE\n"},
    }
    assert volume.commits == 1


def test_execution_result_rejects_a_changed_publication(tmp_path: Path) -> None:
    """The reference, not file existence alone, controls result recovery."""
    volume = FakeVolume()
    path = execution_result_path(RUN_ID, "seed-predictions", "b" * 64)
    reference = publish_execution_result(
        tmp_path,
        volume,
        path.as_posix(),
        {"status": "complete"},
    )
    tmp_path.joinpath(*path.parts).write_text('{"status":"changed"}')

    assert (
        load_execution_result(
            tmp_path,
            reference,
            expected_path=path,
        )
        is None
    )
