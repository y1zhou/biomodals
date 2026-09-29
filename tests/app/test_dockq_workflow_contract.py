"""Tests for DockQ workflow-compatible result contracts."""

# ruff: noqa: D103

import tarfile
from io import BytesIO
from pathlib import Path

import pytest
import zstandard as zstd

from biomodals.app.score import dockq_app
from biomodals.schema import AppRunStatus, ArtifactKind, InlineBytes
from biomodals.schema.storage import ZSTD_MEDIA_TYPE


class _FakeDockQBatch:
    def __init__(self, archive_bytes: bytes) -> None:
        self.archive_bytes = archive_bytes
        self.calls: list[dict[str, object]] = []

    def get_raw_f(self):
        def run_dockq_batch(**kwargs):
            self.calls.append(kwargs)
            return self.archive_bytes

        return run_dockq_batch


def _dockq_archive(csv_text: str) -> bytes:
    tar_buffer = BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        data = csv_text.encode("utf-8")
        info = tarfile.TarInfo("dockq_results.csv")
        info.size = len(data)
        tar.addfile(info, BytesIO(data))
    return zstd.ZstdCompressor().compress(tar_buffer.getvalue())


def test_dockq_workflow_result_returns_inline_score_archive(monkeypatch) -> None:
    archive = _dockq_archive(
        "id,model,reference,dockq,returncode,error\npair-1,model.pdb,ref.pdb,0.8,0,\n"
    )
    fake_batch = _FakeDockQBatch(archive)
    monkeypatch.setattr(dockq_app, "run_dockq_batch", fake_batch)

    result = dockq_app.run_dockq_workflow.get_raw_f()(
        pairs=[{"id": "pair-1", "model_bytes": b"m", "reference_bytes": b"r"}],
        run_name="dockq-demo",
        dockq_args=["--short"],
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert fake_batch.calls == [
        {
            "pairs": [
                {
                    "id": "pair-1",
                    "model_bytes": b"m",
                    "reference_bytes": b"r",
                }
            ],
            "run_name": "dockq-demo",
            "dockq_args": ["--short"],
        }
    ]
    output = result.outputs[0]
    assert output.name == "dockq_scores"
    assert output.kind == ArtifactKind.SCORES
    assert output.storage == InlineBytes(
        data=archive,
        filename="dockq-demo_dockq.tar.zst",
        media_type=ZSTD_MEDIA_TYPE,
    )
    assert output.metadata == {
        "archive_format": "tar.zst",
        "run_name": "dockq-demo",
        "pair_count": 1,
        "usable_rows": 1,
        "failed": 0,
    }


@pytest.mark.parametrize(
    ("csv_text", "expected_status", "expected_failed"),
    [
        (
            "id,model,reference,dockq,returncode,error\n"
            "pair-1,model.pdb,ref.pdb,0.8,0,\n"
            "pair-2,model.pdb,ref.pdb,,1,bad\n",
            AppRunStatus.PARTIAL,
            1,
        ),
        (
            "id,model,reference,dockq,returncode,error\n"
            "pair-1,model.pdb,ref.pdb,,1,bad\n",
            AppRunStatus.FAILED,
            1,
        ),
    ],
)
def test_dockq_workflow_reports_partial_or_failed_pairs(
    monkeypatch,
    csv_text: str,
    expected_status: AppRunStatus,
    expected_failed: int,
) -> None:
    fake_batch = _FakeDockQBatch(_dockq_archive(csv_text))
    monkeypatch.setattr(dockq_app, "run_dockq_batch", fake_batch)

    result = dockq_app.run_dockq_workflow.get_raw_f()(
        pairs=[
            {"id": "pair-1", "model_bytes": b"m", "reference_bytes": b"r"},
            {"id": "pair-2", "model_bytes": b"m", "reference_bytes": b"r"},
        ][: 2 if expected_status == AppRunStatus.PARTIAL else 1],
        run_name="dockq-demo",
        dockq_args=["--short"],
    )

    assert result.status == expected_status
    assert result.outputs[0].metadata["failed"] == expected_failed


def test_dockq_batch_shortens_long_structure_filenames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type("Result", (), {"returncode": 0, "stdout": "DockQ 0.8\n"})()

    monkeypatch.setattr(dockq_app.subprocess, "run", fake_run)
    long_name = f"stage2-filter-{'x' * 240}.pdb"

    row = dockq_app._row_from_pair(
        {
            "id": "candidate-a",
            "model_name": "model.pdb",
            "reference_name": long_name,
            "model_bytes": b"MODEL",
            "reference_bytes": b"REF",
        },
        pair_idx=1,
        workdir=tmp_path,
        dockq_args=["--short"],
    )

    reference_path = Path(calls[0][2])
    assert len(reference_path.name) <= dockq_app.MAX_STRUCTURE_FILENAME_LENGTH
    assert reference_path.exists()
    assert row["reference"] == reference_path.name
