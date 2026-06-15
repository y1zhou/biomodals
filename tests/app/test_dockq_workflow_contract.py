"""Tests for DockQ workflow-compatible result contracts."""

# ruff: noqa: D103

from pathlib import Path

from biomodals.app.score import dockq_app
from biomodals.schema import AppRunResult, AppRunStatus, ArtifactKind, InlineBytes
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


class _FakeDockQWorkflow:
    def __init__(self, result: AppRunResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_dockq_workflow_result_returns_inline_score_archive(monkeypatch) -> None:
    fake_batch = _FakeDockQBatch(b"zstd-bytes")
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
        data=b"zstd-bytes",
        filename="dockq-demo_dockq.tar.zst",
        media_type=ZSTD_MEDIA_TYPE,
    )
    assert output.metadata == {
        "archive_format": "tar.zst",
        "run_name": "dockq-demo",
    }


def test_dockq_local_entrypoint_writes_tarball_from_workflow_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = tmp_path / "model.pdb"
    reference = tmp_path / "reference.pdb"
    model.write_text("MODEL\n", encoding="utf-8")
    reference.write_text("REF\n", encoding="utf-8")
    input_csv = tmp_path / "pairs.csv"
    input_csv.write_text(
        f"id,model,reference\npair-1,{model.name},{reference.name}\n",
        encoding="utf-8",
    )
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            {
                "name": "dockq_scores",
                "kind": ArtifactKind.SCORES,
                "storage": InlineBytes(
                    data=b"archive-bytes",
                    filename="demo_dockq.tar.zst",
                    media_type=ZSTD_MEDIA_TYPE,
                ),
            }
        ],
    )
    fake_workflow = _FakeDockQWorkflow(result)
    monkeypatch.setattr(dockq_app, "run_dockq_workflow", fake_workflow)

    dockq_app.submit_dockq_task(
        input_csv=str(input_csv),
        out_dir=str(tmp_path / "out"),
        run_name="demo",
        dockq_args="--short",
    )

    assert fake_workflow.calls[0]["run_name"] == "demo"
    assert fake_workflow.calls[0]["dockq_args"] == ["--short"]
    assert (tmp_path / "out" / "demo.tar.zst").read_bytes() == b"archive-bytes"
