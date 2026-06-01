"""Standalone contract tests for the Rosetta app."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace

from biomodals.app.bioinfo import rosetta_app


def test_rosetta_no_local_output_reports_volume_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_pdb = tmp_path / "demo.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    uploaded = []
    queued = []
    deleted = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((local_path, remote_path))

    class FakeVolume:
        def batch_upload(self):
            return FakeBatch()

    class FakeQueue:
        def put(self, item):
            queued.append(item)

    monkeypatch.setattr(
        rosetta_app,
        "CONF",
        SimpleNamespace(
            name="Rosetta",
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="Rosetta-outputs",
        ),
    )
    monkeypatch.setattr(
        rosetta_app,
        "uuid4",
        lambda: SimpleNamespace(hex="abc123"),
    )
    monkeypatch.setattr(
        rosetta_app.modal,
        "Queue",
        SimpleNamespace(
            from_name=lambda *args, **kwargs: FakeQueue(),
            objects=SimpleNamespace(delete=lambda name: deleted.append(name)),
        ),
    )
    monkeypatch.setattr(
        rosetta_app.modal,
        "FunctionCall",
        SimpleNamespace(gather=lambda *tasks: None),
    )
    monkeypatch.setattr(
        rosetta_app,
        "run_rosetta",
        SimpleNamespace(spawn=lambda *args: SimpleNamespace(object_id="call-1")),
    )

    rosetta_app.submit_rosetta_task(
        rosetta_binary="relax",
        input_pdb=str(input_pdb),
        out_dir=None,
    )

    assert uploaded[0] == (input_pdb.resolve(), "/demo-abc123/1/demo.pdb")
    assert uploaded[1][1] == "/demo-abc123/tasks.parquet"
    assert queued[0]["pdb"] == "demo-abc123/1/demo.pdb"
    assert deleted == ["Rosetta-queue-abc123"]
    assert (
        "Results saved to 'demo-abc123' from volume 'Rosetta-outputs'"
        in capsys.readouterr().out
    )
