"""Standalone contract tests for the Rosetta app."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace

from biomodals.app.bioinfo import rosetta_app
from biomodals.helper import shell as shell_helper


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

    assert uploaded[0] == (input_pdb.resolve(), "/demo-abc123/inputs/1/demo.pdb")
    assert uploaded[1][1] == "/demo-abc123/inputs/tasks.parquet"
    assert queued[0]["pdb"] == "inputs/1/demo.pdb"
    assert deleted == ["Rosetta-queue-abc123"]
    assert (
        "Results saved to 'demo-abc123' in volume 'Rosetta-outputs'"
        in capsys.readouterr().out
    )


def test_run_rosetta_uses_app_run_layout(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

    class FakeQueue:
        def __init__(self) -> None:
            self.items = [
                {
                    "index": 1,
                    "binary": "/usr/bin/relax",
                    "pdb": "inputs/1/demo.pdb",
                    "rosetta_script": "inputs/_script/script.xml",
                    "flags_file": "inputs/_flags/options.flags",
                }
            ]

        def get(self, block=False):
            if self.items:
                return self.items.pop(0)
            return None

    output_volume = FakeVolume()
    monkeypatch.setattr(
        rosetta_app,
        "CONF",
        SimpleNamespace(
            name="Rosetta",
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
        ),
    )
    monkeypatch.setattr(
        rosetta_app.modal,
        "Queue",
        SimpleNamespace(from_name=lambda *args, **kwargs: FakeQueue()),
    )

    def fake_run_command(cmd, *, output_mode, log_file):
        captured["cmd"] = cmd
        captured["output_mode"] = output_mode
        captured["log_file"] = log_file
        return []

    monkeypatch.setattr(shell_helper, "run_command", fake_run_command)

    rosetta_app.run_rosetta.get_raw_f()("demo", "abc123", 1)

    run_root = tmp_path / "demo-abc123"
    assert captured["cmd"] == [
        "/usr/bin/relax",
        "-parser:protocol",
        str(run_root / "inputs" / "_script" / "script.xml"),
        f"@{run_root / 'inputs' / '_flags' / 'options.flags'}",
        "-s",
        str(run_root / "inputs" / "1" / "demo.pdb"),
        "-out:path:all",
        str(run_root / "outputs" / "1"),
    ]
    assert captured["output_mode"] == "capture"
    assert captured["log_file"] == run_root / "logs" / "1.log"
    assert (run_root / "outputs" / "1").is_dir()
    assert (run_root / "logs").is_dir()
    assert output_volume.commit_count == 1
