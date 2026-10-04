"""Standalone contracts for the Rosetta app."""

# ruff: noqa: D103

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from biomodals.app.bioinfo import rosetta_app
from biomodals.execution import RunStatus

WORKLOAD_UUID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
EXECUTION_RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
PROVIDER_CALL_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")


def test_rosetta_no_local_output_uses_remote_coordinator(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_pdb = tmp_path / "demo.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    uploaded = []
    captured = {}

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

    class CoordinatorMethod:
        def spawn(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return SimpleNamespace(
                object_id="fc-coordinator",
                get=lambda: SimpleNamespace(
                    run=SimpleNamespace(
                        status=RunStatus.SUCCEEDED,
                        status_reason=None,
                        status_message=None,
                    )
                ),
            )

    def stage(output_volume, execution_run_id, request):
        captured["staged"] = (output_volume, execution_run_id, request)

    generated_ids = iter((WORKLOAD_UUID, EXECUTION_RUN_ID))
    output_volume = FakeVolume()
    monkeypatch.setattr(
        rosetta_app,
        "CONF",
        SimpleNamespace(
            name="Rosetta",
            version="2025.51",
            output_volume=output_volume,
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="Rosetta-outputs",
        ),
    )
    monkeypatch.setattr(rosetta_app, "uuid4", lambda: next(generated_ids))
    monkeypatch.setattr(rosetta_app, "stage_execution_request", stage)
    monkeypatch.setattr(
        rosetta_app,
        "submit_staged_execution_run",
        lambda volume, **kwargs: (
            captured.update(submit=(volume, kwargs))
            or CoordinatorMethod().spawn().get()
        ),
    )
    monkeypatch.setattr(
        rosetta_app,
        "load_execution_request_from_volume",
        lambda output, execution_run_id: captured["staged"][2],
    )

    entrypoint = rosetta_app.submit_rosetta_task.info
    assert entrypoint is not None and entrypoint.raw_f is not None
    entrypoint.raw_f(
        rosetta_binary="relax",
        input_pdb=str(input_pdb),
        out_dir=None,
    )

    workload_run_key = f"demo-{WORKLOAD_UUID.hex}"
    assert uploaded[0] == (
        input_pdb.resolve(),
        f"/{workload_run_key}/inputs/1/demo.pdb",
    )
    assert uploaded[1][1] == f"/{workload_run_key}/inputs/tasks.parquet"
    _, staged_run_id, request = captured["staged"]
    assert staged_run_id == EXECUTION_RUN_ID
    assert request.workload_run_key == workload_run_key
    assert request.tasks[0].pdb == "inputs/1/demo.pdb"
    volume, submit_kwargs = captured["submit"]
    assert volume is output_volume
    assert submit_kwargs == {
        "execution_run_id": EXECUTION_RUN_ID,
        "deployment": rosetta_app.DeploymentIdentity("main", "Rosetta", 1),
        "predecessor_execution_run_id": None,
        "use_deployed_coordinator": False,
        "local_coordinator": rosetta_app.ExecutionCoordinator,
        "workload_name": "Rosetta",
    }
    output = capsys.readouterr().out
    assert (
        f"Results saved to '{workload_run_key}' in volume 'Rosetta-outputs'" in output
    )


def test_rosetta_staging_hashes_exact_script_and_flag_bytes(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "input.pdb"
    script = tmp_path / "protocol.xml"
    flags = tmp_path / "options.flags"
    pdb.write_bytes(b"ATOM\r\n")
    script.write_bytes(b"<ROSETTASCRIPTS />\r\n")
    flags.write_bytes(b"-nstruct 1\r\n")

    [row] = rosetta_app._prepare_input_csv(
        input_pdb=str(pdb),
        input_rosetta_script=str(script),
        input_flags_file=str(flags),
    ).to_dicts()

    assert row["script_hash"] == sha256(script.read_bytes()).hexdigest()
    assert row["flags_hash"] == sha256(flags.read_bytes()).hexdigest()


def test_rosetta_worker_rejects_path_escaping_run_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        rosetta_app,
        "CONF",
        SimpleNamespace(
            output_volume=SimpleNamespace(),
            output_volume_mountpoint=str(tmp_path),
        ),
    )

    with pytest.raises(ValueError, match="safe filename component"):
        rosetta_app.run_rosetta_worker.local(
            coordinator=SimpleNamespace(),
            provider_call_id=str(PROVIDER_CALL_ID),
            run_name="../escape",
            run_id="abc123",
            claim_capacity=1,
            max_parallel=1,
        )

    assert not (tmp_path.parent / "escape-abc123").exists()
