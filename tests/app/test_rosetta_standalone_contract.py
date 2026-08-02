"""Standalone contracts for the Rosetta app."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from biomodals.app.bioinfo import rosetta_app
from biomodals.execution import PullTaskClaim, RunStatus, WorkerAssignmentRecord
from biomodals.helper import shell as shell_helper

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

    class Coordinator:
        run = CoordinatorMethod()

    def stage(output_volume, execution_run_id, request):
        captured["staged"] = (output_volume, execution_run_id, request)

    def coordinator_handle(**kwargs):
        captured["coordinator"] = kwargs
        return Coordinator()

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
        "stage_execution_launch",
        lambda _volume, run_id, predecessor: captured.update(
            launch=(run_id, predecessor)
        ),
    )
    monkeypatch.setattr(
        rosetta_app,
        "_execution_coordinator_handle",
        coordinator_handle,
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
    assert captured["launch"] == (EXECUTION_RUN_ID, None)
    assert captured["run_kwargs"] == {"development": True}
    assert captured["coordinator"] == {
        "execution_run_id": EXECUTION_RUN_ID,
        "deployment": rosetta_app.DeploymentIdentity("main", "Rosetta", 1),
        "use_deployed_coordinator": False,
        "local_coordinator": rosetta_app.ExecutionCoordinator,
    }
    output = capsys.readouterr().out
    assert f"Execution Run ID: {EXECUTION_RUN_ID}" in output
    assert (
        f"Results saved to '{workload_run_key}' in volume 'Rosetta-outputs'" in output
    )


def test_rosetta_worker_uses_app_run_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

    task = rosetta_app.RosettaTaskSpec(
        task_key="1",
        index=1,
        binary="/usr/bin/relax",
        pdb="inputs/1/demo.pdb",
        rosetta_script="inputs/_script/script.xml",
        flags_file="inputs/_flags/options.flags",
        output_dir="outputs/1",
        worker_log="logs/1.log",
        expected_files=(),
        input_sha256="a" * 64,
        script_sha256="b" * 64,
        flags_sha256="c" * 64,
    )
    assignment = WorkerAssignmentRecord(
        execution_run_id=EXECUTION_RUN_ID,
        node_key="rosetta-tasks",
        task_key=task.task_key,
        task_fingerprint="fingerprint",
        execution_payload=task.to_dict(),
        provider_call_id=PROVIDER_CALL_ID,
        request_id="claim",
        ordinal=0,
        created_at=1,
    )
    claim_count = 0
    completions = []

    def claim(provider_call_id, request_id, capacity):
        nonlocal claim_count
        captured.setdefault("claims", []).append((
            provider_call_id,
            request_id,
            capacity,
        ))
        assignments = (assignment,) if claim_count == 0 else ()
        claim_count += 1
        return PullTaskClaim(
            request_id=request_id,
            provider_call_id=PROVIDER_CALL_ID,
            assignments=assignments,
        )

    def complete(provider_call_id, task_key, request_id, result):
        completions.append((provider_call_id, task_key, request_id, result))

    output_volume = FakeVolume()
    monkeypatch.setattr(
        rosetta_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
        ),
    )
    coordinator = SimpleNamespace(
        claim_tasks=SimpleNamespace(remote=claim),
        complete_task=SimpleNamespace(remote=complete),
    )

    def fake_run_command(cmd, *, output_mode, log_file):
        captured["cmd"] = cmd
        captured["output_mode"] = output_mode
        captured["log_file"] = log_file
        Path(log_file).write_text("log\n", encoding="utf-8")
        Path(cmd[-1], "result.pdb").write_text("ATOM\n", encoding="utf-8")

    monkeypatch.setattr(shell_helper, "run_command", fake_run_command)

    summary = rosetta_app.run_rosetta_worker.get_raw_f()(
        coordinator=coordinator,
        provider_call_id=str(PROVIDER_CALL_ID),
        run_name="demo",
        run_id="abc123",
        claim_capacity=1,
        max_parallel=1,
    )

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
    assert captured["output_mode"] == "log"
    assert captured["log_file"] == run_root / "logs" / "1.log"
    assert completions[0][0:3] == (
        str(PROVIDER_CALL_ID),
        "1",
        f"{PROVIDER_CALL_ID}:complete:fingerprint",
    )
    assert completions[0][3]["status"] == "succeeded"
    assert summary == {"claimed_tasks": 1, "claim_requests": 2}
    assert output_volume.commit_count == 2


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
        rosetta_app.run_rosetta_worker.get_raw_f()(
            coordinator=SimpleNamespace(),
            provider_call_id=str(PROVIDER_CALL_ID),
            run_name="../escape",
            run_id="abc123",
            claim_capacity=1,
            max_parallel=1,
        )

    assert not (tmp_path.parent / "escape-abc123").exists()
