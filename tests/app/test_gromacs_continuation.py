"""Native continuation boundaries, without GROMACS or Modal compute."""

# ruff: noqa: D101,D102,D103,D107

import asyncio
from dataclasses import replace
from hashlib import md5, sha256
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from test_gromacs_execution_runtime import RUN_ID, _request

from biomodals.app.bioinfo.gromacs import (
    app,
    continuation,
    continue_run,
    execution_runtime,
)
from biomodals.app.bioinfo.gromacs.continuation import ContinuationSource
from biomodals.app.bioinfo.gromacs.execution import PREPARE_CONTINUATION, PREPARE_RESULT
from biomodals.app.bioinfo.gromacs.execution_runtime import (
    GromacsExecutionRequest,
    _operation_kwargs,
    gromacs_node_paths,
    gromacs_publication_path,
    persist_execution_request,
)
from biomodals.execution import ContentBoundFileSet
from biomodals.schema import ArtifactFile


def _publish(root, request, node):
    paths = gromacs_node_paths(request, node)
    ContentBoundFileSet(
        root=request.run_root(root),
        marker_path=root / gromacs_publication_path(request, node),
        expected_paths=paths,
        identity={
            "node_key": node,
            "workload_plan_fingerprint": request.execution_plan.workload_plan_fingerprint,
        },
    ).write(
        tuple(
            ArtifactFile(
                path=name,
                size_bytes=(request.run_root(root) / name).stat().st_size,
                content_sha256=sha256(
                    (request.run_root(root) / name).read_bytes()
                ).hexdigest(),
            )
            for name in paths
        )
    )


def _source(tmp_path, *, legacy=True):
    parent = replace(_request(), execution_plan_version="2" if legacy else "3")
    root = parent.run_root(tmp_path)
    root.mkdir()
    for node in parent.execution_plan.nodes:
        for name in gromacs_node_paths(parent, node.node_key):
            (root / name).write_bytes(name.encode())
    (root / "example.pdb").write_bytes(parent.pdb_content)
    (root / "production_example.cpt").write_bytes(b"full checkpoint")
    (root / "production_example.log").write_bytes(b"native log")
    for node in parent.execution_plan.nodes:
        _publish(tmp_path, parent, node.node_key)
    persist_execution_request(tmp_path, RUN_ID, parent)
    marker = (tmp_path / gromacs_publication_path(parent, PREPARE_RESULT)).read_bytes()
    source = ContinuationSource(
        execution_run_id=RUN_ID,
        run_name=parent.run_name,
        file_stem=parent.file_stem,
        simulation_time_ns=parent.simulation_time_ns,
        request_sha256=sha256(parent.to_bytes()).hexdigest(),
        publication_sha256=sha256(marker).hexdigest(),
        checkpoint_sha256=sha256(b"full checkpoint").hexdigest(),
    )
    child = replace(
        parent,
        run_name="child",
        simulation_time_ns=255,
        execution_plan_version="3",
        continuation=source,
        cpu_only=True,
    )
    return parent, child


def _dump(root, *, step=2500000, time=5000, warning="", bad_checksum=False):
    names = [f"production_example.{suffix}" for suffix in ("log", "xtc", "edr")]
    lines = [
        f"step = {step}",
        f"t = {time:.6f}",
        f"number of output files = {len(names)}",
    ]
    for name in names:
        content = (root / name).read_bytes()
        checksum = md5(content[-1048576:], usedforsecurity=False).hexdigest()
        lines += [
            f"output filename = {name}",
            "file_offset_high = 0",
            f"file_offset_low = {len(content)}",
            f"file_checksum_size = {min(len(content), 1048576)}",
            f"file_checksum = {'0' * 32 if bad_checksum else checksum}",
        ]
    return "\n".join([*lines, warning]) + "\n"


def _native(monkeypatch, source_root, **dump_options):
    calls = []
    monkeypatch.setattr(continue_run.shutil, "which", lambda _: "/bin/gmx")

    def command(args, **kwargs):
        calls.append(args)
        if "-om" in args:
            Path(args[args.index("-om") + 1]).write_text(
                "dt = 0.002\nnsteps = 2500000\ntinit = 0\ninit-step = 0\n"
            )
        elif "-cp" in args:
            Path(kwargs["log_file"]).write_text(_dump(source_root, **dump_options))
        elif "convert-tpr" in args:
            Path(args[args.index("-o") + 1]).write_bytes(b"extended fixed TPR")
        else:
            raise AssertionError(args)
        return []

    monkeypatch.setattr(continue_run, "run_command", command)
    return calls


@pytest.mark.parametrize("legacy", [True, False])
def test_isolated_cumulative_import_and_redelivery(tmp_path, monkeypatch, legacy):
    parent, child = _source(tmp_path, legacy=legacy)
    parent_root = parent.run_root(tmp_path)
    original = {
        p.relative_to(parent_root): p.read_bytes()
        for p in parent_root.rglob("*")
        if p.is_file()
    }
    calls = _native(monkeypatch, parent_root)
    child_root = continue_run.prepare_continuation_files(child, tmp_path)
    assert calls[-1][calls[-1].index("-extend") + 1] == "250000"
    assert (child_root / "production_example.xtc").read_bytes() == (
        parent_root / "production_example.xtc"
    ).read_bytes()
    assert (child_root / "production_example.xtc").stat().st_ino != (
        parent_root / "production_example.xtc"
    ).stat().st_ino
    assert (child_root / "source.tpr").read_bytes() == b"production_example.tpr"
    record = orjson.loads((child_root / "continuation.json").read_bytes())
    assert record["target_time_ns"] == 255
    assert record["equilibration_analysis"] == "inherited"
    assert child.execution_plan.node_keys == (
        PREPARE_CONTINUATION,
        "production_run_cpu",
        "collect_traj_stats:production_",
        PREPARE_RESULT,
    )
    assert GromacsExecutionRequest.from_bytes(child.to_bytes()) == child
    assert (
        child.execution_plan.workload_plan_fingerprint
        != parent.execution_plan.workload_plan_fingerprint
    )
    (child_root / "production_example.cpt").write_bytes(b"later child checkpoint")
    assert continue_run.prepare_continuation_files(child, tmp_path) == child_root
    assert len(calls) == 3  # no recopies, conversion or endpoint reset
    assert (
        child_root / "production_example.cpt"
    ).read_bytes() == b"later child checkpoint"
    assert original == {
        p.relative_to(parent_root): p.read_bytes()
        for p in parent_root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("damage", ["checkpoint", "trajectory", "topology", "input"])
def test_source_corruption_blocks_import(tmp_path, monkeypatch, damage):
    parent, child = _source(tmp_path)
    name = {
        "checkpoint": "production_example.cpt",
        "trajectory": "production_example.xtc",
        "topology": "production_example.tpr",
        "input": "example.pdb",
    }[damage]
    (parent.run_root(tmp_path) / name).write_bytes(b"changed")
    calls = _native(monkeypatch, parent.run_root(tmp_path))
    with pytest.raises(ValueError, match="changed|publication"):
        continue_run.prepare_continuation_files(child, tmp_path)
    assert calls == []


@pytest.mark.parametrize(
    "dump_options,reason",
    [
        ({"step": 2499999}, "endpoint"),
        ({"time": 4999.9}, "endpoint"),
        ({"bad_checksum": True}, "checksum"),
        (
            {"warning": "WARNING: Checkpoint file is corrupted or truncated"},
            "corrupted",
        ),
    ],
)
def test_native_checkpoint_errors_fail_before_conversion(
    tmp_path, monkeypatch, dump_options, reason
):
    parent, child = _source(tmp_path)
    calls = _native(monkeypatch, parent.run_root(tmp_path), **dump_options)
    with pytest.raises(ValueError, match=reason):
        continue_run.prepare_continuation_files(child, tmp_path)
    assert all("convert-tpr" not in call for call in calls)


def test_lost_preparation_cannot_overwrite_child_progress(tmp_path, monkeypatch):
    parent, child = _source(tmp_path)
    _native(monkeypatch, parent.run_root(tmp_path))
    root = continue_run.prepare_continuation_files(child, tmp_path)
    (root / "production_example.cpt").write_bytes(b"child progress")
    (root / "production_example.tpr").unlink()
    with pytest.raises(ValueError, match="replace child progress"):
        continue_run.prepare_continuation_files(child, tmp_path)
    assert (root / "production_example.cpt").read_bytes() == b"child progress"


def test_interrupted_checkpoint_import_can_be_redelivered(tmp_path, monkeypatch):
    parent, child = _source(tmp_path)
    _native(monkeypatch, parent.run_root(tmp_path))
    copyfile = continue_run.shutil.copyfile

    def interrupted_copy(source, destination):
        if source.suffix == ".cpt":
            destination.write_bytes(source.read_bytes()[:4])
            raise KeyboardInterrupt("Provider interrupted during checkpoint copy")
        return copyfile(source, destination)

    monkeypatch.setattr(continue_run.shutil, "copyfile", interrupted_copy)
    with pytest.raises(KeyboardInterrupt, match="checkpoint copy"):
        continue_run.prepare_continuation_files(child, tmp_path)
    monkeypatch.setattr(continue_run.shutil, "copyfile", copyfile)

    root = continue_run.prepare_continuation_files(child, tmp_path)
    assert (root / "production_example.cpt").read_bytes() == b"full checkpoint"
    assert (parent.run_root(tmp_path) / "production_example.cpt").read_bytes() == (
        b"full checkpoint"
    )
    assert continue_run.prepare_continuation_files(child, tmp_path) == root


@pytest.mark.parametrize("phase", ["source", "completion"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_checkpoint_scratch_is_temporary_and_outside_volume(
    tmp_path, monkeypatch, phase, corrupt
):
    parent, child = _source(tmp_path)
    parent_root = parent.run_root(tmp_path)
    _native(monkeypatch, parent_root, bad_checksum=corrupt)
    endpoint = continue_run.native_endpoint
    scratch_paths = []

    def inspect(gmx, tpr, checkpoint, scratch):
        scratch_paths.append(scratch)
        return endpoint(gmx, tpr, checkpoint, scratch)

    monkeypatch.setattr(continue_run, "native_endpoint", inspect)

    def verify():
        if phase == "source":
            continue_run.prepare_continuation_files(child, tmp_path)
        else:
            app._verify_production_endpoint(
                "/bin/gmx",
                parent_root / "production_example.tpr",
                parent_root / "production_example.cpt",
            )

    if corrupt:
        with pytest.raises(ValueError, match="checksum"):
            verify()
    else:
        verify()
    assert len(scratch_paths) == 1
    assert not scratch_paths[0].is_relative_to(tmp_path)
    assert not scratch_paths[0].exists()


def test_multiple_branches_and_fixed_endpoint_dispatch(tmp_path):
    parent, child = _source(tmp_path)
    sibling = replace(child, run_name="sibling", simulation_time_ns=6)
    assert sibling.continuation == child.continuation
    assert (
        sibling.execution_plan.workload_plan_fingerprint
        != child.execution_plan.workload_plan_fingerprint
    )
    assert _operation_kwargs(child, "production_run_cpu") == {
        "run_name": "child",
        "file_stem": "example",
        "simulation_time_ns": 255,
        "require_checkpoint": True,
        "fixed_target": True,
        "num_threads": 8,
        "use_openmp_threads": False,
    }
    assert (
        _operation_kwargs(child, "collect_traj_stats:production_")["file_stem"]
        == "example"
    )


@pytest.mark.parametrize("function", [app.production_run_cpu, app.production_run_gpu])
def test_production_uses_fixed_tpr_and_requires_checkpoint(
    tmp_path, monkeypatch, function
):
    root = tmp_path / "child"
    root.mkdir()
    (root / "production_parent.tpr").write_bytes(b"tpr")
    commits = []
    monkeypatch.setattr(
        app,
        "CONF",
        SimpleNamespace(
            output_volume_mountpoint=str(tmp_path),
            output_volume=SimpleNamespace(commit=lambda: commits.append(True)),
        ),
    )
    monkeypatch.setattr(continue_run.shutil, "which", lambda _: "/bin/gmx")
    commands = []
    monkeypatch.setattr(
        app, "run_command", lambda args, **kwargs: commands.append(args)
    )
    endpoint_checks = []
    monkeypatch.setattr(
        app, "_verify_production_endpoint", lambda *args: endpoint_checks.append(args)
    )
    kwargs = {
        "run_name": "child",
        "simulation_time_ns": 15,
        "file_stem": "parent",
        "require_checkpoint": True,
    }
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        function.get_raw_f()(**kwargs)
    (root / "production_parent.cpt").write_bytes(b"state")
    (root / "production_parent.xtc").write_bytes(b"trajectory behind checkpoint")
    function.get_raw_f()(**kwargs)
    command = commands[0]
    assert "-append" in command
    assert "-nsteps" not in command  # actual endpoint comes from the fixed TPR
    assert command[command.index("-cpi") + 1] == "production_parent.cpt"
    assert len(endpoint_checks) == len(commits) == 1


@pytest.mark.parametrize("duration", [0, 251])
def test_interval_limits_do_not_become_a_cumulative_cap(tmp_path, duration):
    _, child = _source(tmp_path)
    with pytest.raises(ValueError, match="1–250"):
        replace(
            child, simulation_time_ns=child.continuation.simulation_time_ns + duration
        )
    with pytest.raises(ValueError, match="positive|1–250"):
        replace(_request(), simulation_time_ns=duration)


def test_read_only_source_evidence_uses_bounded_volume_reads(tmp_path, monkeypatch):
    parent, child = _source(tmp_path, legacy=False)
    reads = []

    async def read(path):
        reads.append(path)
        yield (tmp_path / path).read_bytes()

    async def listdir(path):
        return [
            SimpleNamespace(
                path=p.relative_to(tmp_path).as_posix(), size=p.stat().st_size
            )
            for p in (tmp_path / path).iterdir()
            if p.is_file()
        ]

    volume = SimpleNamespace(
        read_file=SimpleNamespace(aio=read), listdir=SimpleNamespace(aio=listdir)
    )
    monkeypatch.setattr(
        execution_runtime, "load_execution_request_from_volume", lambda *_: parent
    )
    source, saved = asyncio.run(continuation.read_continuation_source(volume, RUN_ID))
    assert saved == parent
    assert source == child.continuation
    assert reads == [
        gromacs_publication_path(parent, PREPARE_RESULT).as_posix(),
        gromacs_publication_path(parent, "production_run_gpu").as_posix(),
        "example/production_example.cpt",
    ]
    monkeypatch.setattr(continuation, "MAX_CHECKPOINT_BYTES", 2)
    with pytest.raises(ValueError, match="exceeds"):
        asyncio.run(continuation.read_continuation_source(volume, RUN_ID))


def test_cli_continuation_reuses_source_without_input_pdb(tmp_path, monkeypatch):
    parent, child = _source(tmp_path)

    async def source(*_):
        return child.continuation, parent

    monkeypatch.setattr(continuation, "read_continuation_source", source)
    captured = {}
    monkeypatch.setattr(
        app,
        "stage_execution_request",
        lambda _volume, run_id, request: captured.update(
            request=request, run_id=run_id
        ),
    )
    monkeypatch.setattr(
        app,
        "submit_staged_execution_run",
        lambda _volume, **kwargs: captured.update(submission=kwargs),
    )
    info = app.submit_gromacs_task.info
    info.raw_f(continue_from=str(RUN_ID), additional_time_ns=250, run_name="child-cli")
    saved = captured["request"]
    assert saved.simulation_time_ns == 255
    assert saved.cpu_only == parent.cpu_only
    assert saved.ld_seed == parent.ld_seed
    assert saved.continuation == child.continuation
    assert captured["submission"]["predecessor_execution_run_id"] is None
    assert captured["run_id"] != RUN_ID
    with pytest.raises(ValueError, match="cannot be combined"):
        info.raw_f(
            continue_from=str(RUN_ID), additional_time_ns=1, restart_from=str(RUN_ID)
        )
