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
)
from biomodals.app.bioinfo.gromacs.continuation import (
    ContinuationInspection,
    ContinuationSource,
)
from biomodals.app.bioinfo.gromacs.execution import (
    EXECUTION_PLAN_SCHEMA_VERSION,
    PREPARE_CONTINUATION,
    PREPARE_RESULT,
)
from biomodals.app.bioinfo.gromacs.execution_runtime import (
    GromacsExecutionRequest,
    _operation_kwargs,
    gromacs_node_paths,
    gromacs_publication_path,
    persist_execution_request,
)
from biomodals.execution import ContentBoundFileSet, DeploymentIdentity
from biomodals.helper.cli_entrypoint import invoke_local_entrypoint
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
        execution_plan_version=EXECUTION_PLAN_SCHEMA_VERSION,
        continuation=source,
        cpu_only=True,
    )
    return parent, child


def _without_energy_records(tmp_path, parent, child):
    for node in (
        PREPARE_RESULT,
        "production_run_cpu" if parent.cpu_only else "production_run_gpu",
    ):
        path = tmp_path / gromacs_publication_path(parent, node)
        marker = orjson.loads(path.read_bytes())
        marker["files"] = [
            file for file in marker["files"] if not file["path"].endswith(".edr")
        ]
        path.write_bytes(orjson.dumps(marker))
    marker = (tmp_path / gromacs_publication_path(parent, PREPARE_RESULT)).read_bytes()
    return replace(
        child,
        continuation=child.continuation.model_copy(
            update={
                "publication_sha256": sha256(marker).hexdigest(),
            }
        ),
    )


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
    assert len(calls) == 5  # source + staged verification, one conversion
    assert (
        child_root / "production_example.cpt"
    ).read_bytes() == b"later child checkpoint"
    assert original == {
        p.relative_to(parent_root): p.read_bytes()
        for p in parent_root.rglob("*")
        if p.is_file()
    }


def test_historical_records_find_unlisted_native_files_and_copy_for_append(
    tmp_path, monkeypatch
):
    parent, child = _source(tmp_path)
    child = _without_energy_records(tmp_path, parent, child)
    source_root = parent.run_root(tmp_path)
    # Regenerated outputs are not restart prerequisites.
    (source_root / "rmsd_production_example.png").unlink()
    (source_root / "production_example_nopbc.xtc").unlink()
    original = {
        path.relative_to(source_root): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    found = continuation.inspect_continuation_source(tmp_path, RUN_ID)
    assert found.source == child.continuation.model_copy(
        update={"checkpoint_sha256": None}
    )
    assert found.pdb_sha256 == parent.pdb_sha256
    _native(monkeypatch, source_root)
    root = continue_run.prepare_continuation_files(child, tmp_path)
    for suffix in (".cpt", ".xtc", ".edr", ".log"):
        name = f"production_example{suffix}"
        assert (root / name).read_bytes() == original[Path(name)]
        assert (root / name).stat().st_ino != (source_root / name).stat().st_ino
    assert original == {
        path.relative_to(source_root): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("suffix", [".tpr", ".xtc", ".cpt", ".edr", ".log"])
def test_missing_native_file_is_named_after_directory_fallback(tmp_path, suffix):
    parent, child = _source(tmp_path)
    _without_energy_records(tmp_path, parent, child)
    name = f"production_example{suffix}"
    (parent.run_root(tmp_path) / name).unlink()
    with pytest.raises(FileNotFoundError) as caught:
        continuation.inspect_continuation_source(tmp_path, RUN_ID)
    assert caught.value.filename == name


@pytest.mark.parametrize(
    "damage", ["identity", "duplicate", "escape", "digest", "conflict"]
)
def test_source_record_fallback_does_not_accept_invalid_evidence(tmp_path, damage):
    parent, _ = _source(tmp_path)
    final_path = tmp_path / gromacs_publication_path(parent, PREPARE_RESULT)
    marker = orjson.loads(final_path.read_bytes())
    if damage == "identity":
        marker["identity"]["workload_plan_fingerprint"] = "0" * 64
    elif damage == "duplicate":
        marker["files"].append(marker["files"][0])
    elif damage == "escape":
        marker["files"][0]["path"] = "../outside"
    elif damage == "digest":
        marker["files"][0].pop("content_sha256")
    else:
        next(file for file in marker["files"] if file["path"].endswith(".edr"))[
            "content_sha256"
        ] = "0" * 64
    final_path.write_bytes(orjson.dumps(marker))
    with pytest.raises(ValueError):
        continuation.inspect_continuation_source(tmp_path, RUN_ID)


def test_record_order_does_not_change_restart_file_discovery(tmp_path):
    parent, _ = _source(tmp_path)
    final_path = tmp_path / gromacs_publication_path(parent, PREPARE_RESULT)
    marker = orjson.loads(final_path.read_bytes())
    marker["files"].reverse()
    final_path.write_bytes(orjson.dumps(marker))
    found = continuation.inspect_continuation_source(tmp_path, RUN_ID)
    assert found.pdb_sha256 == parent.pdb_sha256
    assert (
        found.source.publication_sha256 == sha256(final_path.read_bytes()).hexdigest()
    )


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
    assert all("convert-tpr" not in call for call in calls)


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


@pytest.mark.parametrize("prepared", [False, True])
def test_recovery_cleans_hard_interruption_staging_only(
    tmp_path, monkeypatch, prepared
):
    parent, child = _source(tmp_path)
    parent_root = parent.run_root(tmp_path)
    _native(monkeypatch, parent_root)
    root = child.run_root(tmp_path)
    if prepared:
        continue_run.prepare_continuation_files(child, tmp_path)
        (root / "production_example.cpt").write_bytes(b"legitimate child progress")
    # Simulate the persisted filesystem left by SIGKILL, not exception unwinding.
    orphan = root / ".continuation-abandoned"
    orphan.mkdir(parents=True)
    (orphan / "production_example.xtc").write_bytes(b"partial copy")
    sibling = tmp_path / "sibling" / ".continuation-active"
    sibling.mkdir(parents=True)
    (sibling / "production_example.xtc").write_bytes(b"another child is copying")
    original = {p.name: p.read_bytes() for p in parent_root.iterdir() if p.is_file()}

    assert continue_run.prepare_continuation_files(child, tmp_path) == root
    assert not orphan.exists()
    assert (
        sibling / "production_example.xtc"
    ).read_bytes() == b"another child is copying"
    assert original == {
        p.name: p.read_bytes() for p in parent_root.iterdir() if p.is_file()
    }
    assert (root / "production_example.cpt").read_bytes() == (
        b"legitimate child progress" if prepared else b"full checkpoint"
    )


def test_staging_cleanup_refuses_symlink_escape(tmp_path):
    parent, child = _source(tmp_path)
    root = child.run_root(tmp_path)
    root.mkdir()
    (root / ".continuation-unsafe").symlink_to(
        parent.run_root(tmp_path), target_is_directory=True
    )
    with pytest.raises(ValueError, match="child-owned"):
        continue_run.prepare_continuation_files(child, tmp_path)
    assert (
        parent.run_root(tmp_path) / "production_example.cpt"
    ).read_bytes() == b"full checkpoint"


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
    assert len(scratch_paths) == (2 if phase == "source" and not corrupt else 1)
    assert all(
        not path.is_relative_to(tmp_path) and not path.exists()
        for path in scratch_paths
    )


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
@pytest.mark.parametrize("use_openmp_threads", [False, True])
def test_production_uses_fixed_tpr_and_requires_checkpoint(
    tmp_path, monkeypatch, function, use_openmp_threads
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
            output_volume=SimpleNamespace(
                commit=lambda: commits.append(True), reload=lambda: None
            ),
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
        "num_threads": 4,
        "use_openmp_threads": use_openmp_threads,
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
    backend = "cpu" if function is app.production_run_cpu else "gpu"
    for flag in ("-nb", "-pmefft", "-pme", "-bonded", "-update"):
        assert command[command.index(flag) + 1] == backend
    if backend == "gpu":
        assert command[command.index("-gpu_id") + 1] == "0"
    else:
        assert "-gpu_id" not in command
    assert command[command.index("-ntomp" if use_openmp_threads else "-nt") + 1] == "4"
    if use_openmp_threads:
        assert command[command.index("-ntmpi") + 1] == "1"
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


def test_inspection_reads_only_metadata_and_never_copies_native_files(
    tmp_path, monkeypatch
):
    parent, child = _source(tmp_path, legacy=False)
    reads = []
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.suffix == ".json", "Inspection must not read native file content"
        reads.append(path)
        return original_open(path, *args, **kwargs)

    before = set(tmp_path.rglob("*"))
    monkeypatch.setattr(Path, "open", metadata_only)
    found = continuation.inspect_continuation_source(tmp_path, RUN_ID)
    assert found.source == child.continuation
    assert found.pdb_sha256 == parent.pdb_sha256
    assert len(reads) == 3
    assert set(tmp_path.rglob("*")) == before
    assert len(found.model_dump_json()) < 2048
    monkeypatch.setattr(continuation, "MAX_CHECKPOINT_BYTES", 2)
    with pytest.raises(ValueError, match="exceeds"):
        continuation.inspect_continuation_source(tmp_path, RUN_ID)


@pytest.mark.parametrize("legacy", [True, False])
def test_metadata_only_child_preserves_identity_and_prepares_inside_volume(
    tmp_path, monkeypatch, legacy
):
    parent, previous_child = _source(tmp_path, legacy=legacy)
    found = continuation.inspect_continuation_source(tmp_path, RUN_ID)
    child = found.request(
        run_name="child",
        additional_time_ns=250,
        cpu_only=True,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=1,
    )
    assert child.pdb_content == b""  # No source bytes staged through the API.
    assert child.pdb_sha256 == parent.pdb_sha256
    assert (
        child.execution_plan.workload_plan_fingerprint
        == replace(
            previous_child, continuation=found.source
        ).execution_plan.workload_plan_fingerprint
    )
    assert GromacsExecutionRequest.from_bytes(child.to_bytes()) == child
    _native(monkeypatch, parent.run_root(tmp_path))
    root = continue_run.prepare_continuation_files(child, tmp_path)
    assert (root / "example.pdb").read_bytes() == parent.pdb_content
    assert (
        orjson.loads((root / "continuation.json").read_bytes())[
            "source_checkpoint_sha256"
        ]
        == sha256(b"full checkpoint").hexdigest()
    )


def test_remote_inspection_transfers_metadata_only(tmp_path, monkeypatch):
    import modal

    _, _ = _source(tmp_path)
    reloads = []
    monkeypatch.setattr(
        app,
        "CONF",
        SimpleNamespace(
            output_volume_mountpoint=str(tmp_path),
            output_volume=SimpleNamespace(reload=lambda: reloads.append(True)),
        ),
    )
    lookups = []

    async def remote(run_id):
        return app.inspect_continuation_source.get_raw_f()(run_id)

    def lookup(*args, **kwargs):
        lookups.append((args, kwargs))
        return SimpleNamespace(remote=SimpleNamespace(aio=remote))

    monkeypatch.setattr(modal.Function, "from_name", lookup)
    found = asyncio.run(
        continuation.read_continuation_source(
            DeploymentIdentity("production", "Gromacs", 7), RUN_ID
        )
    )
    assert found.source.execution_run_id == RUN_ID
    assert reloads == [True]
    assert lookups == [
        (
            ("Gromacs", "inspect_continuation_source"),
            {"environment_name": "production", "version": 7},
        )
    ]


def test_corrupt_staged_append_file_is_not_promoted(tmp_path, monkeypatch):
    parent, child = _source(tmp_path)
    child = _without_energy_records(tmp_path, parent, child)
    _native(monkeypatch, parent.run_root(tmp_path))
    original_copy = continue_run.shutil.copyfile

    def corrupt_copy(source, destination):
        original_copy(source, destination)
        if source.suffix == ".edr":
            destination.write_bytes(b"x" * destination.stat().st_size)

    monkeypatch.setattr(continue_run.shutil, "copyfile", corrupt_copy)
    with pytest.raises(ValueError, match="append checksum"):
        continue_run.prepare_continuation_files(child, tmp_path)
    assert not (child.run_root(tmp_path) / "production_example.cpt").exists()
    assert not list(child.run_root(tmp_path).glob(".continuation-*"))


@pytest.mark.parametrize("positional", [False, True])
def test_cli_fresh_simulation_accepts_named_or_positional_pdb(
    tmp_path, monkeypatch, positional
):
    pdb = tmp_path / "protein.pdb"
    pdb.write_bytes(b"ATOM\n")
    staged = []
    monkeypatch.setattr(
        app,
        "stage_execution_request",
        lambda _volume, _run_id, request: staged.append(request),
    )
    monkeypatch.setattr(
        app, "submit_staged_execution_run", lambda *_args, **_kwargs: None
    )
    invoke_local_entrypoint(
        module_name=app.__name__,
        entrypoint_name="submit_gromacs_task",
        flags=[str(pdb)] if positional else ["--input-pdb", str(pdb)],
        overrides={"use_deployed_coordinator": True},
        program_name="biomodals app run gromacs --",
    )
    assert len(staged) == 1
    assert staged[0].pdb_content == b"ATOM\n"
    assert staged[0].run_name == "protein"
    assert staged[0].simulation_time_ns == 5
    assert not staged[0].cpu_only


def test_cli_continuation_reuses_source_without_input_pdb(tmp_path, monkeypatch):
    parent, child = _source(tmp_path)

    async def source(*_):
        return ContinuationInspection.from_request(child.continuation, parent)

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
    invoke_local_entrypoint(
        module_name=app.__name__,
        entrypoint_name="submit_gromacs_task",
        flags=[
            "--continue-from",
            str(RUN_ID),
            "--additional-time-ns",
            "250",
            "--run-name",
            "child-cli",
        ],
        overrides={"use_deployed_coordinator": True},
        program_name="biomodals app run gromacs --",
    )
    saved = captured["request"]
    assert saved.simulation_time_ns == 255
    assert saved.cpu_only == parent.cpu_only
    assert saved.ld_seed == parent.ld_seed
    assert saved.continuation == child.continuation
    assert captured["submission"]["predecessor_execution_run_id"] is None
    assert captured["run_id"] != RUN_ID
    with pytest.raises(ValueError, match="cannot be combined"):
        app.submit_gromacs_task.info.raw_f(
            continue_from=str(RUN_ID), additional_time_ns=1, restart_from=str(RUN_ID)
        )


@pytest.mark.parametrize(
    "flags",
    [
        ["--simulation-time-ns", "5"],
        ["--num-threads", "16"],
        ["--run-pdbfixer"],
        ["--no-run-pdbfixer"],
        ["--use-openmp-threads"],
        ["--no-use-openmp-threads"],
        ["--ld-seed", "-1"],
        ["--gen-seed", "-1"],
        ["--genion-seed", "0"],
    ],
)
def test_cli_continuation_rejects_explicit_ignored_options_before_remote_read(
    monkeypatch, flags
):
    async def unexpected_read(*args):
        raise AssertionError("Invalid arguments must fail before inspecting a source")

    monkeypatch.setattr(continuation, "read_continuation_source", unexpected_read)
    with pytest.raises(ValueError, match="inherits settings"):
        invoke_local_entrypoint(
            module_name=app.__name__,
            entrypoint_name="submit_gromacs_task",
            flags=["--continue-from", str(RUN_ID), "--additional-time-ns", "1", *flags],
            overrides={"use_deployed_coordinator": True},
            program_name="biomodals app run gromacs --",
        )
