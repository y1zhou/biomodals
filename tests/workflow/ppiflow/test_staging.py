"""Tests for PPIFlow staging helpers."""

# ruff: noqa: D103

import tarfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from biomodals.app.design import ppiflow_app
from biomodals.schema import ArtifactKind, VolumePath, WorkflowArtifact
from biomodals.workflow import ppiflow_workflow
from biomodals.workflow.ppiflow import manifests, staging
from biomodals.workflow.ppiflow_workflow import (
    _active_ppiflow_app_steps,
    _inline_rosetta_config_files,
    _stage_ppiflow_app_inputs,
)


def _source_artifact(path: str) -> WorkflowArtifact:
    return WorkflowArtifact(
        artifact_id="upstream-structures",
        producing_node_id="upstream",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath(volume_name="source-volume", path=path),
    )


def _tar_zst_bytes(files: dict[str, bytes]) -> bytes:
    import zstandard as zstd

    tar_bytes = BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
    return zstd.ZstdCompressor().compress(tar_bytes.getvalue())


def test_select_structure_files_from_artifacts_reads_matching_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    structure_dir = source_root / "structures"
    structure_dir.mkdir(parents=True)
    (structure_dir / "design-1.pdb").write_text("ATOM 1\n", encoding="utf-8")
    (structure_dir / "notes.txt").write_text("skip\n", encoding="utf-8")

    selected = staging.select_structure_files_from_artifacts(
        [_source_artifact("structures")],
        {"source-volume": str(source_root)},
    )

    assert selected == [("upstream-structures__design-1.pdb", b"ATOM 1\n")]


def test_csv_files_from_artifact_reads_directory_csvs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    table_dir = source_root / "scores"
    table_dir.mkdir(parents=True)
    (table_dir / "metrics.csv").write_text("score\n1\n", encoding="utf-8")
    (table_dir / "notes.txt").write_text("skip\n", encoding="utf-8")
    artifact = WorkflowArtifact(
        artifact_id="scores",
        producing_node_id="scores",
        kind=ArtifactKind.SCORES,
        storage=VolumePath(volume_name="source-volume", path="scores"),
    )

    assert staging.csv_files_from_artifact(
        artifact,
        {"source-volume": str(source_root)},
    ) == [("metrics.csv", b"score\n1\n")]


def test_archive_readers_extract_selected_members(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    archive_path = source_root / "outputs.tar.zst"
    archive_path.write_bytes(
        _tar_zst_bytes({
            "nested/design.pdb": b"ATOM\n",
            "scores/metrics.csv": b"score\n1\n",
            "notes.txt": b"skip\n",
        })
    )
    artifact = WorkflowArtifact(
        artifact_id="upstream-structures",
        producing_node_id="upstream",
        kind=ArtifactKind.ARCHIVE,
        storage=VolumePath(
            volume_name="source-volume",
            path="outputs.tar.zst",
            media_type="application/zstd",
        ),
    )
    roots = {"source-volume": str(source_root)}

    assert staging.structure_files_from_artifact(artifact, None, roots) == [
        ("upstream-structures__nested__design.pdb", b"ATOM\n")
    ]
    assert staging.csv_files_from_artifact(artifact, roots) == [
        ("scores/metrics.csv", b"score\n1\n")
    ]
    assert staging.files_from_tar_zst_bytes(
        archive_path.read_bytes(),
        suffixes=(".csv",),
    ) == [("scores/metrics.csv", b"score\n1\n")]

    records = staging.selected_structure_file_records_from_artifact(
        artifact,
        None,
        roots,
    )
    assert [record.artifact_file_path for record in records] == ["nested/design.pdb"]
    assert records[0].size_bytes == archive_path.stat().st_size


def test_stage2_input_manifest_rows_scan_structure_directory(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    structure_dir = source_root / "existing"
    structure_dir.mkdir(parents=True)
    (structure_dir / "design-b.pdb").write_text("ATOM B\n", encoding="utf-8")
    (structure_dir / "design-a.pdb").write_text("ATOM A\n", encoding="utf-8")

    rows = staging.stage2_input_manifest_rows(
        _source_artifact("existing"),
        {"source-volume": str(source_root)},
        patterns=("*.pdb",),
    )

    assert [row["candidate_id"] for row in rows] == [
        manifests.stage2_input_candidate_id(1),
        manifests.stage2_input_candidate_id(2),
    ]
    assert [row["source_path"] for row in rows] == [
        "existing/design-a.pdb",
        "existing/design-b.pdb",
    ]
    assert rows[0]["files"][0]["volume_name"] == "source-volume"
    assert rows[0]["files"][0]["app_volume_path"] == "existing/design-a.pdb"
    assert rows[0]["files"][0]["expected"] is True


def test_candidate_structure_files_use_manifest_candidate_ids() -> None:
    manifest = pl.DataFrame({
        "candidate_id": ["candidate-a"],
        "source_path": ["inputs/design-a.pdb"],
        "derived_path": ["inputs/design-a.pdb"],
        "files": [[{"path": "design-a.pdb"}]],
    })

    selected = staging.candidate_structure_files_from_selected(
        [("artifact__design-a.pdb", b"ATOM\n")],
        manifest_frame=manifest,
    )

    assert selected == [
        staging.CandidateStructureFile(
            candidate_id="candidate-a",
            file_name="artifact__design-a.pdb",
            data=b"ATOM\n",
            source_path="artifact__design-a.pdb",
        )
    ]


def test_prepare_dockq_pairs_by_candidate_matches_ids() -> None:
    pairs = staging.prepare_dockq_pairs_by_candidate(
        references=[
            staging.CandidateStructureFile("b", "b-ref.pdb", b"REF B"),
            staging.CandidateStructureFile("a", "a-ref.pdb", b"REF A"),
        ],
        models=[
            staging.CandidateStructureFile("a", "a-model.pdb", b"MODEL A"),
            staging.CandidateStructureFile("b", "b-model.pdb", b"MODEL B"),
        ],
        mapping="A:B",
    )

    assert [pair["candidate_id"] for pair in pairs] == ["a", "b"]
    assert pairs[0]["reference_name"] == "a-ref.pdb"
    assert pairs[0]["model_name"] == "a-model.pdb"
    assert pairs[0]["mapping"] == "A:B"


def test_prepare_dockq_pairs_by_candidate_rejects_missing_pairs() -> None:
    with pytest.raises(ValueError, match="pairing mismatch"):
        staging.prepare_dockq_pairs_by_candidate(
            references=[staging.CandidateStructureFile("a", "a-ref.pdb", b"REF")],
            models=[staging.CandidateStructureFile("b", "b-model.pdb", b"MODEL")],
        )


def test_discover_partial_sample_dirs(tmp_path: Path) -> None:
    sample_dir = tmp_path / "stage2" / "partial" / "sample_0"
    sample_dir.mkdir(parents=True)
    (sample_dir / "model.pdb").write_text("ATOM\n", encoding="utf-8")
    other_dir = tmp_path / "stage2" / "other"
    other_dir.mkdir()
    (other_dir / "model.pdb").write_text("ATOM\n", encoding="utf-8")

    assert staging.discover_partial_sample_dirs(tmp_path) == [sample_dir]


def test_rosetta_job_manifest_rows_and_writer(tmp_path: Path) -> None:
    rows = staging.rosetta_job_manifest_rows(
        [
            staging.CandidateStructureFile(
                "candidate-a",
                "design-a.pdb",
                b"ATOM\n",
            )
        ],
        rosetta_binary="relax",
        rosetta_script="workflow.xml",
        flags_file="workflow.flags",
    )

    assert rows == [
        {
            "candidate_id": "candidate-a",
            "index": 1,
            "status": "pending",
            "binary": "relax",
            "pdb": "inputs/1/design-a.pdb",
            "rosetta_script": "workflow.xml",
            "flags_file": "workflow.flags",
            "expected_output_dir": "outputs/1",
            "expected_score_file": "outputs/1/score.sc",
            "worker_log": "logs/1.log",
        }
    ]
    manifest_path = staging.write_rosetta_job_manifest(
        rows,
        tmp_path / "rosetta_job_manifest.csv",
    )
    assert pl.read_csv(manifest_path).get_column("candidate_id").to_list() == [
        "candidate-a"
    ]


def test_ppiflow_entrypoint_stages_local_app_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    uploaded = []
    upload_forces = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self, force: bool = False):
            upload_forces.append(force)
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )

    steps_doc = {
        "PPIFlowStep": {
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "input_pdb": str(input_pdb),
                "binder_chain": "B",
            }
        }
    }

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=("PPIFlowStep",),
    )

    assert staged["PPIFlowStep"]["args"]["input_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/input_pdb/input.pdb"
    )
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input_pdb/input.pdb")]
    assert upload_forces == [False]

    uploaded.clear()
    upload_forces.clear()
    _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=("PPIFlowStep",),
        force=True,
    )
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input_pdb/input.pdb")]
    assert upload_forces == [True]


def test_ppiflow_staging_uses_active_stage_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self, force: bool = False):
            _ = force
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )
    task_doc = {
        "steps": {
            "PPIFlowStep": True,
            "PartialStep": True,
        }
    }
    steps_doc = {
        "PPIFlowStep": {
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "input_pdb": str(input_pdb),
                "binder_chain": "B",
            }
        },
        "PartialStep": {
            "args": {
                "name": "demo-partial",
                "specified_hotspots": "A1",
                "input_pdb": str(tmp_path / "stage2-not-local.pdb"),
                "fixed_positions": "B1",
                "start_t": 0.5,
            }
        },
    }

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=_active_ppiflow_app_steps(task_doc, stage=1),
    )

    assert staged["PPIFlowStep"]["args"]["input_pdb"].endswith(
        "/PPIFlowStep/input_pdb/input.pdb"
    )
    assert staged["PartialStep"]["args"]["input_pdb"].endswith("stage2-not-local.pdb")
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input_pdb/input.pdb")]

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=_active_ppiflow_app_steps(task_doc, stage=2),
    )

    assert staged["PartialStep"]["args"]["input_pdb"].endswith("stage2-not-local.pdb")
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input_pdb/input.pdb")]


def test_ppiflow_staging_keeps_same_basename_inputs_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    antigen_pdb = tmp_path / "antigen" / "input.pdb"
    framework_pdb = tmp_path / "framework" / "input.pdb"
    antigen_pdb.parent.mkdir()
    framework_pdb.parent.mkdir()
    antigen_pdb.write_text("ATOM antigen\n", encoding="utf-8")
    framework_pdb.write_text("ATOM framework\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self, force: bool = False):
            _ = force
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )
    steps_doc = {
        "PPIFlowStep": {
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "antigen_pdb": str(antigen_pdb),
                "antigen_chain": "A",
                "framework_pdb": str(framework_pdb),
                "heavy_chain": "H",
            }
        }
    }

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=("PPIFlowStep",),
    )

    assert staged["PPIFlowStep"]["args"]["antigen_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/antigen_pdb/input.pdb"
    )
    assert staged["PPIFlowStep"]["args"]["framework_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/framework_pdb/input.pdb"
    )
    assert uploaded == [
        (antigen_pdb, "/run-1/PPIFlowStep/antigen_pdb/input.pdb"),
        (framework_pdb, "/run-1/PPIFlowStep/framework_pdb/input.pdb"),
    ]


def test_ppiflow_rosetta_staging_inlines_local_config_files(tmp_path: Path) -> None:
    script_path = tmp_path / "protocol.xml"
    flags_path = tmp_path / "options.flags"
    script_path.write_text("<ROSETTASCRIPTS />\n", encoding="utf-8")
    flags_path.write_text("-relax:fast\n", encoding="utf-8")

    staged = _inline_rosetta_config_files({
        "RosettaRelaxStep": {
            "rosetta_script": str(script_path),
            "flags_file": str(flags_path),
        }
    })

    assert staged["RosettaRelaxStep"]["rosetta_script"] == "<ROSETTASCRIPTS />\n"
    assert staged["RosettaRelaxStep"]["flags_file"] == "-relax:fast\n"


def test_report_node_reads_rank_artifact_from_configured_volume_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    ranked_csv = source_root / "ranked.csv"
    ranked_csv.write_text("design,rank_score\ndesign-1,1.0\n", encoding="utf-8")
    artifact = WorkflowArtifact(
        artifact_id="rank",
        producing_node_id="rank",
        kind=ArtifactKind.TABLE,
        storage=VolumePath(volume_name="source-volume", path="ranked.csv"),
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "PPI_FLOW_SOURCE_VOLUME_ROOTS",
        {"source-volume": str(source_root)},
    )

    result = ppiflow_workflow.ReportNode("ReportStep").run(
        ppiflow_workflow.NodeRunContext(
            run_id="run-1",
            node_id="report",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={"rank": [artifact]},
        )
    )

    markdown = result.outputs[0].storage.data.decode()
    assert "Ranked designs: 1" in markdown
    assert "design-1" in markdown
