"""Tests for standalone OligoFormer app behavior."""

# ruff: noqa: D101,D102,D103,D107

import shlex
import shutil
import subprocess as sp
import tarfile
import zipfile
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import orjson
import polars as pl
import pytest

from biomodals.app.score import oligoformer_app


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def _fake_conf(tmp_path: Path, volume: FakeVolume, repo_dir: Path | None = None):
    return SimpleNamespace(
        name=oligoformer_app.CONF.name,
        package_name=oligoformer_app.CONF.package_name,
        version=oligoformer_app.CONF.version,
        repo_commit_hash=oligoformer_app.CONF.repo_commit_hash,
        output_volume_mountpoint=str(tmp_path / "outputs-volume"),
        git_clone_dir=repo_dir or tmp_path / "repo",
        output_volume=volume,
    )


def _run_config(**changes):
    return replace(oligoformer_app.OligoformerRunConfig(), **changes)


def _execution_config(**changes):
    return replace(oligoformer_app.DEFAULT_EXECUTION_CONFIG, **changes)


def _write_valid_output_bundle(output_dir: Path, stem: str = "target") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    contents = (
        "pos\tsense\tsiRNA\tefficacy\tfunc_filter\tfilter\n"
        "1\tGC\tAUGCUAGCUAGCUAGCUAG\t0.8\t0\t0\n"
    )
    for suffix in ("", "_ranked", "_ranked_filtered"):
        output_dir.joinpath(f"{stem}{suffix}.txt").write_text(
            contents,
            encoding="utf-8",
        )


def _publish_efficacy_marker(
    layout: oligoformer_app.AppRunLayout,
    output_dir: Path,
    *,
    efficacy_key: str = "efficacy123",
    output_stems: tuple[str, ...] = ("target",),
) -> None:
    oligoformer_app._publish_output_bundle_marker(
        oligoformer_app._marker_path(layout, "efficacy.done"),
        output_dir=output_dir,
        paths=oligoformer_app._output_paths(output_dir, output_stems),
        identity={
            "efficacy_key": efficacy_key,
            "output_stems": list(output_stems),
        },
    )


def _publish_test_off_target_evidence(
    run_root: Path,
    *,
    stem: str,
    pita: str,
    targetscan: str,
) -> None:
    evidence_dir = (
        oligoformer_app.AppRunLayout.from_run_root(run_root).prep_dir
        / "off_target"
        / stem
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.joinpath("pita.tab").write_text(pita, encoding="utf-8")
    evidence_dir.joinpath("targetscan.tab").write_text(
        targetscan,
        encoding="utf-8",
    )
    oligoformer_app._publish_off_target_manifest(
        evidence_dir,
        identity=oligoformer_app._off_target_evidence_identity(run_root, stem),
    )


def _write_targetscan_context_output(
    path: Path,
    rows: list[tuple[str, str, str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"h{index}" for index in range(1, 29)]
    lines = ["\t".join(header)]
    for target, sirna, site_type, score in rows:
        fields = [f"x{index}" for index in range(1, 29)]
        fields[0] = target
        fields[2] = sirna
        fields[3] = site_type
        fields[27] = score
        lines.append("\t".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_prepare_oligoformer_run_writes_volume_inputs(tmp_path: Path, monkeypatch):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))

    result = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        mrna_fasta_bytes=b">target one\nAUGCUAGCUAGCUAGCUAGCUAGC\n",
        sirna_fasta_bytes=b">s\nAUGCUAGCUAGCUAGCUAG\n",
        off_target=True,
        utr_bytes=b">utr\nAUGC\n",
        orf_bytes=b">orf\nAUGC\n",
    )

    input_dir = oligoformer_app.AppRunLayout.from_run_root(result.run_root).inputs_dir
    assert result.output_stems == ("target_@_one",)
    assert Path(result.run_root) == (
        Path(tmp_path / "outputs-volume") / "target_@_one" / result.cache_key
    )
    assert input_dir.joinpath("mrna.fa").read_bytes().startswith(b">target_@_one")
    assert input_dir.joinpath("sirna.fa").read_bytes().startswith(b">s")
    assert input_dir.joinpath("utr.txt").read_bytes() == b">utr\nAUGC\n"
    assert input_dir.joinpath("orf.txt").read_bytes() == b">orf\nAUGC\n"
    assert result.efficacy_ready is False
    assert result.final_ready is False
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_prepare_oligoformer_run_sanitizes_fasta_names_and_rewrites_input(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))

    result = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        mrna_fasta_bytes=b">../../target; touch unsafe\nAUGCUAGCUAGCUAGCUAGC\n",
    )

    assert result.output_stems == ("target_@_touch_@_unsafe",)
    input_path = oligoformer_app.AppRunLayout.from_run_root(
        result.run_root
    ).inputs_dir.joinpath("mrna.fa")
    assert input_path.read_bytes() == (
        b">target_@_touch_@_unsafe\nAUGCUAGCUAGCUAGCUAGC\n"
    )


def test_prepare_oligoformer_run_bounds_long_fasta_names(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    raw_name = "record-" + "x" * 300

    result = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        mrna_fasta_bytes=f">{raw_name}\nAUGCUAGCUAGCUAGCUAGC\n".encode(),
    )

    (safe_name,) = result.output_stems
    assert len(safe_name) <= 180
    assert safe_name.endswith(oligoformer_app.hash_string(raw_name)[:16])
    input_path = oligoformer_app.AppRunLayout.from_run_root(
        result.run_root
    ).inputs_dir.joinpath("mrna.fa")
    assert input_path.read_text(encoding="utf-8").startswith(f">{safe_name}\n")


@pytest.mark.parametrize(
    "fasta_bytes",
    [
        b">same\nAAAA\n>same\nCCCC\n",
        b">a/b\nAAAA\n>a_b\nCCCC\n",
    ],
)
def test_prepare_oligoformer_run_rejects_duplicate_or_colliding_fasta_names(
    tmp_path: Path, monkeypatch, fasta_bytes: bytes
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))

    with pytest.raises(ValueError, match="unique after sanitization"):
        oligoformer_app.prepare_oligoformer_run.get_raw_f()(
            mrna_fasta_bytes=fasta_bytes,
        )


def test_prepare_oligoformer_run_sanitizes_paired_custom_reference_names(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))

    result = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        mrna_fasta_bytes=b">target\nAUGCUAGCUAGCUAGCUAGC\n",
        off_target=True,
        utr_bytes=b">../../tx; one\naugtn\n",
        orf_bytes=b">../../tx; one\nAUGTN\n",
    )

    input_dir = oligoformer_app.AppRunLayout.from_run_root(result.run_root).inputs_dir
    assert input_dir.joinpath("utr.txt").read_bytes() == b">tx_@_one\nAUGTN\n"
    assert input_dir.joinpath("orf.txt").read_bytes() == b">tx_@_one\nAUGTN\n"


@pytest.mark.parametrize(
    ("utr_bytes", "orf_bytes", "message"),
    [
        (
            b">a/b\nAUGC\n>a_b\nAUGC\n",
            b">a/b\nAUGC\n",
            "unique after sanitization",
        ),
        (b">tx\nAUGZ\n", b">tx\nAUGC\n", "IUPAC nucleotide"),
        (b">a/b\nAUGC\n", b">a_b\nAUGC\n", "same original name"),
    ],
)
def test_prepare_oligoformer_run_rejects_unsafe_custom_references(
    tmp_path: Path,
    monkeypatch,
    utr_bytes: bytes,
    orf_bytes: bytes,
    message: str,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))

    with pytest.raises(ValueError, match=message):
        oligoformer_app.prepare_oligoformer_run.get_raw_f()(
            mrna_fasta_bytes=b">target\nAUGCUAGCUAGCUAGCUAGC\n",
            off_target=True,
            utr_bytes=utr_bytes,
            orf_bytes=orf_bytes,
        )


def test_prepare_oligoformer_run_reuses_compute_cache_across_final_thresholds(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    shared = {
        "mrna_fasta_bytes": b">target\nAUGCUAGCUAGCUAGCUAGCUAGC\n",
        "off_target": True,
        "utr_bytes": b">utr\nAUGC\n",
        "orf_bytes": b">orf\nAUGC\n",
    }

    first = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        **shared,
        pita_threshold=-10.0,
        toxicity=False,
    )
    second = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        **shared,
        pita_threshold=-5.0,
        toxicity=True,
    )

    assert first.cache_key == second.cache_key
    assert first.postprocess_key != second.postprocess_key
    assert first.run_root == second.run_root
    assert first.efficacy_dir == second.efficacy_dir
    assert first.output_dir != second.output_dir


def test_prepare_oligoformer_run_reuses_efficacy_across_evidence_inputs(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    shared = {
        "mrna_fasta_bytes": b">target\nAUGCUAGCUAGCUAGCUAGCUAGC\n",
        "off_target": True,
        "orf_bytes": b">orf\nAUGC\n",
    }

    first = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        **shared,
        utr_bytes=b">utr\nAUGC\n",
        top_n=20,
    )
    second = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        **shared,
        utr_bytes=b">utr\nAUGCAUGC\n",
        top_n=40,
    )

    assert first.efficacy_key == second.efficacy_key
    assert first.efficacy_dir == second.efficacy_dir
    assert first.cache_key != second.cache_key
    assert first.run_root != second.run_root


def test_force_uses_isolated_cache_generation_without_deleting_shared_cache(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    kwargs = {"mrna_fasta_bytes": b">target\nAUGCUAGCUAGCUAGCUAGCUAGC\n"}

    shared = oligoformer_app.prepare_oligoformer_run.get_raw_f()(**kwargs)
    forced = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        **kwargs,
        force=True,
        force_generation="force-run",
    )
    repeated_plan = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        **kwargs,
        force=True,
        force_generation="force-run",
    )

    assert forced.efficacy_key != shared.efficacy_key
    assert forced.cache_key != shared.cache_key
    assert forced.run_root != shared.run_root
    assert repeated_plan == forced
    assert (
        oligoformer_app.AppRunLayout
        .from_run_root(shared.run_root)
        .inputs_dir.joinpath("mrna.fa")
        .exists()
    )


def test_all_human_evidence_key_uses_converted_reference_content_identity(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    identity_path = oligoformer_app.APP_INFO.targetscan_ref_identity_path
    identity_path.parent.mkdir(parents=True)
    shared_identity = oligoformer_app.APP_INFO.targetscan_ref_metadata | {
        "row_counts": {"human_UTR.txt": 1, "human_ORF.txt": 1},
    }
    shared = {
        "mrna_fasta_bytes": b">target\nAUGCUAGCUAGCUAGCUAGCUAGC\n",
        "off_target": True,
        "all_human": True,
    }

    identity_path.write_bytes(
        orjson.dumps(
            shared_identity
            | {
                "content_sha256": {
                    "human_UTR.txt": "utr-v1",
                    "human_ORF.txt": "orf-v1",
                }
            }
        )
    )
    first = oligoformer_app.prepare_oligoformer_run.get_raw_f()(**shared)
    identity_path.write_bytes(
        orjson.dumps(
            shared_identity
            | {
                "content_sha256": {
                    "human_UTR.txt": "utr-v2",
                    "human_ORF.txt": "orf-v1",
                }
            }
        )
    )
    second = oligoformer_app.prepare_oligoformer_run.get_raw_f()(**shared)

    assert first.efficacy_key == second.efficacy_key
    assert first.reference_identity != second.reference_identity
    assert first.cache_key != second.cache_key


def test_targetscan_rnaplfold_cache_uses_output_volume_v2(tmp_path: Path, monkeypatch):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    info = oligoformer_app.AppInfo(
        model_ref_dir=tmp_path / "models" / "off-target" / "ref"
    )

    cache_root = Path(tmp_path / "outputs-volume" / "reference-cache")
    assert info.targetscan_rnaplfold_cache_dir.is_relative_to(cache_root)
    assert info.targetscan_rnaplfold_shard_dir.is_relative_to(cache_root)
    assert info.targetscan_rnaplfold_marker_path.is_relative_to(cache_root)


def test_targetscan_rnaplfold_cache_identity_includes_converted_utr_digest(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    info = oligoformer_app.AppInfo(
        model_ref_dir=tmp_path / "models" / "off-target" / "ref"
    )
    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    identity = info.targetscan_ref_metadata | {
        "content_sha256": {
            "human_UTR.txt": "utr-v1",
            "human_ORF.txt": "orf-v1",
        }
    }
    info.targetscan_ref_identity_path.parent.mkdir(parents=True)
    info.targetscan_ref_identity_path.write_bytes(orjson.dumps(identity))
    info.targetscan_rnaplfold_cache_dir.mkdir(parents=True)
    info.targetscan_rnaplfold_shard_dir.mkdir(parents=True)
    spec = oligoformer_app._targetscan_rnaplfold_shard_spec(0)
    Path(spec.shard_path).write_text(">ENST1\nAUGC\n", encoding="utf-8")
    info.targetscan_rnaplfold_cache_dir.joinpath("ENST1.9606_lunp").write_text(
        "cache\n", encoding="utf-8"
    )
    oligoformer_app._publish_targetscan_rnaplfold_shard_manifest(spec)
    oligoformer_app._publish_targetscan_rnaplfold_cache_marker(
        [spec],
        record_count=1,
    )

    assert oligoformer_app._targetscan_rnaplfold_cache_ready()

    info.targetscan_ref_identity_path.write_bytes(
        orjson.dumps(
            identity
            | {
                "content_sha256": {
                    "human_UTR.txt": "utr-v2",
                    "human_ORF.txt": "orf-v1",
                }
            }
        )
    )
    assert not oligoformer_app._targetscan_rnaplfold_cache_ready()


@pytest.mark.parametrize("damage", ["delete", "truncate"])
def test_targetscan_rnaplfold_cache_rejects_damaged_non_sample_output(
    tmp_path: Path,
    monkeypatch,
    damage: str,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    info = oligoformer_app.AppInfo(
        model_ref_dir=tmp_path / "models" / "off-target" / "ref"
    )
    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    identity = info.targetscan_ref_metadata | {
        "content_sha256": {
            "human_UTR.txt": "utr-v1",
            "human_ORF.txt": "orf-v1",
        }
    }
    info.targetscan_ref_identity_path.parent.mkdir(parents=True)
    info.targetscan_ref_identity_path.write_bytes(orjson.dumps(identity))
    info.targetscan_rnaplfold_cache_dir.mkdir(parents=True)
    info.targetscan_rnaplfold_shard_dir.mkdir(parents=True)
    spec = oligoformer_app._targetscan_rnaplfold_shard_spec(0)
    records = [(f"ENST{index}", "AUGC") for index in range(5)]
    Path(spec.shard_path).write_text(
        "".join(f">{name}\n{sequence}\n" for name, sequence in records),
        encoding="utf-8",
    )
    for name, _ in records:
        info.targetscan_rnaplfold_cache_dir.joinpath(f"{name}.9606_lunp").write_text(
            f"cache-{name}\n", encoding="utf-8"
        )
    oligoformer_app._publish_targetscan_rnaplfold_shard_manifest(spec)
    oligoformer_app._publish_targetscan_rnaplfold_cache_marker(
        [spec],
        record_count=len(records),
    )
    assert oligoformer_app._targetscan_rnaplfold_cache_ready()

    damaged = info.targetscan_rnaplfold_cache_dir / "ENST1.9606_lunp"
    if damage == "delete":
        damaged.unlink()
    else:
        damaged.write_text("x", encoding="utf-8")

    assert not oligoformer_app._targetscan_rnaplfold_cache_ready()


def test_cache_build_lock_reclaims_stale_generation_without_deleting_owner(
    monkeypatch,
):
    info = replace(
        oligoformer_app.APP_INFO,
        cache_lock_poll_seconds=0,
        cache_lock_stale_seconds=-1,
    )
    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    lock_key = oligoformer_app.hash_string("stage\nidentity")
    old_owner = {"id": "old", "acquired_at": 0.0}
    values = {f"{lock_key}:owner:0": old_owner}

    class FakeDictInstance:
        def put(self, key, value, *, skip_if_exists=False):
            if skip_if_exists and key in values:
                return False
            values[key] = value
            return True

        def get(self, key, default=None):
            return values.get(key, default)

    class FakeDict:
        @staticmethod
        def from_name(_name, create_if_missing=False):
            assert create_if_missing
            return FakeDictInstance()

    monkeypatch.setattr(oligoformer_app.modal, "Dict", FakeDict)

    with oligoformer_app._cache_build_lock("stage", "identity") as owns_build:
        assert owns_build

    assert values[f"{lock_key}:owner:0"] == old_owner
    assert values[f"{lock_key}:status:0"]["state"] == "abandoned"
    repair_status = values[f"{lock_key}:status:1"]
    assert isinstance(repair_status, dict)
    assert repair_status["state"] == "complete"

    with oligoformer_app._cache_build_lock(
        "stage", "identity", rebuild=True
    ) as owns_rebuild:
        assert owns_rebuild

    assert values[f"{lock_key}:status:2"]["state"] == "complete"


def test_cache_build_lock_rebuild_waiter_follows_existing_next_generation(
    monkeypatch,
):
    lock_key = oligoformer_app.hash_string("stage\nidentity")
    values = {
        f"{lock_key}:head": 0,
        f"{lock_key}:owner:0": {"id": "original", "acquired_at": 0.0},
        f"{lock_key}:status:0": {"state": "complete"},
        f"{lock_key}:owner:1": {"id": "repair", "acquired_at": 1.0},
        f"{lock_key}:status:1": {"state": "complete"},
    }

    class FakeDictInstance:
        def put(self, key, value, *, skip_if_exists=False):
            if skip_if_exists and key in values:
                return False
            values[key] = value
            return True

        def get(self, key, default=None):
            return values.get(key, default)

    class FakeDict:
        @staticmethod
        def from_name(_name, create_if_missing=False):
            assert create_if_missing
            return FakeDictInstance()

    monkeypatch.setattr(oligoformer_app.modal, "Dict", FakeDict)

    with oligoformer_app._cache_build_lock(
        "stage", "identity", rebuild=True
    ) as owns_rebuild:
        assert not owns_rebuild

    assert f"{lock_key}:owner:2" not in values


def test_cache_build_lock_exclusive_guards_own_sequential_generations(monkeypatch):
    lock_key = oligoformer_app.hash_string("stage\nidentity")
    values = {
        f"{lock_key}:head": 0,
        f"{lock_key}:owner:0": {"id": "initial", "acquired_at": 0.0},
        f"{lock_key}:status:0": {"state": "complete"},
    }

    class FakeDictInstance:
        def put(self, key, value, *, skip_if_exists=False):
            if skip_if_exists and key in values:
                return False
            values[key] = value
            return True

        def get(self, key, default=None):
            return values.get(key, default)

    class FakeDict:
        @staticmethod
        def from_name(_name, create_if_missing=False):
            assert create_if_missing
            return FakeDictInstance()

    monkeypatch.setattr(oligoformer_app.modal, "Dict", FakeDict)

    for expected_generation in (1, 2):
        with oligoformer_app._cache_build_lock(
            "stage",
            "identity",
            rebuild=True,
            coalesce_rebuild=False,
        ) as owns_rebuild:
            assert owns_rebuild
        status = values[f"{lock_key}:status:{expected_generation}"]
        assert isinstance(status, dict)
        assert status["state"] == "complete"


def test_final_ready_requires_current_postprocess_marker_salt(tmp_path: Path):
    layout = oligoformer_app.AppRunLayout.from_run_root(tmp_path / "run")
    postprocess_key = "final-key"
    output_dir = layout.outputs_dir / postprocess_key
    _write_valid_output_bundle(output_dir)
    layout.markers_dir.mkdir(parents=True)

    marker_name = oligoformer_app._final_marker_name(postprocess_key)
    oligoformer_app._marker_path(layout, marker_name).write_bytes(
        orjson.dumps({
            "cache_key": "abc123",
            "output_stems": ["target"],
        })
    )
    assert not oligoformer_app._build_plan(
        "abc123",
        "efficacy123",
        ("target",),
        layout.run_root,
        config=_run_config(),
        postprocess_key=postprocess_key,
    ).final_ready

    stale_plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        efficacy_key="efficacy123",
        run_root=str(layout.run_root),
        efficacy_dir=str(layout.prep_dir / "efficacy"),
        output_dir=str(output_dir),
        output_stems=("target",),
        config=_run_config(),
        postprocess_key=postprocess_key,
        efficacy_ready=False,
        evidence_ready=False,
        final_ready=False,
    )
    oligoformer_app._write_cache_marker(
        layout,
        marker_name,
        stale_plan,
        extra_metadata={
            "postprocess_cache_salt": oligoformer_app.APP_INFO.postprocess_cache_salt
        },
    )

    assert oligoformer_app._build_plan(
        "abc123",
        "efficacy123",
        ("target",),
        layout.run_root,
        config=_run_config(),
        postprocess_key=postprocess_key,
    ).final_ready

    output_dir.joinpath("target_ranked.txt").write_text(
        "pos\tsense\tsiRNA\tefficacy\tfunc_filter\tfilter\n",
        encoding="utf-8",
    )
    assert not oligoformer_app._build_plan(
        "abc123",
        "efficacy123",
        ("target",),
        layout.run_root,
        config=_run_config(),
        postprocess_key=postprocess_key,
    ).final_ready


def test_efficacy_ready_rejects_corrupt_manifested_table(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    layout = oligoformer_app._efficacy_layout_for_key("efficacy123", ("target",))
    _write_valid_output_bundle(layout.outputs_dir)
    _publish_efficacy_marker(layout, layout.outputs_dir)

    assert oligoformer_app._build_plan(
        "abc123",
        "efficacy123",
        ("target",),
        tmp_path / "run",
        config=_run_config(),
        postprocess_key="postprocess",
    ).efficacy_ready

    layout.outputs_dir.joinpath("target.txt").write_text(
        "pos\tsense\tsiRNA\tefficacy\tfunc_filter\tfilter\n",
        encoding="utf-8",
    )
    assert not oligoformer_app._build_plan(
        "abc123",
        "efficacy123",
        ("target",),
        tmp_path / "run",
        config=_run_config(),
        postprocess_key="postprocess",
    ).efficacy_ready


def test_run_oligoformer_efficacy_builds_gpu_stage_command(tmp_path: Path, monkeypatch):
    captured = {}
    volume = FakeVolume()
    model_volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    info = oligoformer_app.AppInfo(
        repo_rnafm_dir=repo_dir / "RNA-FM",
        model_rnafm_dir=tmp_path / "models" / "RNA-FM",
        repo_ref_dir=repo_dir / "off-target" / "ref",
        model_ref_dir=tmp_path / "models" / "off-target" / "ref",
    )
    info.model_rnafm_redevelop_dir.mkdir(parents=True)
    info.model_rnafm_redevelop_dir.joinpath("model.pt").write_text(
        "weights", encoding="utf-8"
    )
    input_dir = tmp_path / "run" / "inputs"
    efficacy_dir = (
        tmp_path
        / "outputs-volume"
        / "efficacy-cache"
        / "target"
        / "efficacy123"
        / "outputs"
    )
    input_dir.mkdir(parents=True)
    input_dir.joinpath("mrna.fa").write_text(">target\nAUGCUAGCUAGCUAGCUAGC\n")
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        efficacy_key="efficacy123",
        run_root=str(tmp_path / "run"),
        efficacy_dir=str(efficacy_dir),
        output_dir=str(tmp_path / "run" / "outputs" / "postprocess"),
        output_stems=("target",),
        config=_run_config(functionality_filter=False),
        postprocess_key="postprocess",
        efficacy_ready=False,
        evidence_ready=True,
        final_ready=False,
        model_identity="model-v1",
    )

    def fake_run_command(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_valid_output_bundle(out_dir)

    lock_values = {}

    class FakeDictInstance:
        def put(self, key, value, *, skip_if_exists=False):
            if skip_if_exists and key in lock_values:
                return False
            lock_values[key] = value
            return True

        def get(self, key, default=None):
            return lock_values.get(key, default)

        def pop(self, key, default=None):
            return lock_values.pop(key, default)

    class FakeDict:
        @staticmethod
        def from_name(_name, create_if_missing=False):
            assert create_if_missing
            return FakeDictInstance()

    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    monkeypatch.setattr(oligoformer_app, "MODEL_VOLUME", model_volume)
    monkeypatch.setattr(oligoformer_app, "run_command", fake_run_command)
    monkeypatch.setattr(
        oligoformer_app,
        "_rnafm_model_identity_matches_model",
        lambda: True,
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_rnafm_model_identity_digest",
        lambda: "model-v1",
    )
    monkeypatch.setattr(oligoformer_app.modal, "Dict", FakeDict)

    result = oligoformer_app.run_oligoformer_efficacy.get_raw_f()(
        plan=plan,
        functionality_filter=False,
    )

    assert captured["cmd"][:5] == ["python", "scripts/main.py", "-i", "1", "-i1"]
    assert "--biomodals_stage" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--biomodals_stage") + 1] == "efficacy"
    assert "--no_func" in captured["cmd"]
    assert "-off" not in captured["cmd"]
    assert "-tox" not in captured["cmd"]
    assert captured["cwd"] == repo_dir
    assert result.efficacy_ready is True
    assert any(
        isinstance(value, dict) and value.get("state") == "complete"
        for value in lock_values.values()
    )
    assert volume.reload_count == 2
    assert volume.commit_count == 1
    assert model_volume.reload_count == 1


def test_rnafm_runtime_keeps_relative_data_in_writable_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    repo_dir = tmp_path / "repo"
    info = oligoformer_app.AppInfo(
        repo_rnafm_dir=repo_dir / "RNA-FM",
        model_rnafm_dir=tmp_path / "models" / "RNA-FM",
    )
    info.model_rnafm_redevelop_dir.mkdir(parents=True)
    info.model_rnafm_redevelop_dir.joinpath("model.pt").write_text(
        "weights", encoding="utf-8"
    )
    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)

    oligoformer_app._ensure_rnafm_runtime()

    assert not info.repo_rnafm_dir.is_symlink()
    assert (
        info.repo_rnafm_redevelop_dir.joinpath("model.pt").read_text(encoding="utf-8")
        == "weights"
    )
    assert (
        info.repo_rnafm_redevelop_dir.joinpath("../../data").resolve()
        == (repo_dir / "data").resolve()
    )


def test_run_oligoformer_efficacy_rejects_mismatched_plan_settings():
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        efficacy_key="efficacy123",
        run_root="/remote/run",
        efficacy_dir="/remote/run/prepare/efficacy",
        output_dir="/remote/run/outputs/postprocess",
        output_stems=("target",),
        config=_run_config(functionality_filter=False),
        postprocess_key="postprocess",
        efficacy_ready=False,
        evidence_ready=True,
        final_ready=False,
    )

    with pytest.raises(ValueError, match="efficacy settings"):
        oligoformer_app.run_oligoformer_efficacy.get_raw_f()(plan)


def test_run_oligoformer_efficacy_rechecks_cache_after_lock(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    efficacy_layout = oligoformer_app._efficacy_layout_for_key(
        "efficacy123", ("target",)
    )
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        efficacy_key="efficacy123",
        run_root=str(tmp_path / "run"),
        efficacy_dir=str(efficacy_layout.outputs_dir),
        output_dir=str(tmp_path / "run" / "outputs" / "postprocess"),
        output_stems=("target",),
        config=_run_config(),
        postprocess_key="postprocess",
        efficacy_ready=False,
        evidence_ready=True,
        final_ready=False,
    )

    class PublishEfficacyOnEnter:
        def __enter__(self):
            _write_valid_output_bundle(efficacy_layout.outputs_dir)
            _publish_efficacy_marker(
                efficacy_layout,
                efficacy_layout.outputs_dir,
            )

        def __exit__(self, *_args):
            return False

    def fake_cache_build_lock(stage, identity, *, rebuild=False):
        assert (stage, identity) == ("efficacy", "efficacy123")
        assert rebuild
        return PublishEfficacyOnEnter()

    def exploding_run_command(*_args, **_kwargs):
        raise AssertionError("the cache published by the lock holder must be reused")

    monkeypatch.setattr(oligoformer_app, "_cache_build_lock", fake_cache_build_lock)
    monkeypatch.setattr(oligoformer_app, "run_command", exploding_run_command)

    result = oligoformer_app.run_oligoformer_efficacy.get_raw_f()(plan)

    assert result.efficacy_ready
    assert volume.reload_count == 2
    assert volume.commit_count == 0


def test_run_oligoformer_postprocess_rejects_mismatched_plan_settings():
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        efficacy_key="efficacy123",
        run_root="/remote/run",
        efficacy_dir="/remote/run/prepare/efficacy",
        output_dir="/remote/run/outputs/postprocess",
        output_stems=("target",),
        config=_run_config(),
        postprocess_key="postprocess",
        efficacy_ready=True,
        evidence_ready=True,
        final_ready=False,
    )

    with pytest.raises(ValueError, match="post-processing settings"):
        oligoformer_app.run_oligoformer_postprocess.get_raw_f()(
            plan,
            off_target=True,
        )


def test_run_postprocess_rechecks_final_cache_after_lock(tmp_path: Path, monkeypatch):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    layout = oligoformer_app.AppRunLayout.from_run_root(tmp_path / "run")
    output_dir = layout.outputs_dir / "postprocess"
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        efficacy_key="efficacy123",
        run_root=str(layout.run_root),
        efficacy_dir=str(tmp_path / "efficacy"),
        output_dir=str(output_dir),
        output_stems=("target",),
        config=_run_config(),
        postprocess_key="postprocess",
        efficacy_ready=False,
        evidence_ready=True,
        final_ready=False,
    )

    class PublishFinalOnEnter:
        def __enter__(self):
            _write_valid_output_bundle(output_dir)
            oligoformer_app._write_cache_marker(
                layout,
                oligoformer_app._final_marker_name(plan.postprocess_key),
                plan,
                extra_metadata={
                    "postprocess_cache_salt": (
                        oligoformer_app.APP_INFO.postprocess_cache_salt
                    )
                },
            )

        def __exit__(self, *_args):
            return False

    def fake_cache_build_lock(*_args, **kwargs):
        assert kwargs["rebuild"]
        return PublishFinalOnEnter()

    monkeypatch.setattr(
        oligoformer_app,
        "_cache_build_lock",
        fake_cache_build_lock,
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_package_output_tables",
        lambda *_args, **_kwargs: b"archive",
    )

    result = oligoformer_app.run_oligoformer_postprocess.get_raw_f()(plan)

    assert result == b"archive"
    assert volume.reload_count == 2


def test_result_publication_is_fingerprint_bound_and_digested(tmp_path: Path) -> None:
    archive = tmp_path / "run" / "outputs" / "oligoformer.tar.zst"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"archive")
    publication_key = "a" * 64

    oligoformer_app._publish_oligoformer_result_record(
        tmp_path,
        publication_key,
        archive,
        model_identity="model-content-v1",
        reference_identity=None,
    )

    publication = oligoformer_app._oligoformer_result_publication(
        tmp_path,
        publication_key,
    )
    assert publication is not None
    assert publication["result_path"] == "run/outputs/oligoformer.tar.zst"
    assert publication["model_identity"] == "model-content-v1"
    assert oligoformer_app._oligoformer_result_publication(tmp_path, "b" * 64) is None

    class VolumeReader:
        def read_file(self, path):
            yield tmp_path.joinpath(*Path(path).parts).read_bytes()

    assert (
        oligoformer_app._oligoformer_result_publication_from_volume(
            VolumeReader(),
            publication_key,
        )
        == publication
    )

    archive.write_bytes(b"damaged")

    assert (
        oligoformer_app._oligoformer_result_publication(tmp_path, publication_key)
        is None
    )


def test_result_publication_rejects_changed_model_identity(tmp_path: Path) -> None:
    archive = tmp_path / "run" / "outputs" / "oligoformer.tar.zst"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"archive")
    publication_key = "a" * 64
    oligoformer_app._publish_oligoformer_result_record(
        tmp_path,
        publication_key,
        archive,
        model_identity="model-content-v1",
        reference_identity="reference-content-v1",
    )

    assert (
        oligoformer_app._oligoformer_result_publication(
            tmp_path,
            publication_key,
            expected_identities=("model-content-v2", "reference-content-v1"),
        )
        is None
    )


def test_run_postprocess_rejects_changed_all_human_reference_identity(
    tmp_path: Path, monkeypatch
):
    output_volume = FakeVolume()
    model_volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, output_volume))
    monkeypatch.setattr(oligoformer_app, "MODEL_VOLUME", model_volume)
    efficacy_layout = oligoformer_app._efficacy_layout_for_key(
        "efficacy123", ("target",)
    )
    _write_valid_output_bundle(efficacy_layout.outputs_dir)
    _publish_efficacy_marker(
        efficacy_layout,
        efficacy_layout.outputs_dir,
    )
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        efficacy_key="efficacy123",
        run_root=str(tmp_path / "run"),
        efficacy_dir=str(efficacy_layout.outputs_dir),
        output_dir=str(tmp_path / "run" / "outputs" / "postprocess"),
        output_stems=("target",),
        config=_run_config(off_target=True, all_human=True),
        postprocess_key="postprocess",
        efficacy_ready=True,
        evidence_ready=False,
        final_ready=False,
        reference_identity="reference-v1",
    )

    monkeypatch.setattr(oligoformer_app, "_ensure_human_refs", lambda: None)
    monkeypatch.setattr(
        oligoformer_app, "_targetscan_rnaplfold_cache_ready", lambda: True
    )
    monkeypatch.setattr(
        oligoformer_app, "_targetscan_ref_identity_digest", lambda: "reference-v2"
    )
    monkeypatch.setattr(
        oligoformer_app, "_targetscan_ref_identity_matches_model", lambda: True
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_cache_build_lock",
        lambda *_args, **_kwargs: nullcontext(True),
    )

    with pytest.raises(FileNotFoundError, match="changed after run preparation"):
        oligoformer_app.run_oligoformer_postprocess.get_raw_f()(
            plan,
            off_target=True,
            all_human=True,
        )


@pytest.mark.parametrize(
    ("process_slots", "message"),
    [(1, "at least 2"), (65, "must not exceed")],
)
def test_execution_config_rejects_unsafe_process_slot_budget(
    process_slots: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        _execution_config(off_target_process_slots=process_slots)


@pytest.mark.parametrize(
    "field_name",
    oligoformer_app.OligoformerExecutionConfig.__slots__,
)
def test_execution_config_rejects_nonpositive_values(field_name: str):
    with pytest.raises(ValueError, match=field_name):
        _execution_config(**{field_name: 0})


@pytest.mark.parametrize(
    "field_name",
    [
        "off_target_workers",
        "off_target_prep_workers",
        "pita_prepare_workers",
        "targetscan_rnaplfold_nodes",
        "targetscan_rnaplfold_workers",
        "targetscan_context_workers",
    ],
)
def test_execution_config_rejects_cpu_envelope_overrides(field_name: str):
    with pytest.raises(ValueError, match=field_name):
        _execution_config(**{field_name: 33})


def test_runtime_image_does_not_bake_oligoformer_tuning_environment():
    source = Path(oligoformer_app.__file__).read_text(encoding="utf-8")

    assert 'os.environ.get("OLIGOFORMER_' not in source
    assert "tuning_env_names" not in source


def test_off_target_manifest_detects_corrupt_evidence_before_cleanup(tmp_path: Path):
    raw_dir = tmp_path / "run" / "prepare" / "off_target" / "target"
    raw_dir.mkdir(parents=True)
    raw_dir.joinpath("pita.tab").write_text(
        "RefSeq\tmicroRNA\tSites\tScore\nref\tRNA0\t1\t-1\n",
        encoding="utf-8",
    )
    targetscan_path = raw_dir / "targetscan.tab"
    targetscan_path.write_text("ref\tRNA0\t0.5\n", encoding="utf-8")
    identity = "evidence-v1"

    oligoformer_app._publish_off_target_manifest(raw_dir, identity=identity)

    assert oligoformer_app._raw_off_target_ready(
        raw_dir,
        expected_identity=identity,
    )
    manifest = orjson.loads(raw_dir.joinpath("off_target.done").read_bytes())
    assert manifest["version"] == 1
    assert manifest["identity"] == identity
    assert manifest["tables"]["pita.tab"]["columns"] == [
        "RefSeq",
        "microRNA",
        "Sites",
        "Score",
    ]
    assert manifest["tables"]["pita.tab"]["row_count"] == 1

    targetscan_path.write_text(
        "refseq\tsiRNA\ttargetscan_score\n",
        encoding="utf-8",
    )

    assert not oligoformer_app._raw_off_target_ready(
        raw_dir,
        expected_identity=identity,
    )
    with pytest.raises(RuntimeError, match="validated off-target evidence"):
        oligoformer_app._cleanup_off_target_transients(raw_dir)


def test_off_target_cleanup_tolerates_concurrent_child_deletion(
    tmp_path: Path,
    monkeypatch,
):
    raw_dir = tmp_path / "run" / "prepare" / "off_target" / "target"
    raw_dir.mkdir(parents=True)
    raw_dir.joinpath("pita.tab").write_text(
        "RefSeq\tmicroRNA\tSites\tScore\n",
        encoding="utf-8",
    )
    raw_dir.joinpath("targetscan.tab").write_text(
        "refseq\tsiRNA\ttargetscan_score\n",
        encoding="utf-8",
    )
    transient_dir = raw_dir / "shards"
    transient_dir.mkdir()
    transient_dir.joinpath("part.tsv").write_text("row\n", encoding="utf-8")
    oligoformer_app._publish_off_target_manifest(raw_dir, identity="race")
    real_rmtree = shutil.rmtree
    ignore_errors_seen = []

    def racing_rmtree(path, *args, ignore_errors=False, **kwargs):
        if Path(path) == transient_dir and transient_dir.exists():
            real_rmtree(transient_dir)
        ignore_errors_seen.append(ignore_errors)
        return real_rmtree(path, *args, ignore_errors=ignore_errors, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", racing_rmtree)

    oligoformer_app._cleanup_off_target_transients(raw_dir)

    assert ignore_errors_seen == [True]
    assert raw_dir.joinpath("off_target.done").is_file()


def test_off_target_manifest_accepts_no_hit_pita_evidence(tmp_path: Path):
    raw_dir = tmp_path / "run" / "prepare" / "off_target" / "target"
    raw_dir.mkdir(parents=True)
    pita_path = raw_dir / "pita.tab"
    oligoformer_app._merge_pita_shards([], pita_path)
    raw_dir.joinpath("targetscan.tab").write_text(
        "refseq\tsiRNA\ttargetscan_score\n",
        encoding="utf-8",
    )

    oligoformer_app._publish_off_target_manifest(raw_dir, identity="no-hits")

    assert pita_path.read_text(encoding="utf-8") == ("RefSeq\tmicroRNA\tSites\tScore\n")
    assert oligoformer_app._raw_off_target_ready(
        raw_dir,
        expected_identity="no-hits",
    )
    manifest = orjson.loads(raw_dir.joinpath("off_target.done").read_bytes())
    assert manifest["tables"]["pita.tab"]["row_count"] == 0


def test_orphaned_off_target_tables_are_discarded_without_marker(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "run" / "prepare" / "off_target" / "target"
    raw_dir.mkdir(parents=True)
    raw_dir.joinpath("pita.tab").write_text("orphaned\n", encoding="utf-8")
    raw_dir.joinpath("targetscan.tab").write_text("orphaned\n", encoding="utf-8")

    assert oligoformer_app._discard_invalid_off_target_evidence(
        raw_dir,
        expected_identity="evidence-v1",
    )
    assert not raw_dir.joinpath("pita.tab").exists()
    assert not raw_dir.joinpath("targetscan.tab").exists()


def test_corrupt_completed_evidence_advances_one_repair_generation(
    tmp_path: Path,
    monkeypatch,
):
    raw_dir = tmp_path / "run" / "prepare" / "off_target" / "target"
    raw_dir.mkdir(parents=True)
    pita_contents = "RefSeq\tmicroRNA\tSites\tScore\nref\tRNA0\t1\t-1\n"
    targetscan_contents = "ref\tRNA0\t0.5\n"
    raw_dir.joinpath("pita.tab").write_text(pita_contents, encoding="utf-8")
    raw_dir.joinpath("targetscan.tab").write_text(
        targetscan_contents,
        encoding="utf-8",
    )
    identity = "evidence-v1"
    oligoformer_app._publish_off_target_manifest(raw_dir, identity=identity)
    raw_dir.joinpath("targetscan.tab").write_text("truncated\n", encoding="utf-8")

    lock_identity = "run\ntarget"
    lock_key = oligoformer_app.hash_string(f"off-target-evidence\n{lock_identity}")
    values = {
        f"{lock_key}:head": 0,
        f"{lock_key}:owner:0": {"id": "original", "acquired_at": 0.0},
        f"{lock_key}:status:0": {"state": "complete"},
    }

    class FakeDictInstance:
        def put(self, key, value, *, skip_if_exists=False):
            if skip_if_exists and key in values:
                return False
            values[key] = value
            return True

        def get(self, key, default=None):
            return values.get(key, default)

    class FakeDict:
        @staticmethod
        def from_name(_name, create_if_missing=False):
            assert create_if_missing
            return FakeDictInstance()

    monkeypatch.setattr(oligoformer_app.modal, "Dict", FakeDict)

    with oligoformer_app._cache_build_lock(
        "off-target-evidence",
        lock_identity,
        rebuild=True,
    ) as owns_repair:
        assert owns_repair
        assert oligoformer_app._discard_invalid_off_target_evidence(
            raw_dir,
            expected_identity=identity,
        )
        raw_dir.joinpath("pita.tab").write_text(pita_contents, encoding="utf-8")
        raw_dir.joinpath("targetscan.tab").write_text(
            targetscan_contents,
            encoding="utf-8",
        )
        oligoformer_app._publish_off_target_manifest(raw_dir, identity=identity)

    assert oligoformer_app._raw_off_target_ready(
        raw_dir,
        expected_identity=identity,
    )
    repair_status = values[f"{lock_key}:status:1"]
    assert isinstance(repair_status, dict)
    assert repair_status["state"] == "complete"
    assert f"{lock_key}:owner:2" not in values


def test_apply_off_target_filters_handles_header_only_pita(tmp_path: Path, monkeypatch):
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))

    run_root = tmp_path / "run"
    _publish_test_off_target_evidence(
        run_root,
        stem="target",
        pita="RefSeq\tmicroRNA\tSites\tScore\n",
        targetscan="ref\tRNA0\t0.5\n",
    )

    result = pl.DataFrame({
        "pos": [1, 2],
        "sense": ["GC", "CG"],
        "siRNA": ["AUGCUAGCUAGCUAGCUAG", "UGCUAGCUAGCUAGCUAGC"],
        "efficacy": [0.8, 0.6],
        "func_filter": [0, 0],
    })

    filtered = oligoformer_app._apply_off_target_filters(
        result=result,
        run_root=str(run_root),
        stem="target",
        top_n=1,
        pita_threshold=-10.0,
        targetscan_threshold=1.0,
    )

    assert filtered.select(
        "pita_score", "targetscan_score", "off_target_filter"
    ).rows() == [
        (None, 0.5, 0),
        (None, None, -5),
    ]


def test_apply_off_target_filters_handles_header_only_targetscan(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))

    run_root = tmp_path / "run"
    _publish_test_off_target_evidence(
        run_root,
        stem="target",
        pita="RefSeq\tmicroRNA\tSites\tScore\nref\tRNA0\t1\t-2\n",
        targetscan="refseq\tsiRNA\ttargetscan_score\n",
    )

    result = pl.DataFrame({
        "pos": [1],
        "sense": ["GC"],
        "siRNA": ["AUGCUAGCUAGCUAGCUAG"],
        "efficacy": [0.8],
        "func_filter": [0],
    })

    filtered = oligoformer_app._apply_off_target_filters(
        result=result,
        run_root=str(run_root),
        stem="target",
        top_n=-1,
        pita_threshold=-10.0,
        targetscan_threshold=1.0,
    )

    assert filtered.select(
        "pita_score", "targetscan_score", "off_target_filter"
    ).rows() == [("-2", 0.0, 0)]


def test_apply_off_target_filters_fills_missing_targetscan_for_pita_hits(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))

    run_root = tmp_path / "run"
    _publish_test_off_target_evidence(
        run_root,
        stem="target",
        pita="RefSeq\tmicroRNA\tSites\tScore\nref\tRNA0\t1\t-2\n",
        targetscan="ref\tRNA1\t0.5\n",
    )

    result = pl.DataFrame({
        "pos": [1, 2],
        "sense": ["GC", "CG"],
        "siRNA": ["AUGCUAGCUAGCUAGCUAG", "UGCUAGCUAGCUAGCUAGC"],
        "efficacy": [0.8, 0.6],
        "func_filter": [0, 0],
    })

    filtered = oligoformer_app._apply_off_target_filters(
        result=result,
        run_root=str(run_root),
        stem="target",
        top_n=-1,
        pita_threshold=-10.0,
        targetscan_threshold=1.0,
    )

    assert filtered.select(
        "pita_score", "targetscan_score", "off_target_filter"
    ).rows() == [
        ("-2", 0.0, 0),
        (None, 0.5, 0),
    ]


def test_apply_off_target_filters_streams_summaries_and_preserves_pita_ties(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))

    def exploding_read_csv(*_args, **_kwargs):
        raise AssertionError("merged evidence must not be eagerly read")

    run_root = tmp_path / "run"
    _publish_test_off_target_evidence(
        run_root,
        stem="target",
        pita=(
            "RefSeq\tmicroRNA\tSites\tScore\nref1\tRNA0\t1\t-2.00\nref2\tRNA0\t1\t-2\n"
        ),
        targetscan="ref1\tRNA0\t0.5\nref2\tRNA0\t1.5\n",
    )
    monkeypatch.setattr(oligoformer_app.pl, "read_csv", exploding_read_csv)
    result = pl.DataFrame({
        "pos": [1],
        "sense": ["GC"],
        "siRNA": ["AUGCUAGCUAGCUAGCUAG"],
        "efficacy": [0.8],
        "func_filter": [0],
    })

    filtered = oligoformer_app._apply_off_target_filters(
        result=result,
        run_root=str(run_root),
        stem="target",
        top_n=-1,
        pita_threshold=-10.0,
        targetscan_threshold=1.0,
    )

    assert filtered.select(
        "pita_score", "targetscan_score", "off_target_filter"
    ).rows() == [("-2.00", 1.5, 1)]


def test_off_target_top_n_records_bind_ranked_names_to_ranked_sequences():
    result = pl.DataFrame({
        "pos": [1, 2, 3],
        "siRNA": ["AAAA", "CCCC", "GGGG"],
        "efficacy": [0.1, 0.9, 0.2],
    })

    records = oligoformer_app._off_target_sirna_records(result, top_n=2)

    assert [(record.name, record.sequence) for record in records] == [
        ("RNA1", "CCCC"),
        ("RNA2", "GGGG"),
    ]


def test_final_filter_does_not_let_negative_sentinel_cancel_failures():
    result = pl.DataFrame({
        "func_filter": [4, 0],
        "off_target_filter": [-5, -5],
        "toxicity_filter": [1, 0],
    })

    filtered = oligoformer_app._apply_final_filter(
        result=result,
        off_target=True,
        toxicity=True,
        functionality_filter=True,
    )

    assert filtered.get_column("filter").to_list() == [3, 1]


def test_targetscan_batch_merge_matches_split_sirna_merge(tmp_path: Path):
    rows = [
        ("refA", "RNA0", "6mer", "-0.5"),
        ("refA", "RNA0", "6mer", "-0.25"),
        ("refB", "RNA1", "6mer", "-0.75"),
    ]
    batch_context = tmp_path / "batch_context.txt"
    _write_targetscan_context_output(batch_context, rows)
    batch_output = tmp_path / "batch_targetscan.tab"
    oligoformer_app._merge_targetscan_context_outputs(
        context_outputs=[str(batch_context)],
        targetscan_path=batch_output,
        log_file=tmp_path / "batch_merge.log",
    )

    split_outputs = []
    for sirna in ("RNA0", "RNA1"):
        context_path = tmp_path / f"{sirna}_context.txt"
        _write_targetscan_context_output(
            context_path,
            [row for row in rows if row[1] == sirna],
        )
        output_path = tmp_path / f"{sirna}_targetscan.tab"
        oligoformer_app._merge_targetscan_context_outputs(
            context_outputs=[str(context_path)],
            targetscan_path=output_path,
            log_file=tmp_path / f"{sirna}_merge.log",
        )
        split_outputs.append(output_path)

    split_lines = [
        line
        for output_path in split_outputs
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    batch_lines = [
        line for line in batch_output.read_text(encoding="utf-8").splitlines() if line
    ]
    assert sorted(batch_lines) == sorted(split_lines)


def test_merge_targetscan_context_outputs_applies_site_type_thresholds(
    tmp_path: Path,
):
    context_path = tmp_path / "context.txt"
    _write_targetscan_context_output(
        context_path,
        [
            ("refA", "RNA0", "6mer", "-0.5"),
            ("refA", "RNA0", "7mer-1a", "-0.02"),
            ("refA", "RNA0", "7mer-1a", "-0.01"),
            ("refA", "RNA0", "7mer-m8", "-0.03"),
            ("refA", "RNA0", "7mer-m8", "-0.02"),
            ("refA", "RNA0", "8mer-1a", "-0.04"),
            ("refA", "RNA0", "8mer-1a", "-0.03"),
            ("refA", "RNA1", "8mer-1a", "0.04"),
            ("refB", "RNA2", "7mer-1a", "-0.02"),
        ],
    )
    output_path = tmp_path / "targetscan.tab"

    oligoformer_app._merge_targetscan_context_outputs(
        context_outputs=[str(context_path)],
        targetscan_path=output_path,
        log_file=tmp_path / "merge.log",
    )

    merged = pl.read_csv(
        output_path,
        separator="\t",
        has_header=False,
        new_columns=["refseq", "siRNA", "targetscan_score"],
        schema_overrides={"targetscan_score": pl.Float64},
    )
    assert merged.select("refseq", "siRNA").rows() == [
        ("refA", "RNA0"),
        ("refB", "RNA2"),
    ]
    assert merged.get_column("targetscan_score").to_list() == pytest.approx([
        0.59,
        0.02,
    ])


def test_merge_targetscan_context_outputs_warms_multi_file_directory(
    tmp_path: Path, monkeypatch
):
    first_context = tmp_path / "context_1.txt"
    second_context = tmp_path / "context_2.txt"
    _write_targetscan_context_output(first_context, [("refA", "RNA0", "6mer", "-0.5")])
    _write_targetscan_context_output(second_context, [("refB", "RNA1", "6mer", "-0.2")])
    warmup_calls = []
    monkeypatch.setattr(
        oligoformer_app,
        "warmup_directory",
        lambda path: warmup_calls.append(Path(path)),
    )

    oligoformer_app._merge_targetscan_context_outputs(
        context_outputs=[str(first_context), str(second_context)],
        targetscan_path=tmp_path / "targetscan.tab",
        log_file=tmp_path / "merge.log",
    )

    assert warmup_calls == [tmp_path]


def test_merge_targetscan_context_outputs_writes_header_only_empty_table(
    tmp_path: Path,
):
    output_path = tmp_path / "targetscan.tab"

    oligoformer_app._merge_targetscan_context_outputs(
        context_outputs=[],
        targetscan_path=output_path,
        log_file=tmp_path / "merge.log",
    )

    assert output_path.read_text(encoding="utf-8") == (
        "refseq\tsiRNA\ttargetscan_score\n"
    )
    assert oligoformer_app._read_targetscan_table(output_path).height == 0


def test_merge_targetscan_batch_outputs_sorts_upstream_order(tmp_path: Path):
    first = tmp_path / "first.tab"
    second = tmp_path / "second.tab"
    first.write_text("tx2\tRNA1\t0.2\ntx1\tRNA2\t0.3\n", encoding="utf-8")
    second.write_text("tx1\tRNA1\t0.1\n", encoding="utf-8")

    output = tmp_path / "targetscan.tab"
    oligoformer_app._merge_targetscan_batch_outputs(
        targetscan_paths=[str(first), str(second)],
        output_path=output,
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "tx1\tRNA1\t0.1",
        "tx1\tRNA2\t0.3",
        "tx2\tRNA1\t0.2",
    ]


def test_merge_targetscan_batch_outputs_preserves_header_only_empty_table(
    tmp_path: Path,
):
    first = tmp_path / "first.tab"
    second = tmp_path / "second.tab"
    first.write_text("refseq\tsiRNA\ttargetscan_score\n", encoding="utf-8")
    second.write_text("", encoding="utf-8")

    output = tmp_path / "targetscan.tab"
    oligoformer_app._merge_targetscan_batch_outputs(
        targetscan_paths=[str(first), str(second)],
        output_path=output,
    )

    assert output.read_text(encoding="utf-8") == "refseq\tsiRNA\ttargetscan_score\n"


def test_targetscan_context_shard_retries_interrupted_commands_atomically(
    tmp_path: Path, monkeypatch, capfd
):
    attempts: list[list[str]] = []
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    script_path = repo_dir / "off-target/targetscan/targetscan_70_context_scores.pl"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("script\n", encoding="utf-8")
    spec = oligoformer_app.TargetscanContextShardSpec(
        shard_index=7,
        common_dir=str(tmp_path / "common"),
        targets_path=str(tmp_path / "targets.txt"),
        output_path=str(tmp_path / "outputs" / "context.txt"),
        log_path=str(tmp_path / "logs" / "context.log"),
        rnaplfold_cache_dir="",
    )

    def fake_run_command(cmd, **_kwargs):
        attempts.append(cmd)
        tmp_output_path = Path(cmd[-1])
        tmp_output_path.write_text("partial\n", encoding="utf-8")
        if len(attempts) == 1:
            raise sp.CalledProcessError(-2, cmd)
        tmp_output_path.write_text("success\n", encoding="utf-8")

    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(oligoformer_app, "run_command", fake_run_command)
    assert (
        oligoformer_app._run_targetscan_context_shard(spec, attempts=2)
        == spec.output_path
    )
    output_path = Path(spec.output_path)
    assert output_path.read_text(encoding="utf-8") == "success\n"
    assert output_path.with_suffix(output_path.suffix + ".done").exists()
    assert len(attempts) == 2
    assert "Retrying OligoFormer TargetScan context shard 7" in capfd.readouterr().out

    output_path.write_text("corrupt\n", encoding="utf-8")
    assert not oligoformer_app._targetscan_context_shard_ready(spec)
    assert (
        oligoformer_app._run_targetscan_context_shard(spec, attempts=2)
        == spec.output_path
    )
    assert output_path.read_text(encoding="utf-8") == "success\n"
    assert len(attempts) == 3


def test_merge_pita_shards_sorts_like_upstream_score_table(tmp_path: Path):
    first = tmp_path / "first_pita.tab"
    second = tmp_path / "second_pita.tab"
    first.write_text(
        "RefSeq\tmicroRNA\tSites\tScore\ntx1\tRNA4\t5\t-20.13\ntx2\tRNA4\t3\t3.17\n",
        encoding="utf-8",
    )
    second.write_text(
        "RefSeq\tmicroRNA\tSites\tScore\ntx1\tRNA12\t9\t-19.42\ntx2\tRNA12\t7\t4.00\n",
        encoding="utf-8",
    )

    output = tmp_path / "pita.tab"
    oligoformer_app._merge_pita_shards(
        [
            oligoformer_app.OffTargetShardResult(
                index=0,
                pita_path=str(first),
            ),
            oligoformer_app.OffTargetShardResult(
                index=1,
                pita_path=str(second),
            ),
        ],
        output,
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "RefSeq\tmicroRNA\tSites\tScore",
        "tx1\tRNA4\t5\t-20.13",
        "tx1\tRNA12\t9\t-19.42",
        "tx2\tRNA4\t3\t3.17",
        "tx2\tRNA12\t7\t4.00",
    ]


def test_oligoformer_execution_defaults_use_distributed_cpu_nodes():
    execution = oligoformer_app.OligoformerExecutionConfig()

    assert execution == oligoformer_app.DEFAULT_EXECUTION_CONFIG
    assert oligoformer_app._bounded_node_count(100, execution.off_target_nodes) == 32
    assert (
        oligoformer_app._bounded_node_count(
            100,
            execution.targetscan_context_nodes,
        )
        == 100
    )
    assert (
        oligoformer_app._targetscan_ref_shard_size(
            100,
            prepare_nodes=execution.targetscan_prepare_nodes,
        )
        == 4
    )
    assert (
        oligoformer_app._targetscan_rnaplfold_node_count(
            1000,
            execution.targetscan_rnaplfold_nodes,
        )
        == 32
    )
    assert (
        oligoformer_app._targetscan_rnaplfold_worker_count(
            execution.targetscan_rnaplfold_workers
        )
        == 8
    )


def test_targetscan_batch_specs_split_transcript_aligned_refs(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    run_root = tmp_path / "run"
    output_dir = run_root / "outputs"
    utr_path = tmp_path / "UTR.fa"
    orf_path = tmp_path / "ORF.fa"
    utr_path.write_text(
        ">tx1\nAAAA\n>tx2\nCCCC\n>tx3\nGGGG\n",
        encoding="utf-8",
    )
    orf_path.write_text(
        ">tx1\nAUG\n>tx2\nCCC\n>tx3\nGGG\n>extra\nUUU\n",
        encoding="utf-8",
    )
    records = [
        oligoformer_app.OffTargetSirnaRecord("RNA0", "AUGCUAGCUAGCUAGCUAG"),
    ]

    specs = oligoformer_app._targetscan_batch_specs(
        run_root=str(run_root),
        output_dir=output_dir,
        stem="target",
        records=records,
        utr_path=str(utr_path),
        orf_path=str(orf_path),
        ref_shard_size=2,
    )

    assert [spec.shard_index for spec in specs] == [0, 1]
    assert [spec.ref_shard_size for spec in specs] == [2, 2]
    assert [Path(spec.utr_path).read_text(encoding="utf-8") for spec in specs] == [
        ">tx1\nAAAA\n>tx2\nCCCC\n",
        ">tx3\nGGGG\n",
    ]
    assert [Path(spec.orf_path).read_text(encoding="utf-8") for spec in specs] == [
        ">tx1\nAUG\n>tx2\nCCC\n",
        ">tx3\nGGG\n",
    ]
    assert all(spec.sirna_count == 1 for spec in specs)
    assert len({spec.sirna_path for spec in specs}) == 1
    assert Path(specs[0].sirna_path).read_text(encoding="utf-8") == (
        ">RNA0\nAUGCUAGCUAGCUAGCUAG\n"
    )
    assert [
        Path(spec.run_root)
        / "prepare"
        / "off_target"
        / "target"
        / "targetscan"
        / f"candidates_{spec.candidate_shard_size}"
        / f"{spec.candidate_shard_index:05d}"
        / f"size_{spec.ref_shard_size}"
        / f"{spec.shard_index:05d}"
        for spec in specs
    ] == [oligoformer_app._targetscan_batch_cache_dir(spec) for spec in specs]

    Path(specs[0].utr_path).write_text(">tx1\nTRUNCATED\n", encoding="utf-8")
    repaired_specs = oligoformer_app._targetscan_batch_specs(
        run_root=str(run_root),
        output_dir=output_dir,
        stem="target",
        records=records,
        utr_path=str(utr_path),
        orf_path=str(orf_path),
        ref_shard_size=2,
    )
    assert Path(repaired_specs[0].utr_path).read_text(encoding="utf-8") == (
        ">tx1\nAAAA\n>tx2\nCCCC\n"
    )

    updated_specs = oligoformer_app._targetscan_batch_specs(
        run_root=str(run_root),
        output_dir=output_dir,
        stem="target",
        records=records,
        utr_path=str(utr_path),
        orf_path=str(orf_path),
        ref_shard_size=3,
    )

    assert [spec.ref_shard_size for spec in updated_specs] == [3]
    assert Path(updated_specs[0].utr_path).read_text(encoding="utf-8") == (
        ">tx1\nAAAA\n>tx2\nCCCC\n>tx3\nGGGG\n"
    )
    assert oligoformer_app._targetscan_batch_cache_dir(updated_specs[0]).parts[-2:] == (
        "size_3",
        "00000",
    )


def test_targetscan_batch_specs_tile_candidate_and_reference_shards(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    execution = _execution_config(targetscan_candidate_shard_size=2)
    utr_path = tmp_path / "UTR.fa"
    orf_path = tmp_path / "ORF.fa"
    utr_path.write_text(">tx1\nAAAA\n>tx2\nCCCC\n", encoding="utf-8")
    orf_path.write_text(">tx1\nAAAA\n>tx2\nCCCC\n", encoding="utf-8")
    records = [
        oligoformer_app.OffTargetSirnaRecord(f"RNA{index}", "AUGCUAGCUAGCUAGCUAG")
        for index in range(3)
    ]

    specs = oligoformer_app._targetscan_batch_specs(
        run_root=str(tmp_path / "run"),
        output_dir=tmp_path / "outputs",
        stem="target",
        records=records,
        utr_path=str(utr_path),
        orf_path=str(orf_path),
        ref_shard_size=1,
        execution=execution,
    )

    assert len(specs) == 4
    assert {(spec.candidate_shard_index, spec.shard_index) for spec in specs} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert {(spec.candidate_shard_index, spec.sirna_count) for spec in specs} == {
        (0, 2),
        (1, 1),
    }
    candidate_inputs = {
        spec.candidate_shard_index: Path(spec.sirna_path).read_text(encoding="utf-8")
        for spec in specs
    }
    assert candidate_inputs == {
        0: ">RNA0\nAUGCUAGCUAGCUAGCUAG\n>RNA1\nAUGCUAGCUAGCUAGCUAG\n",
        1: ">RNA2\nAUGCUAGCUAGCUAGCUAG\n",
    }
    assert (
        len({oligoformer_app._targetscan_batch_cache_dir(spec) for spec in specs}) == 4
    )


def test_targetscan_multi_candidate_reference_tiles_merge_all_pairs(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    execution = _execution_config(targetscan_candidate_shard_size=2)
    utr_path = tmp_path / "UTR.fa"
    orf_path = tmp_path / "ORF.fa"
    utr_path.write_text(">tx1\nAAAA\n>tx2\nCCCC\n", encoding="utf-8")
    orf_path.write_text(">tx1\nAAAA\n>tx2\nCCCC\n", encoding="utf-8")
    records = [
        oligoformer_app.OffTargetSirnaRecord(f"RNA{index}", "AUGCUAGCUAGCUAGCUAG")
        for index in range(3)
    ]
    waves = list(
        oligoformer_app._targetscan_batch_spec_waves(
            run_root=str(tmp_path / "run"),
            output_dir=tmp_path / "outputs",
            stem="target",
            records=records,
            utr_path=str(utr_path),
            orf_path=str(orf_path),
            ref_shard_size=1,
            max_tiles_per_wave=1,
            execution=execution,
        )
    )
    tile_paths = []
    for wave in waves:
        assert len(wave) == 1
        spec = wave[0]
        reference_name = oligoformer_app._read_fasta_pairs(Path(spec.utr_path))[0][0]
        candidates = oligoformer_app._read_fasta_pairs(Path(spec.sirna_path))
        tile_path = oligoformer_app._targetscan_batch_cache_dir(spec) / "targetscan.tab"
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        tile_path.write_text(
            "".join(
                f"{reference_name}\t{candidate}\t0.5\n"
                for candidate, _sequence in candidates
            ),
            encoding="utf-8",
        )
        tile_paths.append(str(tile_path))

    output_path = tmp_path / "targetscan.tab"
    oligoformer_app._merge_targetscan_batch_outputs(
        targetscan_paths=tile_paths,
        output_path=output_path,
    )

    merged = oligoformer_app._read_targetscan_table(output_path)
    assert merged.height == 6
    assert set(merged.select("refseq", "siRNA").iter_rows()) == {
        (reference, candidate)
        for reference in ("tx1", "tx2")
        for candidate in ("RNA0", "RNA1", "RNA2")
    }
    assert [len(wave) for wave in waves] == [1, 1, 1, 1]


def test_targetscan_ref_shard_size_uses_prepare_node_fanout():
    assert (
        oligoformer_app._targetscan_ref_shard_size(
            28_352,
            prepare_nodes=300,
        )
        == 95
    )
    assert oligoformer_app._targetscan_ref_shard_size(28_352, 100) == 100


def test_targetscan_batch_reuses_seed_checkpoint_after_interruption(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    run_root = tmp_path / "run"
    output_dir = run_root / "outputs"
    sirna_path = tmp_path / "siRNA.fa"
    utr_path = tmp_path / "UTR.fa"
    orf_path = tmp_path / "ORF.fa"
    sirna_path.write_text(
        ">RNA0\nAUGCUAGCUAGCUAGCUAG\n",
        encoding="utf-8",
    )
    utr_path.write_text(">tx1\nAAAA\n", encoding="utf-8")
    orf_path.write_text(">tx1\nAUGC\n", encoding="utf-8")
    spec = oligoformer_app.TargetscanBatchSpec(
        run_root=str(run_root),
        output_dir=str(output_dir),
        stem="target",
        ref_shard_size=1,
        shard_index=0,
        sirna_path=str(sirna_path),
        sirna_count=1,
        utr_path=str(utr_path),
        orf_path=str(orf_path),
        rnaplfold_cache_dir="",
    )
    calls = []

    def fake_run_command(cmd, **kwargs):
        calls.append(list(cmd))
        cwd = Path(kwargs["cwd"])
        if cmd[:2] == ["perl", "targetscan_70_BL_bins.pl"]:
            return ["bins"]
        if cmd[0] == "perl":
            cwd.joinpath(cmd[-1]).write_text("seed output\n", encoding="utf-8")
        else:
            context_dir = Path(cmd[-2])
            context_dir.joinpath("common").mkdir(parents=True)
            for name in oligoformer_app.TARGETSCAN_CONTEXT_COMMON_NAMES:
                context_dir.joinpath("common", name).write_text(
                    "common\n",
                    encoding="utf-8",
                )
            context_dir.joinpath("outputs").mkdir()
            context_dir.joinpath("shards").mkdir()
            context_dir.joinpath("shards", "targets_00000").write_text(
                "targets\n",
                encoding="utf-8",
            )
        return []

    monkeypatch.setattr(oligoformer_app, "run_command", fake_run_command)
    first_batch_root = tmp_path / "first-batch"
    first_batch_root.joinpath("off-target", "targetscan").mkdir(parents=True)

    first = oligoformer_app._prepare_targetscan_batch_context_plan(
        spec,
        first_batch_root,
    )

    workdir = first_batch_root / "off-target" / "tmp"
    assert (
        workdir.joinpath("sirnas_for_context_scores.txt").read_text(encoding="utf-8")
        == "RNA0\t9606\tRNA0\tAUGCUAGCUAGCUAGCUAG\n"
    )
    assert workdir.joinpath("sirnas.txt").read_text(encoding="utf-8") == (
        "RNA0\tUGCUAGC\t9606\n"
    )
    assert len(first.context_shards) == 1
    assert [cmd[0] for cmd in calls] == ["perl", "perl", "bash"]
    assert sum("targetscan_70_BL_bins.pl" in cmd for cmd in calls) == 1
    assert volume.commit_count == 1

    cache_dir = oligoformer_app._targetscan_batch_cache_dir(spec)
    checkpoint_path = cache_dir / "targetscan_seed" / "targetscan_70_output.txt"
    assert checkpoint_path.read_text(encoding="utf-8") == "seed output\n"
    shutil.rmtree(cache_dir / "targetscan_context")
    calls.clear()
    second_batch_root = tmp_path / "second-batch"
    second_batch_root.joinpath("off-target", "targetscan").mkdir(parents=True)

    second = oligoformer_app._prepare_targetscan_batch_context_plan(
        spec,
        second_batch_root,
    )

    assert len(second.context_shards) == 1
    assert [cmd[0] for cmd in calls] == ["bash"]
    assert volume.commit_count == 1
    assert "Reusing OligoFormer TargetScan seed checkpoint" in capsys.readouterr().out

    shutil.rmtree(cache_dir / "targetscan_context")
    checkpoint_path.write_text("corrupt\n", encoding="utf-8")
    calls.clear()
    third_batch_root = tmp_path / "third-batch"
    third_batch_root.joinpath("off-target", "targetscan").mkdir(parents=True)

    third = oligoformer_app._prepare_targetscan_batch_context_plan(
        spec,
        third_batch_root,
    )

    assert len(third.context_shards) == 1
    assert [cmd[0] for cmd in calls] == ["perl", "bash"]
    assert checkpoint_path.read_text(encoding="utf-8") == "seed output\n"


def test_pita_prepare_rejects_missing_reference_files(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    monkeypatch.setattr(
        oligoformer_app,
        "_write_pita_stage0_script",
        lambda *_args: None,
    )
    commands = []
    monkeypatch.setattr(
        oligoformer_app,
        "run_command",
        lambda *args, **_kwargs: commands.append(args),
    )
    spec = oligoformer_app.OffTargetShardSpec(
        run_root=str(tmp_path / "run"),
        output_dir=str(tmp_path / "outputs"),
        stem="target",
        index=0,
        record_name="RNA0",
        record_sequence="AUGCUAGCUAGCUAGCUAG",
        utr_path=str(tmp_path / "missing_utr.fa"),
        orf_path=str(tmp_path / "missing_orf.fa"),
        row_shard_size=1000,
    )

    with pytest.raises(
        FileNotFoundError,
        match="PITA reference files are missing or empty",
    ):
        oligoformer_app._prepare_pita_target_discovery_plan(
            spec,
            tmp_path / "prepare",
        )

    assert commands == []


def test_pita_reference_stage0_is_reused_across_candidates(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    repo_dir.joinpath("off-target", "pita").mkdir(parents=True)
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(oligoformer_app, "_write_pita_stage0_script", lambda *_: None)
    execution = _execution_config(pita_prepare_utr_shard_size=2)
    utr_path = tmp_path / "UTR.fa"
    orf_path = tmp_path / "ORF.fa"
    utr_path.write_text(">utr\nAAAA\n", encoding="utf-8")
    orf_path.write_text(">orf\nAAAA\n", encoding="utf-8")
    spec = oligoformer_app.OffTargetShardSpec(
        run_root=str(tmp_path / "run"),
        output_dir=str(tmp_path / "outputs"),
        stem="target",
        index=0,
        record_name="RNA0",
        record_sequence="AUGCUAGCUAGCUAGCUAG",
        utr_path=str(utr_path),
        orf_path=str(orf_path),
        row_shard_size=1000,
    )
    commands = []

    def fake_run_command(cmd, **_kwargs):
        commands.append(cmd)
        reference_root = Path(cmd[1]).parents[2]
        reference_root.joinpath("reference_utr.stab").write_text(
            "utr1\tAAAA\nutr2\tCCCC\nutr3\tGGGG\n",
            encoding="utf-8",
        )
        output_dir = Path(cmd[cmd.index("-output") + 1])
        output_dir.joinpath("reference_ext_utr.stab").write_text(
            "utr\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(oligoformer_app, "run_command", fake_run_command)

    first = oligoformer_app._prepare_pita_reference_plan(
        spec,
        tmp_path / "prepare",
        execution,
    )
    second = oligoformer_app._prepare_pita_reference_plan(
        spec,
        tmp_path / "prepare2",
        execution,
    )

    assert len(commands) == 1
    assert first == second
    assert len(first.utr_shard_paths) == 2
    assert all(Path(path).is_file() for path in first.utr_shard_paths)

    Path(first.utr_shard_paths[0]).write_text("corrupt\n", encoding="utf-8")
    repaired = oligoformer_app._prepare_pita_reference_plan(
        spec,
        tmp_path / "prepare3",
        execution,
    )
    assert len(commands) == 2
    assert repaired == first
    assert Path(repaired.utr_shard_paths[0]).read_text(encoding="utf-8") == (
        "utr1\tAAAA\nutr2\tCCCC\n"
    )


def test_pita_prepare_reuses_completed_stage0_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    repo_dir.joinpath("off-target", "pita").mkdir(parents=True)
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(
        oligoformer_app,
        "_write_pita_stage0_script",
        lambda *_args: None,
    )
    execution = _execution_config(pita_prepare_utr_shard_size=2)
    utr_path = tmp_path / "human_UTR.txt"
    orf_path = tmp_path / "human_ORF.txt"
    utr_path.write_text(">utr\nAAAA\n", encoding="utf-8")
    orf_path.write_text(">orf\nAAAA\n", encoding="utf-8")
    spec = oligoformer_app.OffTargetShardSpec(
        run_root=str(tmp_path / "run"),
        output_dir=str(tmp_path / "final"),
        stem="target",
        index=3,
        record_name="RNA3",
        record_sequence="AUGCUAGCUAGCUAGCUAG",
        utr_path=str(utr_path),
        orf_path=str(orf_path),
        row_shard_size=1000,
    )
    commands = []

    def fake_run_command(cmd, **_kwargs):
        commands.append(cmd)
        shard_root = tmp_path / "first"
        shard_root.joinpath("target_shard_00003_utr.stab").write_text(
            "utr1\tAAAA\nutr2\tCCCC\nutr3\tGGGG\n",
            encoding="utf-8",
        )
        shard_root.joinpath("target_shard_00003_mir.stab").write_text(
            "RNA3\tUUUU\n",
            encoding="utf-8",
        )
        cache_dir = oligoformer_app._off_target_shard_cache_dir(spec)
        cache_dir.joinpath("target_shard_00003_ext_utr.stab").write_text(
            "utr\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(oligoformer_app, "run_command", fake_run_command)
    first_root = tmp_path / "first"
    first_root.joinpath("off-target", "pita").mkdir(parents=True)
    first = oligoformer_app._prepare_pita_target_discovery_plan(
        spec,
        first_root,
        execution=execution,
    )
    second = oligoformer_app._prepare_pita_target_discovery_plan(
        spec,
        tmp_path / "second",
        execution=execution,
    )

    assert len(commands) == 1
    assert len(first.utr_shards) == 2
    assert second.utr_shards == first.utr_shards
    assert all(
        Path(shard.input_path).is_relative_to(
            oligoformer_app._off_target_shard_cache_dir(spec)
        )
        for shard in second.utr_shards
    )
    stage0_marker = orjson.loads(
        oligoformer_app
        ._off_target_shard_cache_dir(spec)
        .joinpath("pita_stage0.done")
        .read_bytes()
    )
    assert stage0_marker["kind"] == "pita-candidate-stage0"
    assert (
        len([
            name for name in stage0_marker["artifacts"] if name.startswith("utr_shard/")
        ])
        == 2
    )


def test_pita_prepare_wrapper_removes_per_sirna_worktree(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    repo_dir.joinpath("off-target", "pita").mkdir(parents=True)
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    spec = oligoformer_app.OffTargetShardSpec(
        run_root="run",
        output_dir="output",
        stem="target",
        index=4,
        record_name="RNA4",
        record_sequence="AUGCUAGCUAGCUAGCUAG",
        utr_path="utr.fa",
        orf_path="orf.fa",
        row_shard_size=1000,
    )
    persistent_path = tmp_path / "outputs-volume" / "persistent.utr.stab"
    persistent_path.parent.mkdir(parents=True)
    persistent_path.write_text("utr\n", encoding="utf-8")

    def fake_prepare(inner_spec, shard_root, reference=None, execution=None):
        assert inner_spec is spec
        assert reference is None
        assert execution == oligoformer_app.DEFAULT_EXECUTION_CONFIG
        shard_root.joinpath("large-transient.stab").write_text(
            "transient\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            utr_shards=(SimpleNamespace(input_path=str(persistent_path)),)
        )

    monkeypatch.setattr(
        oligoformer_app,
        "_prepare_pita_target_discovery_plan",
        fake_prepare,
    )
    prepare_root = tmp_path / "prepare"

    plan = oligoformer_app._prepare_pita_target_discovery_plan_for_spec(
        spec=spec,
        prepare_root=prepare_root,
    )

    assert Path(plan.utr_shards[0].input_path).exists()
    assert not prepare_root.joinpath("target_shard_00004").exists()


def test_pita_prepare_streams_utr_stab_into_shards(tmp_path: Path, monkeypatch):
    utr_stab_path = tmp_path / "utr.stab"
    utr_stab_path.write_text(
        "utr1\tAAAA\nutr2\tCCCC\nutr3\tGGGG\n",
        encoding="utf-8",
    )
    mir_stab_path = tmp_path / "mir.stab"
    mir_stab_path.write_text("RNA0\tUUUU\n", encoding="utf-8")
    original_read_text = Path.read_text

    def reject_whole_file_read(path: Path, *args, **kwargs):
        if path == utr_stab_path:
            raise AssertionError("UTR stabilization data must be streamed")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_whole_file_read)

    specs = oligoformer_app._pita_prepare_utr_shard_specs(
        utr_stab_path=utr_stab_path,
        mir_stab_path=mir_stab_path,
        shard_dir=tmp_path / "shards",
        logs_dir=tmp_path / "logs",
        shard_size=2,
    )

    assert [Path(spec.input_path).read_text(encoding="utf-8") for spec in specs] == [
        "utr1\tAAAA\nutr2\tCCCC\n",
        "utr3\tGGGG\n",
    ]


def test_finalize_pita_discovery_warms_and_writes_row_inputs_in_one_pass(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    warmup_calls = []
    monkeypatch.setattr(
        oligoformer_app,
        "warmup_directory",
        lambda path, file_pattern=".": warmup_calls.append((Path(path), file_pattern)),
    )
    spec = oligoformer_app.OffTargetShardSpec(
        run_root=str(tmp_path / "run"),
        output_dir=str(tmp_path / "outputs"),
        stem="target",
        index=0,
        record_name="RNA0",
        record_sequence="AUGCUAGCUAGCUAGCUAG",
        utr_path=str(tmp_path / "utr.fa"),
        orf_path=str(tmp_path / "orf.fa"),
        row_shard_size=2,
    )
    cache_dir = oligoformer_app._off_target_shard_cache_dir(spec)
    shard_dir = cache_dir / "pita_prepare_utr_shards"
    shard_dir.mkdir(parents=True)
    cache_dir.joinpath("target_shard_00000_ext_utr.stab").write_text(
        "utr\n",
        encoding="utf-8",
    )
    utr_shards = []
    for shard_index, rows in enumerate(("RNA0\tutr1\nRNA0\tutr2", "RNA0\tutr3\n")):
        output_path = shard_dir / f"{shard_index:05d}.potential.tsv"
        output_path.write_text(rows, encoding="utf-8")
        oligoformer_app._publish_artifact_marker(
            output_path.with_suffix(output_path.suffix + ".done"),
            kind="pita-target-discovery",
            artifacts={"potential_targets": output_path},
        )
        utr_shards.append(
            oligoformer_app.PitaPrepareUtrShardSpec(
                shard_index=shard_index,
                input_path=str(shard_dir / f"{shard_index:05d}.utr.stab"),
                mir_stab_path=str(cache_dir / "mir.stab"),
                output_path=str(output_path),
                log_path=str(tmp_path / "logs" / f"{shard_index:05d}.log"),
            )
        )
    plan = oligoformer_app.PitaPreparePlan(
        spec=spec,
        utr_shards=tuple(utr_shards),
        row_count=None,
    )

    prepared = oligoformer_app._finalize_pita_target_discovery_plan(plan)

    assert warmup_calls == [(shard_dir, r"\.potential\.tsv$")]
    assert (
        cache_dir.joinpath("potential_targets.tsv").read_text(encoding="utf-8")
        == "RNA0\tutr1\nRNA0\tutr2\nRNA0\tutr3\n"
    )
    prepare_marker = orjson.loads(cache_dir.joinpath("pita_prepare.done").read_bytes())
    assert prepare_marker["kind"] == "pita-target-discovery-merge"
    assert prepare_marker["artifacts"]["potential_targets"]["line_count"] == 3
    assert [(row.start_row, row.end_row) for row in prepared.row_shards] == [
        (0, 2),
        (2, 3),
    ]
    assert [
        Path(row.input_path).read_text(encoding="utf-8") for row in prepared.row_shards
    ] == ["RNA0\tutr1\nRNA0\tutr2\n", "RNA0\tutr3\n"]


def test_worker_aware_batches_fill_containers_before_spawning_more():
    assert [
        len(batch)
        for batch in oligoformer_app._batch_items_for_local_workers(
            list(range(45)),
            max_nodes=32,
            local_workers=32,
        )
    ] == [32, 13]

    large_batches = oligoformer_app._batch_items_for_local_workers(
        list(range(1317)),
        max_nodes=32,
        local_workers=32,
    )
    assert len(large_batches) == 32
    assert max(len(batch) for batch in large_batches) == 42


def test_pita_row_shard_retries_interrupted_commands_atomically(
    tmp_path: Path, monkeypatch, capfd
):
    attempts: list[list[str]] = []
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    repo_dir.joinpath("off-target", "pita", "lib").mkdir(parents=True)
    input_path = tmp_path / "row.potential.tsv"
    ext_utr_path = tmp_path / "ext_utr.stab"
    output_path = tmp_path / "rows" / "00000.scored.tsv"
    log_path = tmp_path / "logs" / "00000.log"
    input_path.write_text("potential\n", encoding="utf-8")
    ext_utr_path.write_text("stab\n", encoding="utf-8")
    spec = oligoformer_app.PitaRowShardSpec(
        run_root=str(tmp_path / "run"),
        stem="target",
        sirna_index=0,
        record_name="RNA0",
        shard_index=0,
        start_row=0,
        end_row=1,
        potential_targets_path=str(tmp_path / "potential.tsv"),
        input_path=str(input_path),
        ext_utr_path=str(ext_utr_path),
        output_path=str(output_path),
        log_path=str(log_path),
    )

    def fake_run_command(cmd, **_kwargs):
        attempts.append(cmd)
        tmp_output = Path(shlex.split(cmd[-1].rsplit("> ", 1)[1])[0])
        tmp_output.write_text("partial\n", encoding="utf-8")
        if len(attempts) == 1:
            raise sp.CalledProcessError(-2, cmd)
        tmp_output.write_text("success\n", encoding="utf-8")

    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(oligoformer_app, "run_command", fake_run_command)

    assert oligoformer_app._run_pita_row_shard(spec, attempts=2) == str(output_path)
    assert output_path.read_text(encoding="utf-8") == "success\n"
    assert output_path.with_suffix(output_path.suffix + ".done").exists()
    assert list(output_path.parent.glob("*.tmp.*")) == []
    assert len(attempts) == 2
    assert "Retrying OligoFormer PITA row shard RNA0:0" in capfd.readouterr().out

    output_path.write_text("corrupt\n", encoding="utf-8")
    assert oligoformer_app._run_pita_row_shard(spec, attempts=2) == str(output_path)
    assert output_path.read_text(encoding="utf-8") == "success\n"
    assert len(attempts) == 3


def test_package_output_tables_bundles_only_final_tables(tmp_path: Path, monkeypatch):
    captured = {}
    output_dir = tmp_path / "outputs"
    output_dir.joinpath("logs", "off_target").mkdir(parents=True)
    output_dir.joinpath("logs", "off_target", "row.log").write_text(
        "debug\n", encoding="utf-8"
    )
    for path in oligoformer_app._output_bundle_paths(("target",)):
        output_dir.joinpath(path).write_text("result\n", encoding="utf-8")

    def fake_package_outputs(root, *, paths_to_bundle):
        captured["root"] = Path(root)
        captured["paths"] = [str(path) for path in paths_to_bundle]
        return b"archive"

    monkeypatch.setattr(oligoformer_app, "package_outputs", fake_package_outputs)

    assert oligoformer_app._package_output_tables(output_dir, ("target",)) == b"archive"
    assert captured == {
        "root": output_dir,
        "paths": [
            "target.txt",
            "target_ranked.txt",
            "target_ranked_filtered.txt",
        ],
    }


def test_package_output_tables_requires_all_final_tables(tmp_path: Path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    output_dir.joinpath("target.txt").write_text("result\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="target_ranked.txt"):
        oligoformer_app._package_output_tables(output_dir, ("target",))


def test_read_efficacy_output_preserves_legacy_float_format(tmp_path: Path):
    efficacy_path = tmp_path / "target.txt"
    efficacy_path.write_text(
        "pos\tsense\tsiRNA\tefficacy\tfunc_filter\tfilter\n"
        "14\tA\tU\t0.9193557665348053\t4\t4\n"
        "18\tC\tG\t0.9641572347879409\t4\t4\n",
        encoding="utf-8",
    )

    result = oligoformer_app._read_efficacy_output(efficacy_path)
    oligoformer_app._write_final_outputs(result, tmp_path / "outputs", "target")

    assert (tmp_path / "outputs" / "target.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "pos\tsense\tsiRNA\tefficacy\tfunc_filter\tfilter",
        "14\tA\tU\t0.9193557665348052\t4\t4",
        "18\tC\tG\t0.9641572347879408\t4\t4",
    ]


def test_download_oligoformer_models_writes_to_model_volume(
    tmp_path: Path, monkeypatch
) -> None:
    source = Path(oligoformer_app.__file__).read_text(encoding="utf-8")
    commit_events = []

    class TrackingVolume(FakeVolume):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def commit(self):
            super().commit()
            commit_events.append(self.name)

    model_volume = TrackingVolume("model")
    output_volume = TrackingVolume("output")
    info = oligoformer_app.AppInfo(
        repo_rnafm_dir=tmp_path / "repo" / "RNA-FM",
        model_rnafm_dir=tmp_path / "models" / "RNA-FM",
        repo_ref_dir=tmp_path / "repo" / "off-target" / "ref",
        model_ref_dir=tmp_path / "models" / "off-target" / "ref",
    )
    calls = []
    lock_calls = []

    def fake_download_files(urls, *, force, num_retries, progress_bar_desc):
        calls.append({
            "urls": urls,
            "force": force,
            "num_retries": num_retries,
            "progress_bar_desc": progress_bar_desc,
        })
        for archive_path in urls.values():
            Path(archive_path).parent.mkdir(parents=True, exist_ok=True)
            if str(archive_path).endswith(".tar.gz"):
                with tarfile.open(archive_path, "w:gz") as archive:
                    payload = tmp_path / "model.pt"
                    payload.write_text("weights", encoding="utf-8")
                    archive.add(payload, arcname="RNA-FM/redevelop/model.pt")
            else:
                archive_name = Path(archive_path).name
                with zipfile.ZipFile(archive_path, "w") as archive:
                    if "UTR" in archive_name:
                        archive.writestr(
                            "UTR_Sequences.txt",
                            "Gene ID\tTranscript ID\tGene Symbol\tSpecies ID\t"
                            "UTR Sequence\n"
                            "ENST000001\tGENE1\tGENE1\t9606\tAUGC\n",
                        )

                    else:
                        archive.writestr(
                            "ORF_Sequences.txt",
                            "Transcript ID\tSpecies ID\tORF Sequence\n"
                            "ENST000001\t9606\tAUGC\n",
                        )

    def fake_cache_build_lock(stage, identity, *, rebuild=False):
        lock_calls.append((stage, identity, rebuild))
        return nullcontext(True)

    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    monkeypatch.setattr(
        oligoformer_app,
        "CONF",
        _fake_conf(tmp_path, output_volume),
    )
    monkeypatch.setattr(oligoformer_app, "MODEL_VOLUME", model_volume)
    monkeypatch.setattr(oligoformer_app, "download_files", fake_download_files)
    monkeypatch.setattr(
        oligoformer_app,
        "_cache_build_lock",
        fake_cache_build_lock,
    )

    oligoformer_app.download_oligoformer_models.get_raw_f()(force=True)

    assert list(calls[0]["urls"]) == [info.rnafm_archive_url]
    assert calls[0]["force"] is True
    assert calls[0]["num_retries"] == 3
    assert calls[0]["progress_bar_desc"] == "OligoFormer model downloads"
    assert calls[1]["urls"] == info.human_ref_downloads
    assert calls[1]["progress_bar_desc"] == "OligoFormer human ref downloads"
    assert (
        info.model_rnafm_redevelop_dir.joinpath("model.pt").read_text(encoding="utf-8")
        == "weights"
    )
    for ref_path in info.model_human_ref_paths:
        assert ref_path.read_text(encoding="utf-8") == ">ENST000001\nAUGC\n"
    identity = orjson.loads(info.targetscan_ref_identity_path.read_bytes())
    assert identity["content_sha256"] == {
        path.name: oligoformer_app._hash_path(path)
        for path in info.model_human_ref_paths
    }
    model_identity = orjson.loads(info.rnafm_identity_path.read_bytes())
    assert model_identity == orjson.loads(info.model_rnafm_identity_path.read_bytes())
    assert model_identity["content_sha256"] == oligoformer_app._hash_directory(
        info.model_rnafm_redevelop_dir
    )
    assert not info.targetscan_rnaplfold_marker_path.exists()
    assert model_volume.commit_count == 1
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1
    assert commit_events == ["model", "output"]
    assert lock_calls == [("targetscan-reference-state", "global", True)]
    assert "min(Args.top_n, RESULT_ranked.shape[0])" in source
    assert "--biomodals_stage" in source


def test_download_models_refreshes_stale_output_reference_identity(
    tmp_path: Path, monkeypatch
) -> None:
    model_volume = FakeVolume()
    output_volume = FakeVolume()
    info = oligoformer_app.AppInfo(
        repo_rnafm_dir=tmp_path / "repo" / "RNA-FM",
        model_rnafm_dir=tmp_path / "models" / "RNA-FM",
        repo_ref_dir=tmp_path / "repo" / "off-target" / "ref",
        model_ref_dir=tmp_path / "models" / "off-target" / "ref",
    )
    info.model_rnafm_redevelop_dir.mkdir(parents=True)
    info.model_rnafm_redevelop_dir.joinpath("model.pt").write_text(
        "weights", encoding="utf-8"
    )
    info.model_ref_dir.mkdir(parents=True)
    for path in info.model_human_ref_paths:
        path.write_text(f">{path.stem}\nAUGC\n", encoding="utf-8")
    model_identity = info.targetscan_ref_metadata | {
        "row_counts": {path.name: 1 for path in info.model_human_ref_paths},
        "content_sha256": {
            path.name: oligoformer_app._hash_path(path)
            for path in info.model_human_ref_paths
        },
    }
    info.targetscan_ref_marker_path.write_bytes(orjson.dumps(model_identity))

    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    monkeypatch.setattr(
        oligoformer_app,
        "CONF",
        _fake_conf(tmp_path, output_volume),
    )
    monkeypatch.setattr(oligoformer_app, "MODEL_VOLUME", model_volume)
    info.targetscan_ref_identity_path.parent.mkdir(parents=True)
    info.targetscan_ref_identity_path.write_bytes(
        orjson.dumps(
            model_identity
            | {
                "content_sha256": {
                    "human_UTR.txt": "stale",
                    "human_ORF.txt": "stale",
                }
            }
        )
    )

    def exploding_download(*_args, **_kwargs):
        raise AssertionError("valid model-volume assets must not be redownloaded")

    monkeypatch.setattr(oligoformer_app, "download_files", exploding_download)
    monkeypatch.setattr(
        oligoformer_app,
        "_cache_build_lock",
        lambda *_args, **_kwargs: nullcontext(True),
    )

    oligoformer_app.download_oligoformer_models.get_raw_f()(force=False)

    assert (
        orjson.loads(info.targetscan_ref_identity_path.read_bytes()) == model_identity
    )
    assert output_volume.reload_count == 2
    assert output_volume.commit_count == 1
    assert model_volume.commit_count == 1

    def exploding_lock(*_args, **_kwargs):
        raise AssertionError("ready assets must bypass the coordination ledger")

    monkeypatch.setattr(oligoformer_app, "_cache_build_lock", exploding_lock)
    oligoformer_app.download_oligoformer_models.get_raw_f()(force=False)

    assert output_volume.reload_count == 3
    assert output_volume.commit_count == 1
    assert model_volume.commit_count == 1


def test_submit_oligoformer_requires_off_target_references(tmp_path: Path) -> None:
    input_fasta = tmp_path / "target.fa"
    input_fasta.write_text(">m\nAUGCUAGCUAGCUAGCUAGC\n", encoding="utf-8")
    raw_f = oligoformer_app.submit_oligoformer_task.info.raw_f
    assert raw_f is not None

    with pytest.raises(ValueError, match="--utr-file and --orf-file"):
        raw_f(
            mrna_fasta=str(input_fasta),
            out_dir=str(tmp_path),
            off_target=True,
        )
