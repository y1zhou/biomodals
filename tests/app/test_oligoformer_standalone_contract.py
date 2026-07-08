"""Tests for standalone OligoFormer app behavior."""

# ruff: noqa: D101,D102,D103,D107

import shlex
import subprocess as sp
import tarfile
import zipfile
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
        version=oligoformer_app.CONF.version,
        repo_commit_hash=oligoformer_app.CONF.repo_commit_hash,
        output_volume_mountpoint=str(tmp_path / "outputs-volume"),
        git_clone_dir=repo_dir or tmp_path / "repo",
        output_volume=volume,
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
    assert input_dir.joinpath("mrna.fa").read_bytes().startswith(b">target one")
    assert input_dir.joinpath("sirna.fa").read_bytes().startswith(b">s")
    assert input_dir.joinpath("utr.txt").read_bytes() == b">utr\nAUGC\n"
    assert input_dir.joinpath("orf.txt").read_bytes() == b">orf\nAUGC\n"
    assert result.efficacy_ready is False
    assert result.final_ready is False
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_final_ready_requires_current_postprocess_marker_salt(tmp_path: Path):
    layout = oligoformer_app.AppRunLayout.from_run_root(tmp_path / "run")
    layout.outputs_dir.mkdir(parents=True)
    layout.markers_dir.mkdir(parents=True)
    for suffix in ("", "_ranked", "_ranked_filtered"):
        layout.outputs_dir.joinpath(f"target{suffix}.txt").write_text(
            "ok\n", encoding="utf-8"
        )

    oligoformer_app._marker_path(layout, "final.done").write_bytes(
        orjson.dumps({
            "cache_key": "abc123",
            "output_stems": ["target"],
        })
    )
    assert not oligoformer_app._build_plan(
        "abc123", ("target",), layout.run_root
    ).final_ready

    stale_plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        run_root=str(layout.run_root),
        efficacy_dir=str(layout.prep_dir / "efficacy"),
        output_dir=str(layout.outputs_dir),
        output_stems=("target",),
        efficacy_ready=False,
        final_ready=False,
    )
    oligoformer_app._write_cache_marker(
        layout,
        "final.done",
        stale_plan,
        extra_metadata={
            "postprocess_cache_salt": oligoformer_app.APP_INFO.postprocess_cache_salt
        },
    )

    assert oligoformer_app._build_plan(
        "abc123", ("target",), layout.run_root
    ).final_ready


def test_run_oligoformer_efficacy_builds_gpu_stage_command(tmp_path: Path, monkeypatch):
    captured = {}
    volume = FakeVolume()
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
    efficacy_dir = tmp_path / "run" / "prepare" / "efficacy"
    input_dir.mkdir(parents=True)
    input_dir.joinpath("mrna.fa").write_text(">target\nAUGCUAGCUAGCUAGCUAGC\n")
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        run_root=str(tmp_path / "run"),
        efficacy_dir=str(efficacy_dir),
        output_dir=str(tmp_path / "run" / "outputs"),
        output_stems=("target",),
        efficacy_ready=False,
        final_ready=False,
    )

    def fake_run_command(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "_ranked", "_ranked_filtered"):
            out_dir.joinpath(f"target{suffix}.txt").write_text("ok\n")

    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    monkeypatch.setattr(oligoformer_app, "run_command", fake_run_command)

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
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_run_oligoformer_postprocess_packages_cpu_outputs(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}
    targetscan_plan_calls: list[oligoformer_app.TargetscanBatchSpec] = []
    targetscan_batch_calls: list[
        tuple[list[oligoformer_app.TargetscanContextShardSpec], int]
    ] = []
    pita_plan_calls: list[oligoformer_app.OffTargetShardSpec] = []
    pita_prepare_batch_calls: list[
        tuple[list[oligoformer_app.PitaPrepareUtrShardSpec], int]
    ] = []
    row_batch_calls: list[tuple[list[oligoformer_app.PitaRowShardSpec], int]] = []
    finalize_calls: list[oligoformer_app.PreparedOffTargetShard] = []
    finalize_batch_calls: list[
        tuple[list[oligoformer_app.PreparedOffTargetShard], int]
    ] = []
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    repo_dir.joinpath("toxicity").mkdir(parents=True)
    repo_dir.joinpath("toxicity", "cell_viability.txt").write_text(
        "Seed\tcell_viability\nUGCUAG\t60\nGCUAGC\t60\n",
        encoding="utf-8",
    )
    repo_dir.joinpath("off-target", "pita").mkdir(parents=True)
    repo_dir.joinpath("off-target", "targetscan").mkdir(parents=True)
    layout = oligoformer_app.AppRunLayout.from_run_root(tmp_path / "run")
    layout.prep_dir.joinpath("efficacy").mkdir(parents=True)
    layout.inputs_dir.mkdir(parents=True)
    layout.inputs_dir.joinpath("utr.txt").write_text(">utr\nAUGC\n", encoding="utf-8")
    layout.inputs_dir.joinpath("orf.txt").write_text(">orf\nAUGC\n", encoding="utf-8")
    layout.prep_dir.joinpath("efficacy", "target.txt").write_text(
        "pos\tsense\tsiRNA\tefficacy\tfunc_filter\tfilter\n"
        "1\tGC\tAUGCUAGCUAGCUAGCUAG\t0.8\t0\t0\n"
        "2\tCG\tUGCUAGCUAGCUAGCUAGC\t0.6\t0\t0\n",
        encoding="utf-8",
    )
    for suffix in ("_ranked", "_ranked_filtered"):
        layout.prep_dir.joinpath("efficacy", f"target{suffix}.txt").write_text(
            "placeholder\n", encoding="utf-8"
        )
    layout.markers_dir.mkdir(parents=True)
    oligoformer_app._marker_path(layout, "efficacy.done").write_text(
        "{}", encoding="utf-8"
    )
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        run_root=str(layout.run_root),
        efficacy_dir=str(layout.prep_dir / "efficacy"),
        output_dir=str(layout.outputs_dir),
        output_stems=("target",),
        efficacy_ready=True,
        final_ready=False,
    )

    def fake_run_targetscan_prepare_batches(
        specs: list[oligoformer_app.TargetscanBatchSpec],
    ):
        targetscan_plan_calls.extend(specs)
        spec = specs[0]
        cache_dir = oligoformer_app._targetscan_batch_cache_dir(spec)
        context_dir = cache_dir / "targetscan_context"
        context_spec = oligoformer_app.TargetscanContextShardSpec(
            shard_index=0,
            common_dir=str(context_dir / "common"),
            targets_path=str(context_dir / "shards" / "targets_00000"),
            output_path=str(context_dir / "outputs" / "context_00000.txt"),
            log_path=str(
                layout.outputs_dir
                / "logs"
                / "off_target"
                / spec.stem
                / "targetscan"
                / f"{spec.shard_index:05d}"
                / "targetscan_context"
                / "00000.log"
            ),
            rnaplfold_cache_dir="",
        )
        return [
            oligoformer_app.PreparedTargetscanBatch(
                targetscan_path=str(cache_dir / "targetscan.tab"),
                logs_dir=str(
                    layout.outputs_dir
                    / "logs"
                    / "off_target"
                    / spec.stem
                    / "targetscan"
                    / f"{spec.shard_index:05d}"
                ),
                context_shards=(context_spec,),
                needs_merge=True,
            ),
        ]

    class FakeTargetscanContextBatch:
        def remote(
            self,
            specs: list[oligoformer_app.TargetscanContextShardSpec],
            local_workers: int,
        ):
            targetscan_batch_calls.append((list(specs), local_workers))
            outputs = []
            for spec in specs:
                output_path = Path(spec.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("context\n", encoding="utf-8")
                outputs.append(spec.output_path)
            return outputs

    def fake_merge_targetscan_context_outputs(
        *,
        context_outputs: list[str],
        targetscan_path: Path,
        log_file: Path,
    ) -> None:
        assert len(context_outputs) == 1
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("merge log\n", encoding="utf-8")
        targetscan_path.parent.mkdir(parents=True, exist_ok=True)
        targetscan_path.write_text(
            "ref\tRNA0\t2\nref\tRNA1\t0.5\n",
            encoding="utf-8",
        )

    def fake_prepare_pita_target_discovery_plan(
        spec: oligoformer_app.OffTargetShardSpec,
        _shard_root: Path,
    ):
        pita_plan_calls.append(spec)
        cache_dir = oligoformer_app._off_target_shard_cache_dir(spec)
        shard_dir = cache_dir / "pita_prepare_utr_shards"
        input_path = shard_dir / "00000_000000000000_000000000001.utr.stab"
        output_path = shard_dir / "00000_000000000000_000000000001.potential.tsv"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text("utr\n", encoding="utf-8")
        cache_dir.joinpath(
            f"{spec.stem}_shard_{spec.index:05d}_ext_utr.stab"
        ).write_text(
            "utr\n",
            encoding="utf-8",
        )
        return oligoformer_app.PitaPreparePlan(
            spec=spec,
            utr_shards=(
                oligoformer_app.PitaPrepareUtrShardSpec(
                    shard_index=0,
                    input_path=str(input_path),
                    mir_stab_path=str(cache_dir / f"{spec.record_name}_mir.stab"),
                    output_path=str(output_path),
                    log_path=str(
                        layout.outputs_dir
                        / "logs"
                        / "off_target"
                        / spec.stem
                        / f"{spec.index:05d}_{spec.record_name}"
                        / "pita_prepare_utr_shards"
                        / "00000.log"
                    ),
                ),
            ),
            row_count=None,
        )

    class FakePitaPrepareUtrShardBatch:
        def remote(
            self,
            specs: list[oligoformer_app.PitaPrepareUtrShardSpec],
            local_workers: int,
        ):
            pita_prepare_batch_calls.append((list(specs), local_workers))
            outputs = []
            for spec in specs:
                output_path = Path(spec.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                record_name = "RNA0" if "00000_RNA0" in str(output_path) else "RNA1"
                output_path.write_text(
                    f"{record_name}\tpotential\n",
                    encoding="utf-8",
                )
                output_path.with_suffix(output_path.suffix + ".done").write_text(
                    "done",
                    encoding="utf-8",
                )
                outputs.append(spec.output_path)
            return outputs

    class FakeRowBatch:
        def remote(
            self,
            row_shards: list[oligoformer_app.PitaRowShardSpec],
            local_workers: int,
        ):
            row_batch_calls.append((list(row_shards), local_workers))
            outputs = []
            for row in row_shards:
                output_path = Path(row.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("scored\n", encoding="utf-8")
                output_path.with_suffix(output_path.suffix + ".done").write_text(
                    "done", encoding="utf-8"
                )
                log_path = Path(row.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("row log\n", encoding="utf-8")
                outputs.append(row.output_path)
            return outputs

    class FakeFinalizePitaShardBatch:
        def remote(
            self,
            prepared_shards: list[oligoformer_app.PreparedOffTargetShard],
            local_workers: int,
        ):
            finalize_batch_calls.append((list(prepared_shards), local_workers))
            results = []
            for prepared in prepared_shards:
                finalize_calls.append(prepared)
                pita_score = "-11" if prepared.record_name == "RNA0" else "-2"
                Path(prepared.pita_path).write_text(
                    f"microRNA\tScore\n{prepared.record_name}\t{pita_score}\n",
                    encoding="utf-8",
                )
                results.append(
                    oligoformer_app.OffTargetShardResult(
                        index=prepared.index,
                        pita_path=prepared.pita_path,
                    )
                )
            return results

    def fake_package_outputs(root, *, paths_to_bundle=None):
        captured["package_root"] = Path(root)
        captured["package_paths"] = [str(path) for path in paths_to_bundle or ()]
        return b"archive"

    class FakeCall:
        def __init__(self, func, args, kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs
            self.cancelled = False

        def get(self):
            return self.func(*self.args, **self.kwargs)

        def cancel(self, terminate_containers=False):
            self.cancelled = terminate_containers

    class FakeBranch:
        def __init__(self, func):
            self.func = func
            self.spawn_calls = []

        def spawn(self, *args, **kwargs):
            self.spawn_calls.append((args, kwargs))
            return FakeCall(self.func, args, kwargs)

    gather_calls = []
    targetscan_branch = FakeBranch(
        oligoformer_app.run_oligoformer_targetscan_branch.get_raw_f()
    )
    pita_branch = FakeBranch(oligoformer_app.run_oligoformer_pita_branch.get_raw_f())

    def fake_gather(*calls):
        gather_calls.append(calls)
        return tuple(call.get() for call in calls)

    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(
        oligoformer_app,
        "run_oligoformer_targetscan_branch",
        targetscan_branch,
    )
    monkeypatch.setattr(oligoformer_app, "run_oligoformer_pita_branch", pita_branch)
    monkeypatch.setattr(
        oligoformer_app.modal.FunctionCall,
        "gather",
        staticmethod(fake_gather),
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_run_targetscan_prepare_batches",
        fake_run_targetscan_prepare_batches,
    )
    monkeypatch.setattr(
        oligoformer_app,
        "run_oligoformer_targetscan_context_shard_batch",
        FakeTargetscanContextBatch(),
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_merge_targetscan_context_outputs",
        fake_merge_targetscan_context_outputs,
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_prepare_pita_target_discovery_plan",
        fake_prepare_pita_target_discovery_plan,
    )
    monkeypatch.setattr(
        oligoformer_app,
        "run_oligoformer_pita_prepare_utr_shard_batch",
        FakePitaPrepareUtrShardBatch(),
    )
    monkeypatch.setattr(
        oligoformer_app,
        "run_oligoformer_pita_row_shard_batch",
        FakeRowBatch(),
    )
    monkeypatch.setattr(
        oligoformer_app,
        "finalize_oligoformer_pita_shard_batch",
        FakeFinalizePitaShardBatch(),
    )
    monkeypatch.setattr(oligoformer_app, "package_outputs", fake_package_outputs)
    monkeypatch.setenv(oligoformer_app.APP_INFO.off_target_prep_workers_env, "2")
    monkeypatch.setenv(oligoformer_app.APP_INFO.off_target_nodes_env, "4")
    monkeypatch.setenv(oligoformer_app.APP_INFO.off_target_workers_env, "32")

    result = oligoformer_app.run_oligoformer_postprocess.get_raw_f()(
        plan=plan,
        off_target=True,
        toxicity=True,
        top_n=-1,
    )

    final_table = layout.outputs_dir.joinpath("target.txt").read_text(encoding="utf-8")
    raw_off_target_dir = layout.prep_dir / "off_target" / "target"
    assert result == b"archive"
    assert len(targetscan_branch.spawn_calls) == 1
    assert len(pita_branch.spawn_calls) == 1
    assert len(gather_calls) == 1
    assert len(targetscan_plan_calls) == 1
    assert [record.name for record in targetscan_plan_calls[0].records] == [
        "RNA0",
        "RNA1",
    ]
    assert targetscan_plan_calls[0].ref_shard_size == 1000
    assert targetscan_plan_calls[0].shard_index == 0
    assert [len(batch) for batch, _ in targetscan_batch_calls] == [1]
    assert [workers for _, workers in targetscan_batch_calls] == [32]
    assert sorted(call.record_name for call in pita_plan_calls) == ["RNA0", "RNA1"]
    assert [len(batch) for batch, _ in pita_prepare_batch_calls] == [2]
    assert [workers for _, workers in pita_prepare_batch_calls] == [32]
    assert all(call.output_dir == str(layout.outputs_dir) for call in pita_plan_calls)
    assert len(row_batch_calls) == 1
    assert all(local_workers == 32 for _, local_workers in row_batch_calls)
    batched_rows = [row for rows, _ in row_batch_calls for row in rows]
    assert [row.record_name for row in batched_rows] == ["RNA0", "RNA1"]
    assert all(
        Path(row.log_path).is_relative_to(layout.outputs_dir / "logs" / "off_target")
        for row in batched_rows
    )
    assert [len(batch) for batch, _ in finalize_batch_calls] == [2]
    assert [workers for _, workers in finalize_batch_calls] == [2]
    assert [call.record_name for call in finalize_calls] == ["RNA0", "RNA1"]
    assert captured["package_root"] == layout.outputs_dir
    assert captured["package_paths"] == [
        "target.txt",
        "target_ranked.txt",
        "target_ranked_filtered.txt",
    ]
    assert final_table.splitlines() == [
        "pos\tsense\tsiRNA\tefficacy\tfunc_filter\tpita_score\t"
        "targetscan_score\toff_target_filter\tSeed\tcell_viability\t"
        "toxicity_filter\tfilter",
        "1\tGC\tAUGCUAGCUAGCUAGCUAG\t0.8\t0\t-11\t2.0\t1\tUGCUAG\t60\t0\t1",
        "2\tCG\tUGCUAGCUAGCUAGCUAGC\t0.6\t0\t-2\t0.5\t0\tGCUAGC\t60\t0\t0",
    ]
    assert raw_off_target_dir.joinpath("pita.tab").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "microRNA\tScore",
        "RNA0\t-11",
        "RNA1\t-2",
    ]
    assert raw_off_target_dir.joinpath("targetscan.tab").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "ref\tRNA0\t2",
        "ref\tRNA1\t0.5",
    ]
    assert raw_off_target_dir.joinpath("off_target.done").exists()
    assert not raw_off_target_dir.joinpath("targetscan").exists()
    assert not raw_off_target_dir.joinpath("targetscan_ref_shards").exists()
    assert not raw_off_target_dir.joinpath("00000_RNA0").exists()
    assert not raw_off_target_dir.joinpath("00001_RNA1").exists()
    assert volume.reload_count >= 4
    assert volume.commit_count >= 3


def test_run_off_target_shards_reuses_merged_raw_evidence(tmp_path: Path, monkeypatch):
    layout = oligoformer_app.AppRunLayout.from_run_root(tmp_path / "run")
    raw_off_target_dir = layout.prep_dir / "off_target" / "target"
    raw_off_target_dir.mkdir(parents=True)
    raw_off_target_dir.joinpath("off_target.done").write_text("done", encoding="utf-8")
    raw_off_target_dir.joinpath("pita.tab").write_text(
        "microRNA\tScore\nRNA0\t-1\n",
        encoding="utf-8",
    )
    raw_off_target_dir.joinpath("targetscan.tab").write_text(
        "ref\tRNA0\t0.5\n",
        encoding="utf-8",
    )
    infer_dir = tmp_path / "infer"
    infer_dir.mkdir()

    class ExplodingBranch:
        def spawn(self, *_args, **_kwargs):
            raise AssertionError("cached raw evidence should skip branch fanout")

    monkeypatch.setattr(
        oligoformer_app, "run_oligoformer_targetscan_branch", ExplodingBranch()
    )
    monkeypatch.setattr(
        oligoformer_app, "run_oligoformer_pita_branch", ExplodingBranch()
    )

    oligoformer_app._run_off_target_shards(
        run_root=str(layout.run_root),
        records=[oligoformer_app.OffTargetSirnaRecord("RNA0", "AUGCUAGCUAGCUAGCUAG")],
        stem="target",
        utr_path=str(tmp_path / "utr.txt"),
        orf_path=str(tmp_path / "orf.txt"),
        infer_dir=infer_dir,
        output_dir=layout.outputs_dir,
        logs_dir=layout.outputs_dir / "logs" / "off_target" / "target",
    )

    assert infer_dir.joinpath("pita.tab").read_text(encoding="utf-8") == (
        "microRNA\tScore\nRNA0\t-1\n"
    )
    assert infer_dir.joinpath("targetscan.tab").read_text(encoding="utf-8") == (
        "ref\tRNA0\t0.5\n"
    )


def test_apply_off_target_filters_handles_header_only_pita(tmp_path: Path, monkeypatch):
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))

    def fake_run_off_target_shards(*, infer_dir: Path, **_kwargs) -> None:
        infer_dir.mkdir(parents=True, exist_ok=True)
        infer_dir.joinpath("pita.tab").write_text(
            "RefSeq\tmicroRNA\tSites\tScore\n",
            encoding="utf-8",
        )
        infer_dir.joinpath("targetscan.tab").write_text(
            "ref\tRNA0\t0.5\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        oligoformer_app, "_run_off_target_shards", fake_run_off_target_shards
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
        run_root=str(tmp_path / "run"),
        stem="target",
        utr_path=str(tmp_path / "utr.txt"),
        orf_path=str(tmp_path / "orf.txt"),
        output_dir=tmp_path / "outputs",
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


def test_apply_off_target_filters_fills_missing_targetscan_for_pita_hits(
    tmp_path: Path, monkeypatch
):
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))

    def fake_run_off_target_shards(*, infer_dir: Path, **_kwargs) -> None:
        infer_dir.mkdir(parents=True, exist_ok=True)
        infer_dir.joinpath("pita.tab").write_text(
            "RefSeq\tmicroRNA\tSites\tScore\nref\tRNA0\t1\t-2\n",
            encoding="utf-8",
        )
        infer_dir.joinpath("targetscan.tab").write_text(
            "ref\tRNA1\t0.5\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        oligoformer_app, "_run_off_target_shards", fake_run_off_target_shards
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
        run_root=str(tmp_path / "run"),
        stem="target",
        utr_path=str(tmp_path / "utr.txt"),
        orf_path=str(tmp_path / "orf.txt"),
        output_dir=tmp_path / "outputs",
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


def test_oligoformer_off_target_defaults_use_distributed_cpu_nodes(monkeypatch):
    monkeypatch.delenv(oligoformer_app.APP_INFO.off_target_nodes_env, raising=False)
    monkeypatch.delenv(oligoformer_app.APP_INFO.off_target_workers_env, raising=False)
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.off_target_prep_workers_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.off_target_pita_prepare_nodes_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.off_target_pita_prepare_workers_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.off_target_pita_prepare_utr_shard_size_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.off_target_row_shard_size_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.targetscan_rnaplfold_nodes_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.targetscan_rnaplfold_workers_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.targetscan_rnaplfold_shard_size_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.targetscan_prepare_nodes_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.targetscan_prepare_ref_shard_size_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.targetscan_context_nodes_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.targetscan_context_workers_env,
        raising=False,
    )
    monkeypatch.delenv(
        oligoformer_app.APP_INFO.targetscan_context_shard_size_env,
        raising=False,
    )

    assert (
        oligoformer_app._bounded_node_count(
            100,
            env_name=oligoformer_app.APP_INFO.off_target_nodes_env,
            default=oligoformer_app.APP_INFO.default_off_target_nodes,
        )
        == 32
    )
    assert (
        oligoformer_app._bounded_node_count(
            100,
            env_name=oligoformer_app.APP_INFO.off_target_pita_prepare_nodes_env,
            default=oligoformer_app.APP_INFO.default_pita_prepare_nodes,
        )
        == 32
    )
    assert (
        oligoformer_app._bounded_node_count(
            100,
            env_name=oligoformer_app.APP_INFO.targetscan_rnaplfold_nodes_env,
            default=oligoformer_app.APP_INFO.default_targetscan_rnaplfold_nodes,
        )
        == 32
    )
    assert (
        oligoformer_app._bounded_node_count(
            100,
            env_name=oligoformer_app.APP_INFO.targetscan_prepare_nodes_env,
            default=oligoformer_app.APP_INFO.default_targetscan_prepare_nodes,
        )
        == 32
    )
    assert (
        oligoformer_app._bounded_node_count(
            100,
            env_name=oligoformer_app.APP_INFO.targetscan_context_nodes_env,
            default=oligoformer_app.APP_INFO.default_targetscan_context_nodes,
        )
        == 32
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.off_target_workers_env,
            oligoformer_app.APP_INFO.default_off_target_workers_per_node,
        )
        == 32
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.off_target_prep_workers_env,
            oligoformer_app.APP_INFO.default_off_target_prep_workers,
        )
        == 16
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.off_target_pita_prepare_workers_env,
            oligoformer_app.APP_INFO.default_pita_prepare_workers,
        )
        == 32
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.off_target_pita_prepare_utr_shard_size_env,
            oligoformer_app.APP_INFO.default_pita_prepare_utr_shard_size,
        )
        == 1000
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.off_target_row_shard_size_env,
            oligoformer_app.APP_INFO.default_pita_row_shard_size,
        )
        == 1000
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.off_target_row_attempts_env,
            oligoformer_app.APP_INFO.default_pita_row_attempts,
        )
        == 3
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.targetscan_rnaplfold_workers_env,
            oligoformer_app.APP_INFO.default_targetscan_rnaplfold_workers,
        )
        == 8
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.targetscan_rnaplfold_shard_size_env,
            oligoformer_app.APP_INFO.default_targetscan_rnaplfold_shard_size,
        )
        == 500
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.targetscan_prepare_ref_shard_size_env,
            oligoformer_app.APP_INFO.default_targetscan_prepare_ref_shard_size,
        )
        == 1000
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.targetscan_context_workers_env,
            oligoformer_app.APP_INFO.default_targetscan_context_workers,
        )
        == 32
    )
    assert (
        oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.targetscan_context_shard_size_env,
            oligoformer_app.APP_INFO.default_targetscan_context_shard_size,
        )
        == 500
    )


def test_targetscan_batch_specs_split_transcript_aligned_refs(
    tmp_path: Path,
    monkeypatch,
):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))
    monkeypatch.setenv(
        oligoformer_app.APP_INFO.targetscan_prepare_ref_shard_size_env,
        "2",
    )
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
    assert all(spec.records == tuple(records) for spec in specs)
    assert [
        Path(spec.run_root)
        / "prepare"
        / "off_target"
        / "target"
        / "targetscan"
        / f"size_{spec.ref_shard_size}"
        / f"{spec.shard_index:05d}"
        for spec in specs
    ] == [oligoformer_app._targetscan_batch_cache_dir(spec) for spec in specs]

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


def test_pita_prepare_splits_utr_stab_and_launches_remote_shards_in_order(
    tmp_path: Path, monkeypatch
):
    remote_batches = []
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    repo_dir.joinpath("off-target", "pita", "lib").mkdir(parents=True)
    utr_stab_path = tmp_path / "utr.stab"
    mir_stab_path = tmp_path / "mir.stab"
    potential_targets_path = tmp_path / "cache" / "potential_targets.tsv"
    utr_stab_path.write_text(
        "utr1\tAAAA\nutr2\tCCCC\nutr3\tGGGG\n",
        encoding="utf-8",
    )
    mir_stab_path.write_text("RNA0\tUUUU\n", encoding="utf-8")

    class FakePitaPrepareUtrShardBatch:
        def remote(self, specs, local_workers):
            remote_batches.append((specs, local_workers))
            outputs = []
            for spec in specs:
                rows = Path(spec.input_path).read_text(encoding="utf-8").splitlines()
                row_names = [row.split("\t", 1)[0] for row in rows]
                output_path = Path(spec.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    "".join(f"RNA0\t{name}\n" for name in row_names),
                    encoding="utf-8",
                )
                output_path.with_suffix(output_path.suffix + ".done").write_text(
                    "done", encoding="utf-8"
                )
                outputs.append(str(output_path))
            return outputs

    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(
        oligoformer_app,
        "run_oligoformer_pita_prepare_utr_shard_batch",
        FakePitaPrepareUtrShardBatch(),
    )
    monkeypatch.setenv(
        oligoformer_app.APP_INFO.off_target_pita_prepare_utr_shard_size_env,
        "2",
    )
    monkeypatch.setenv(
        oligoformer_app.APP_INFO.off_target_pita_prepare_workers_env,
        "2",
    )
    monkeypatch.setenv(
        oligoformer_app.APP_INFO.off_target_pita_prepare_nodes_env,
        "2",
    )

    specs = oligoformer_app._pita_prepare_utr_shard_specs(
        utr_stab_path=utr_stab_path,
        mir_stab_path=mir_stab_path,
        shard_dir=tmp_path / "cache" / "pita_prepare_utr_shards",
        logs_dir=tmp_path / "logs",
        shard_size=oligoformer_app._positive_int_from_env(
            oligoformer_app.APP_INFO.off_target_pita_prepare_utr_shard_size_env,
            oligoformer_app.APP_INFO.default_pita_prepare_utr_shard_size,
        ),
    )
    volume.commit()
    outputs = oligoformer_app._run_pita_prepare_utr_shard_batches(list(specs))
    volume.reload()
    row_count = oligoformer_app._write_pita_potential_targets_from_outputs(
        outputs=outputs,
        potential_targets_path=potential_targets_path,
    )

    assert row_count == 3
    assert potential_targets_path.read_text(encoding="utf-8") == (
        "RNA0\tutr1\nRNA0\tutr2\nRNA0\tutr3\n"
    )
    remote_specs = [spec for batch, _workers in remote_batches for spec in batch]
    assert [spec.shard_index for spec in remote_specs] == [0, 1]
    assert [workers for _batch, workers in remote_batches] == [2]
    assert {Path(spec.log_path).parent.name for spec in remote_specs} == {
        "pita_prepare_utr_shards"
    }
    assert volume.commit_count == 1
    assert volume.reload_count == 1


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
    monkeypatch.setenv(oligoformer_app.APP_INFO.off_target_row_attempts_env, "2")

    assert oligoformer_app._run_pita_row_shard(spec) == str(output_path)
    assert output_path.read_text(encoding="utf-8") == "success\n"
    assert output_path.with_suffix(output_path.suffix + ".done").exists()
    assert list(output_path.parent.glob("*.tmp.*")) == []
    assert len(attempts) == 2
    assert "Retrying OligoFormer PITA row shard RNA0:0" in capfd.readouterr().out


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
    volume = FakeVolume()
    info = oligoformer_app.AppInfo(
        repo_rnafm_dir=tmp_path / "repo" / "RNA-FM",
        model_rnafm_dir=tmp_path / "models" / "RNA-FM",
        repo_ref_dir=tmp_path / "repo" / "off-target" / "ref",
        model_ref_dir=tmp_path / "models" / "off-target" / "ref",
    )
    calls = []

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

    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    monkeypatch.setattr(oligoformer_app, "MODEL_VOLUME", volume)
    monkeypatch.setattr(oligoformer_app, "download_files", fake_download_files)

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
    assert not info.targetscan_rnaplfold_marker_path.exists()
    assert volume.commit_count == 1
    assert "min(Args.top_n, RESULT_ranked.shape[0])" in source
    assert "--biomodals_stage" in source


def test_submit_oligoformer_orchestrates_split_run(tmp_path: Path, monkeypatch) -> None:
    input_fasta = tmp_path / "target.fa"
    input_fasta.write_text(">m\nAUGCUAGCUAGCUAGCUAGC\n", encoding="utf-8")
    calls = []
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        run_root="/remote/run",
        efficacy_dir="/remote/run/prepare/efficacy",
        output_dir="/remote/run/outputs",
        output_stems=("m",),
        efficacy_ready=False,
        final_ready=False,
    )
    efficacy_plan = replace(plan, efficacy_ready=True)

    class FakeDownloadModels:
        def remote(self, *, force):
            calls.append(("download", force))

    class FakePrepare:
        def remote(self, **kwargs):
            calls.append(("prepare", kwargs))
            return plan

    class FakeEfficacy:
        def remote(self, **kwargs):
            calls.append(("efficacy", kwargs))
            return efficacy_plan

    class FakePostprocess:
        def remote(self, **kwargs):
            calls.append(("postprocess", kwargs))
            return b"archive"

    monkeypatch.setattr(
        oligoformer_app, "download_oligoformer_models", FakeDownloadModels()
    )
    monkeypatch.setattr(oligoformer_app, "prepare_oligoformer_run", FakePrepare())
    monkeypatch.setattr(oligoformer_app, "run_oligoformer_efficacy", FakeEfficacy())
    monkeypatch.setattr(
        oligoformer_app, "run_oligoformer_postprocess", FakePostprocess()
    )
    raw_f = oligoformer_app.submit_oligoformer_task.info.raw_f
    assert raw_f is not None

    raw_f(mrna_fasta=str(input_fasta), out_dir=str(tmp_path), run_name="demo")

    assert calls[0] == ("download", False)
    assert calls[1][0] == "prepare"
    assert calls[1][1]["top_n"] == oligoformer_app.APP_INFO.default_top_n
    assert calls[2] == (
        "efficacy",
        {"plan": plan, "functionality_filter": True},
    )
    assert calls[3][0] == "postprocess"
    assert calls[3][1]["plan"] == efficacy_plan
    assert (tmp_path / "demo_oligoformer.tar.zst").read_bytes() == b"archive"


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
