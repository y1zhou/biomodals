"""Tests for standalone OligoFormer app behavior."""

# ruff: noqa: D101,D102,D103,D107

import shlex
import subprocess as sp
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    prepare_calls: list[oligoformer_app.OffTargetShardSpec] = []
    prepare_batch_calls: list[tuple[list[oligoformer_app.OffTargetShardSpec], int]] = []
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

    class FakePrepareOffTargetShardBatch:
        def remote(
            self,
            specs: list[oligoformer_app.OffTargetShardSpec],
            local_workers: int,
        ):
            prepare_batch_calls.append((list(specs), local_workers))
            prepared_shards = []
            for spec in specs:
                prepare_calls.append(spec)
                cache_dir = oligoformer_app._off_target_shard_cache_dir(spec)
                row_dir = cache_dir / "pita_rows"
                row_dir.mkdir(parents=True, exist_ok=True)
                cache_dir.joinpath("potential_targets.tsv").write_text(
                    "potential\n", encoding="utf-8"
                )
                cache_dir.joinpath("ext_utr.stab").write_text("utr\n", encoding="utf-8")
                targetscan_path = cache_dir / "targetscan.tab"
                targetscan_score = "2" if spec.record_name == "RNA0" else "0.5"
                targetscan_path.write_text(
                    f"ref\t{spec.record_name}\t{targetscan_score}\n",
                    encoding="utf-8",
                )
                row = oligoformer_app.PitaRowShardSpec(
                    run_root=spec.run_root,
                    stem=spec.stem,
                    sirna_index=spec.index,
                    record_name=spec.record_name,
                    shard_index=0,
                    start_row=0,
                    end_row=1,
                    potential_targets_path=str(cache_dir / "potential_targets.tsv"),
                    input_path=str(
                        row_dir / "00000_000000000000_000000000001.potential.tsv"
                    ),
                    ext_utr_path=str(cache_dir / "ext_utr.stab"),
                    output_path=str(
                        row_dir / "00000_000000000000_000000000001.scored.tsv"
                    ),
                    log_path=str(
                        layout.outputs_dir
                        / "logs"
                        / "off_target"
                        / spec.stem
                        / f"{spec.index:05d}_{spec.record_name}"
                        / "pita_rows"
                        / "00000_000000000000_000000000001.log"
                    ),
                )
                Path(row.input_path).write_text("potential\n", encoding="utf-8")
                prepared_shards.append(
                    oligoformer_app.PreparedOffTargetShard(
                        index=spec.index,
                        record_name=spec.record_name,
                        cache_dir=str(cache_dir),
                        logs_dir=str(
                            layout.outputs_dir
                            / "logs"
                            / "off_target"
                            / spec.stem
                            / f"{spec.index:05d}_{spec.record_name}"
                        ),
                        pita_path=str(cache_dir / "pita.tab"),
                        targetscan_path=str(targetscan_path),
                        row_shards=(row,),
                    )
                )
            return prepared_shards

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
                        targetscan_path=prepared.targetscan_path,
                    )
                )
            return results

    def fake_package_outputs(root, *, paths_to_bundle=None):
        captured["package_root"] = Path(root)
        captured["package_paths"] = [str(path) for path in paths_to_bundle or ()]
        return b"archive"

    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(
        oligoformer_app,
        "prepare_oligoformer_off_target_shard_batch",
        FakePrepareOffTargetShardBatch(),
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
    monkeypatch.setenv(oligoformer_app.APP_INFO.off_target_nodes_env, "4")
    monkeypatch.setenv(oligoformer_app.APP_INFO.off_target_workers_env, "32")

    result = oligoformer_app.run_oligoformer_postprocess.get_raw_f()(
        plan=plan,
        off_target=True,
        toxicity=True,
        top_n=-1,
    )

    final_table = layout.outputs_dir.joinpath("target.txt").read_text(encoding="utf-8")
    assert result == b"archive"
    assert [len(batch) for batch, _ in prepare_batch_calls] == [1, 1]
    assert [workers for _, workers in prepare_batch_calls] == [1, 1]
    assert [call.record_name for call in prepare_calls] == ["RNA0", "RNA1"]
    assert all(call.output_dir == str(layout.outputs_dir) for call in prepare_calls)
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
    final_lines = final_table.splitlines()
    assert final_lines[0] == (
        "pos\tsense\tsiRNA\tefficacy\tfunc_filter\tfilter\tpita_score\t"
        "targetscan_score\toff_target_filter\tSeed\tcell_viability\ttoxicity_filter"
    )
    assert final_lines[1].startswith(
        "1\tGC\tAUGCUAGCUAGCUAGCUAG\t0.8\t0\t1\t-11\t2.0\t1"
    )
    assert final_lines[2].startswith(
        "2\tCG\tUGCUAGCUAGCUAGCUAGC\t0.6\t0\t0\t-2\t0.5\t0"
    )
    assert volume.reload_count == 3
    assert volume.commit_count == 1


def test_oligoformer_off_target_defaults_use_distributed_cpu_nodes(monkeypatch):
    monkeypatch.delenv(oligoformer_app.APP_INFO.off_target_nodes_env, raising=False)
    monkeypatch.delenv(oligoformer_app.APP_INFO.off_target_workers_env, raising=False)
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

    assert oligoformer_app._off_target_nodes(100) == 32
    assert oligoformer_app._off_target_workers_per_node() == 32
    assert oligoformer_app._pita_prepare_nodes(100) == 32
    assert oligoformer_app._pita_prepare_workers() == 32
    assert oligoformer_app._pita_prepare_utr_shard_size() == 1000
    assert oligoformer_app._pita_row_shard_size() == 1000
    assert oligoformer_app._pita_row_attempts() == 3
    assert oligoformer_app._targetscan_rnaplfold_nodes(100) == 32
    assert oligoformer_app._targetscan_rnaplfold_workers() == 8
    assert oligoformer_app._targetscan_rnaplfold_shard_size() == 500
    assert oligoformer_app._targetscan_context_nodes(100) == 32
    assert oligoformer_app._targetscan_context_workers() == 32
    assert oligoformer_app._targetscan_context_shard_size() == 500


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

    row_count = oligoformer_app._write_pita_potential_targets_from_utr_shards(
        utr_stab_path=utr_stab_path,
        mir_stab_path=mir_stab_path,
        potential_targets_path=potential_targets_path,
        shard_dir=tmp_path / "cache" / "pita_prepare_utr_shards",
        logs_dir=tmp_path / "logs",
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
