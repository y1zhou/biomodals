"""Tests for standalone ENsiRNA app behavior."""

# ruff: noqa: D101,D102,D103,D107

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from biomodals.app.score import ensirna_app


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def test_runtime_image_uses_rosetta_base_build() -> None:
    source = Path(ensirna_app.__file__).read_text(encoding="utf-8")

    assert 'from_registry("rosettacommons/rosetta:serial-420"' in source
    assert ".debian_slim(" not in source
    assert "tanwenchong/ensirna:v2" not in source
    assert '"MAMBA_ROOT_PREFIX": APP_INFO.mamba_root' in source
    assert '"PATH": APP_INFO.mamba_bin_path' in source
    assert "def rosetta_extract_shim" in source
    assert "rna_denovo.static.linuxgccrelease" in source
    assert "def get_pdb_runtime_patch" in source
    assert "expected_len = 61 + len(seq2) + len(seq1) + 1 + 1" in source
    assert "def _fit_secstruct(secstruct, size):" in source
    assert "-out:file:silent" in source
    assert "https://huggingface.co/cuhkaih/rnafm/resolve/main" in source
    assert "RNA-FM_pretrained.pth" in source
    assert "find . -path '*/pkl/*.ckpt' -delete" in source
    assert "download_ensirna_models" in source
    assert "ensirna_prepare_inputs" in source
    assert "ensirna_prepare_pdb_chunk" in source
    assert "ensirna_finalize_prepared_inputs" in source
    assert "run_ensirna_inference" in source
    assert "MODEL_VOLUME.commit()" in source
    assert "download_files(" in source
    assert "curl -fL --retry" not in source
    assert "micromamba create" not in source
    assert ".micromamba_install(" in source
    assert "viennarna=2.6.4-0" in source
    assert ".uv_pip_install(*APP_INFO.pip_packages)" in source
    assert ".uv_pip_install(*APP_INFO.torch_packages" in source
    assert "https://download.pytorch.org/whl/cu118" in source
    assert "rna-fm" in source
    assert "ENSIRNA_RNAFM_DEVICE" in source
    assert "ENSIRNA_PDB_CORES" in source
    assert '"data.dataset"' in source
    assert '"data.get_pdb"' in source
    assert '"run.py"' in source
    assert "ignore_dep_versions=True" in source
    assert 'skip_deps=["uniaf3"]' in source


def test_download_ensirna_models_writes_to_model_volume(monkeypatch) -> None:
    captured = {}
    volume = FakeVolume()

    def fake_download_files(urls, **kwargs):
        captured["urls"] = urls
        captured["kwargs"] = kwargs

    monkeypatch.setattr(ensirna_app, "download_files", fake_download_files)
    monkeypatch.setattr(ensirna_app, "MODEL_VOLUME", volume)

    ensirna_app.download_ensirna_models.get_raw_f()(force=True)

    assert len(captured["urls"]) == 6
    assert ensirna_app.APP_INFO.rnafm_pretrained_url in captured["urls"]
    assert captured["kwargs"]["force"] is True
    assert captured["kwargs"]["num_retries"] == 3
    assert volume.commit_count == 1
    for filename in ensirna_app.APP_INFO.checkpoint_filenames:
        assert (ensirna_app.APP_INFO.checkpoint_dir / filename) in captured[
            "urls"
        ].values()


def test_prepare_inputs_reuses_completed_volume_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_volume = FakeVolume()
    monkeypatch.setattr(
        ensirna_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume, output_volume_mountpoint=str(tmp_path)
        ),
    )
    fasta = b">m\nAUGCUAGCUAGCUAGCUAGC\n"
    cache_key = ensirna_app._cache_key_for_fasta(fasta)
    layout = ensirna_app._layout_for_cache_key(cache_key)
    layout.outputs_dir.joinpath("mrna_processed").mkdir(parents=True)
    layout.outputs_dir.joinpath("mrna.json").write_text("{}\n", encoding="utf-8")
    layout.outputs_dir.joinpath("mrna_processed", "_metainfo").write_text(
        "{}", encoding="utf-8"
    )
    layout.markers_dir.mkdir(parents=True)
    ensirna_app._prepared_marker_path(layout).write_text(
        (f'{{"cache_key":"{cache_key}","candidate_count":2,"chunk_count":3}}'),
        encoding="utf-8",
    )

    def fake_run_command(*args, **kwargs):
        raise AssertionError("prepare cache hit should not run commands")

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)

    result = ensirna_app.ensirna_prepare_inputs.get_raw_f()(
        mrna_fasta_bytes=fasta,
        run_name="demo",
        max_prepare_jobs=8,
    )

    assert result.cached is True
    assert result.candidate_count == 2
    assert result.chunk_count == 3
    assert result.chunks == []
    assert result.prepared_dir == str(layout.run_root)
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 0


def test_prepare_inputs_creates_cpu_chunk_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}
    ensirna_dir = tmp_path / "ENsiRNA"
    ensirna_dir.mkdir()
    output_volume = FakeVolume()
    monkeypatch.setattr(
        ensirna_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume, output_volume_mountpoint=str(tmp_path)
        ),
    )

    def fake_run_command(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        out_dir = Path(cmd[cmd.index("-o") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir.joinpath("mrna.csv").write_text(
            "siRNA,anti seq,sense seq,mRNA_seq,position,efficacy\n"
            "m_0,GC,CG,AUGC,0,0\n"
            "m_1,GC,CG,AUGC,1,0\n"
            "m_2,GC,CG,AUGC,2,0\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
        ),
    )

    result = ensirna_app.ensirna_prepare_inputs.get_raw_f()(
        mrna_fasta_bytes=b">m\nAUGCUAGCUAGCUAGCUAGC\n",
        run_name="demo",
        max_prepare_jobs=2,
    )

    assert result.cached is False
    assert result.candidate_count == 3
    assert result.chunk_count == 2
    assert len(result.chunks) == 2
    assert captured["cmd"][:5] == [
        "micromamba",
        "run",
        "-n",
        ensirna_app.APP_INFO.conda_env_name,
        "python",
    ]
    assert "get_siRNA.py" in captured["cmd"]
    assert captured["cwd"] == ensirna_dir
    assert (
        Path(result.chunks[0].csv_path)
        .read_text(encoding="utf-8")
        .startswith("siRNA,anti seq")
    )
    assert Path(result.chunks[0].pdb_dir).name == "mrna_pdb"
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_prepare_pdb_chunk_runs_rosetta_on_cpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}
    ensirna_dir = tmp_path / "ENsiRNA"
    ensirna_dir.mkdir()
    output_volume = FakeVolume()
    monkeypatch.setattr(
        ensirna_app,
        "CONF",
        SimpleNamespace(output_volume=output_volume),
    )
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(ensirna_app.APP_INFO, ensirna_dir=ensirna_dir),
    )
    chunk_csv = tmp_path / "chunk.csv"
    chunk_csv.write_text("siRNA\nm_0\n", encoding="utf-8")
    pdb_dir = tmp_path / "mrna_pdb"
    pdb_dir.mkdir()

    def fake_run_command(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        Path(cmd[cmd.index("-f") + 1]).with_suffix(".json").write_text(
            "{}\n", encoding="utf-8"
        )

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)
    result = ensirna_app.ensirna_prepare_pdb_chunk.get_raw_f()(
        chunk=ensirna_app.EnsirnaPdbChunkSpec(
            chunk_name="chunk_0000",
            csv_path=str(chunk_csv),
            json_path=str(chunk_csv.with_suffix(".json")),
            pdb_dir=str(pdb_dir),
        ),
        pdb_cores=3,
    )

    assert result["cached"] == 1
    assert captured["cmd"][:7] == [
        "micromamba",
        "run",
        "-n",
        ensirna_app.APP_INFO.conda_env_name,
        "python",
        "-m",
        "data.get_pdb",
    ]
    assert captured["cwd"] == ensirna_dir
    assert captured["env"] == {ensirna_app.APP_INFO.pdb_cores_env: "3"}
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_finalize_prepared_inputs_merges_json_and_preprocesses_on_cpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}
    output_volume = FakeVolume()
    ensirna_dir = tmp_path / "ENsiRNA"
    ensirna_dir.mkdir()
    rnafm_cache = tmp_path / "models" / "RNA-FM_pretrained.pth"
    rnafm_cache.parent.mkdir()
    rnafm_cache.write_bytes(b"weights")
    monkeypatch.setattr(
        ensirna_app,
        "CONF",
        SimpleNamespace(output_volume=output_volume),
    )
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
            rnafm_cache_path=rnafm_cache,
        ),
    )
    layout = ensirna_app.AppRunLayout.from_run_root(tmp_path / "prepared")
    layout.prep_dir.mkdir(parents=True)
    chunk_a = layout.prep_dir / "chunk_0000.json"
    chunk_b = layout.prep_dir / "chunk_0001.json"
    chunk_a.write_text('{"siRNA":"m_0"}\n', encoding="utf-8")
    chunk_b.write_text('{"siRNA":"m_1"}\n', encoding="utf-8")
    plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key="abc123",
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / "mrna.json"),
        processed_dir=str(layout.outputs_dir / "mrna_processed"),
        result_xlsx=str(layout.outputs_dir / "mrna_result.xlsx"),
        candidate_count=2,
        chunk_count=2,
        chunks=[
            ensirna_app.EnsirnaPdbChunkSpec(
                "chunk_0000", "unused.csv", str(chunk_a), "unused_pdb"
            ),
            ensirna_app.EnsirnaPdbChunkSpec(
                "chunk_0001", "unused.csv", str(chunk_b), "unused_pdb"
            ),
        ],
        cached=False,
    )

    def fake_run_command(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        Path(plan.processed_dir).mkdir(parents=True)
        Path(plan.processed_dir, "_metainfo").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)
    result = ensirna_app.ensirna_finalize_prepared_inputs.get_raw_f()(plan)

    assert Path(plan.json_path).read_text(encoding="utf-8") == (
        '{"siRNA":"m_0"}\n{"siRNA":"m_1"}\n'
    )
    assert result.chunks == []
    assert result.cached is False
    assert ensirna_app._prepared_marker_path(layout).exists()
    assert captured["cmd"][:7] == [
        "micromamba",
        "run",
        "-n",
        ensirna_app.APP_INFO.conda_env_name,
        "python",
        "-m",
        "data.dataset",
    ]
    assert captured["cwd"] == ensirna_dir
    assert captured["env"] == {ensirna_app.APP_INFO.rnafm_device_env: "cpu"}
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_run_ensirna_inference_uses_prepared_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}
    output_volume = FakeVolume()
    ensirna_dir = tmp_path / "ENsiRNA"
    checkpoint_dir = tmp_path / "models" / "pkl"
    (ensirna_dir / "pkl").mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    for filename in ensirna_app.APP_INFO.checkpoint_filenames:
        checkpoint_dir.joinpath(filename).write_bytes(b"checkpoint")
    monkeypatch.setattr(
        ensirna_app,
        "CONF",
        SimpleNamespace(output_volume=output_volume),
    )
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
            checkpoint_dir=checkpoint_dir,
        ),
    )
    layout = ensirna_app.AppRunLayout.from_run_root(tmp_path / "prepared")
    layout.outputs_dir.joinpath("mrna_processed").mkdir(parents=True)
    layout.outputs_dir.joinpath("mrna.json").write_text("{}\n", encoding="utf-8")
    layout.outputs_dir.joinpath("mrna_processed", "_metainfo").write_text(
        "{}", encoding="utf-8"
    )
    layout.markers_dir.mkdir(parents=True)
    ensirna_app._prepared_marker_path(layout).write_bytes(
        b'{"candidate_count":1,"chunk_count":1}'
    )

    def fake_run_command(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        layout.outputs_dir.joinpath("mrna_result.xlsx").write_bytes(b"xlsx")

    def fake_package_outputs(root, **kwargs):
        captured["package_root"] = Path(root)
        captured["package_paths"] = kwargs["paths_to_bundle"]
        return b"archive"

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)
    monkeypatch.setattr(ensirna_app, "package_outputs", fake_package_outputs)
    result = ensirna_app.run_ensirna_inference.get_raw_f()(
        prepared_dir=str(layout.run_root)
    )

    assert result == b"archive"
    assert captured["cmd"][:6] == [
        "micromamba",
        "run",
        "-n",
        ensirna_app.APP_INFO.conda_env_name,
        "python",
        "run.py",
    ]
    assert captured["cmd"][captured["cmd"].index("--test_set") + 1] == str(
        layout.outputs_dir / "mrna.json"
    )
    assert captured["cmd"][captured["cmd"].index("--save_dir") + 1] == str(
        layout.outputs_dir
    )
    assert captured["cmd"][captured["cmd"].index("--gpu") + 1] == "0"
    assert captured["cmd"][captured["cmd"].index("--id") + 1] == "mrna"
    assert captured["cwd"] == ensirna_dir
    assert captured["package_root"] == layout.outputs_dir
    assert captured["package_paths"] == ensirna_app.APP_INFO.output_paths
    for filename in ensirna_app.APP_INFO.checkpoint_filenames:
        assert (ensirna_dir / "pkl" / filename).resolve() == checkpoint_dir / filename
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_submit_ensirna_writes_local_tarball(tmp_path: Path, monkeypatch) -> None:
    input_fasta = tmp_path / "target.fa"
    input_fasta.write_text(">m\nAUGCUAGCUAGCUAGCUAGC\n", encoding="utf-8")
    captured = {}

    class FakeDownload:
        def remote(self, *, force: bool):
            captured["download_force"] = force

    class FakeRun:
        def remote(self, **kwargs):
            captured["run"] = kwargs
            return b"archive"

    monkeypatch.setattr(ensirna_app, "download_ensirna_models", FakeDownload())
    chunk = ensirna_app.EnsirnaPdbChunkSpec(
        chunk_name="chunk_0000",
        csv_path="/remote/prepare/chunk_0000.csv",
        json_path="/remote/prepare/chunk_0000.json",
        pdb_dir="/remote/outputs/mrna_pdb",
    )
    prepare_plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key="abc123",
        prepared_dir="/remote/prepared",
        json_path="/remote/outputs/mrna.json",
        processed_dir="/remote/outputs/mrna_processed",
        result_xlsx="/remote/outputs/mrna_result.xlsx",
        candidate_count=1,
        chunk_count=1,
        chunks=[chunk],
        cached=False,
    )
    finalized_plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key="abc123",
        prepared_dir="/remote/prepared",
        json_path="/remote/outputs/mrna.json",
        processed_dir="/remote/outputs/mrna_processed",
        result_xlsx="/remote/outputs/mrna_result.xlsx",
        candidate_count=1,
        chunk_count=1,
        chunks=[],
        cached=False,
    )

    class FakePrepare:
        def remote(self, **kwargs):
            captured["prepare"] = kwargs
            return prepare_plan

    class FakePdbChunk:
        def remote(self, **kwargs):
            captured["chunk"] = kwargs
            return {"cached": 1}

    class FakeFinalize:
        def remote(self, plan):
            captured["finalize"] = plan
            return finalized_plan

    monkeypatch.setattr(ensirna_app, "ensirna_prepare_inputs", FakePrepare())
    monkeypatch.setattr(ensirna_app, "ensirna_prepare_pdb_chunk", FakePdbChunk())
    monkeypatch.setattr(ensirna_app, "ensirna_finalize_prepared_inputs", FakeFinalize())
    monkeypatch.setattr(ensirna_app, "run_ensirna_inference", FakeRun())
    raw_f = ensirna_app.submit_ensirna_task.info.raw_f
    assert raw_f is not None

    raw_f(
        mrna_fasta=str(input_fasta),
        out_dir=str(tmp_path),
        run_name="demo",
    )

    assert captured["download_force"] is False
    assert captured["prepare"] == {
        "mrna_fasta_bytes": b">m\nAUGCUAGCUAGCUAGCUAGC\n",
        "run_name": "demo",
        "max_prepare_jobs": 4,
        "force": False,
    }
    assert captured["chunk"] == {"chunk": chunk, "pdb_cores": 1}
    assert captured["finalize"] == prepare_plan
    assert captured["run"] == {"prepared_dir": "/remote/prepared", "force": False}
    assert (tmp_path / "demo_ensirna.tar.zst").read_bytes() == b"archive"
