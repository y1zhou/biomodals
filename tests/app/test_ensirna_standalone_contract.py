"""Tests for standalone ENsiRNA app behavior."""

# ruff: noqa: D101,D102,D103,D107

import ast
import base64
import gzip
import re
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from biomodals.app.score import ensirna_app


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


TEST_CACHE_KEY = "a" * 64


def make_test_layout(
    tmp_path: Path,
    monkeypatch,
    *,
    output_volume: FakeVolume | None = None,
):
    output_volume = output_volume or FakeVolume()
    monkeypatch.setattr(
        ensirna_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
        ),
    )
    return ensirna_app._layout_for_cache_key(TEST_CACHE_KEY)


def install_completed_cache_generation(monkeypatch, stage: str, identity: str):
    lock_key = ensirna_app.hash_string(f"{stage}\n{identity}")
    values = {
        f"{lock_key}:head": 0,
        f"{lock_key}:owner:0": {"id": "original", "acquired_at": 0.0},
        f"{lock_key}:status:0": {"state": "complete", "recorded_at": 1.0},
    }

    class FakeDict:
        def get(self, key, default=None):
            return values.get(key, default)

        def put(self, key, value, *, skip_if_exists=False):
            if skip_if_exists and key in values:
                return False
            values[key] = value
            return True

    monkeypatch.setattr(
        ensirna_app.modal.Dict,
        "from_name",
        lambda *_args, **_kwargs: FakeDict(),
    )
    return lock_key, values


def write_prepared_marker(
    layout,
    *,
    cache_key: str = TEST_CACHE_KEY,
    candidate_count: int,
    chunk_count: int,
) -> None:
    ensirna_app._write_prepared_marker(
        layout=layout,
        plan=ensirna_app.EnsirnaPreparationPlan(
            cache_key=cache_key,
            prepared_dir=str(layout.run_root),
            json_path=str(layout.outputs_dir / "mrna.json"),
            processed_dir=str(layout.outputs_dir / "mrna_processed"),
            candidate_count=candidate_count,
            chunk_count=chunk_count,
            chunks=[],
            cached=False,
        ),
        json_records=candidate_count,
    )


def seal_candidate_csv(layout, fasta: bytes) -> None:
    canonical_fasta = ensirna_app._sanitize_fasta_for_upstream(fasta)
    csv_path = layout.outputs_dir / "mrna.csv"
    facts = ensirna_app._candidate_csv_facts(csv_path, reject_unsafe_ids=True)
    assert facts is not None
    ensirna_app._write_candidate_csv_marker(
        layout=layout,
        cache_key=ensirna_app._cache_key_for_fasta(canonical_fasta),
        input_sha256=ensirna_app._bytes_sha256(canonical_fasta),
        facts=facts,
    )


def test_fasta_record_names_are_safe_unique_components() -> None:
    sanitized = ensirna_app._sanitize_fasta_for_upstream(
        b">target / one;$(touch nope)\nAUGCUAGCUAGCUAGCUAGC\n"
    )

    assert sanitized == b">target_one_touch_nope\nAUGCUAGCUAGCUAGCUAGC\n"


@pytest.mark.parametrize(
    "fasta",
    [
        b">duplicate\nAUGCUAGCUAGCUAGCUAGC\n>duplicate\nAUGCUAGCUAGCUAGCUAGC\n",
        b">a/b\nAUGCUAGCUAGCUAGCUAGC\n>a\\b\nAUGCUAGCUAGCUAGCUAGC\n",
    ],
)
def test_fasta_record_name_collisions_are_rejected(fasta: bytes) -> None:
    with pytest.raises(ValueError, match="record name"):
        ensirna_app._sanitize_fasta_for_upstream(fasta)


def test_fasta_sequence_alphabet_is_validated() -> None:
    with pytest.raises(ValueError, match="unsupported bases"):
        ensirna_app._sanitize_fasta_for_upstream(b">target\nAUGCUAGCUAGCUAGCUAG;\n")


def test_cache_layout_rejects_non_content_addressed_keys() -> None:
    with pytest.raises(ValueError, match="cache key"):
        ensirna_app._layout_for_cache_key("not-a-digest")


def test_runtime_image_uses_rosetta_base_build() -> None:
    source = Path(ensirna_app.__file__).read_text(encoding="utf-8")

    assert 'from_registry("rosettacommons/rosetta:serial-420"' in source
    assert ".debian_slim(" not in source
    assert "tanwenchong/ensirna:v2" not in source
    assert '"MAMBA_ROOT_PREFIX": APP_INFO.mamba_root' in source
    assert '"PATH": APP_INFO.mamba_bin_path' in source
    assert ensirna_app.APP_INFO.mamba_lib_path == "/root/micromamba/lib"
    assert '"LD_LIBRARY_PATH": APP_INFO.mamba_lib_path' in source
    assert "python -c 'import RNA'" in source
    assert "def rosetta_extract_shim" in source
    assert "rna_denovo.static.linuxgccrelease" in source
    assert "def get_pdb_runtime_patch" in source
    assert "get_pdb_source_sha256" in source
    assert "dataset_source_sha256" in source
    assert ".run_commands(APP_INFO.patched_sources_compile_command)" in source
    assert "expected_len = 61 + len(seq2) + len(seq1) + 1 + 1" in source
    assert "def _fit_secstruct(secstruct, size):" in source
    assert "-out:file:silent" in source
    assert ensirna_app.APP_INFO.rnafm_revision in source
    assert "RNA-FM_pretrained.pth" in source
    assert "find . -path '*/pkl/*.ckpt' -delete" in source
    assert "download_ensirna_models" in source
    assert "ensirna_prepare_inputs" in source
    assert "ensirna_prepare_pdb_chunk" in source
    assert "ensirna_finalize_prepared_inputs" in source
    assert "ensirna_preprocess_dataset" in source
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
    assert "ENSIRNA_PDB_CORES" not in source
    assert '"--num-cores"' in source
    assert "cpu=(0.125, 32.125)" in source
    assert '"data.dataset"' in source
    assert '"data.get_pdb"' in source
    assert '"run.py"' in source
    assert "ignore_dep_versions=True" in source
    assert 'skip_deps=["uniaf3"]' in source
    preprocess_block = source[
        source.index("def ensirna_preprocess_dataset") - 180 : source.index(
            "def run_ensirna_inference"
        )
    ]
    assert "gpu=CONF.gpu" in preprocess_block
    assert (
        "volumes=CONF.mounts(output_volume=True, model_volume=True)" in preprocess_block
    )
    finalize_block = source[
        source.index("def ensirna_finalize_prepared_inputs") : source.index(
            "def ensirna_preprocess_dataset"
        )
    ]
    assert '"data.dataset"' not in finalize_block


def test_runtime_patch_contract_pins_sources_and_compiles_both_modules(
    tmp_path: Path,
) -> None:
    ensirna_dir = tmp_path / "ENsiRNA"
    data_dir = ensirna_dir / "data"
    data_dir.mkdir(parents=True)
    data_dir.joinpath("get_pdb.py").write_text("unexpected", encoding="utf-8")
    data_dir.joinpath("dataset.py").write_text("unexpected", encoding="utf-8")
    app_info = replace(ensirna_app.APP_INFO, ensirna_dir=ensirna_dir)

    assert app_info.get_pdb_source_sha256 == (
        "8e509f253b552c6312f4bd655bc75a47f9f017b57925d7928ae63459fefe1fb8"
    )
    assert app_info.dataset_source_sha256 == (
        "dc3dae6f9f2b950c6a6c2a31f85b37e95f302f2402324aa8969e1fe7de2bc1c8"
    )
    assert app_info.patched_sources_compile_command == (
        f"python -m py_compile {data_dir / 'get_pdb.py'} {data_dir / 'dataset.py'}"
    )
    with pytest.raises(SystemExit, match="get_pdb.py source hash mismatch"):
        exec(app_info.get_pdb_runtime_patch, {})  # noqa: S102
    with pytest.raises(SystemExit, match="dataset.py source hash mismatch"):
        exec(app_info.dataset_runtime_patch, {})  # noqa: S102


def test_pinned_get_pdb_patch_repairs_only_mismatched_features(tmp_path: Path) -> None:
    fixture_path = (
        Path(__file__).parents[1] / "fixtures" / "ensirna" / "get_pdb.py.gz.b64"
    )
    pinned_source = gzip.decompress(base64.b64decode(fixture_path.read_bytes()))
    assert ensirna_app._bytes_sha256(pinned_source) == (
        ensirna_app.APP_INFO.get_pdb_source_sha256
    )
    ensirna_dir = tmp_path / "ENsiRNA"
    source_path = ensirna_dir / "data" / "get_pdb.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(pinned_source)
    app_info = replace(ensirna_app.APP_INFO, ensirna_dir=ensirna_dir)
    original_source = source_path.read_text(encoding="utf-8")

    def source_function(source: str, name: str, rnaplex_output: str = ""):
        function = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        )
        namespace = {
            "re": re,
            "subprocess": SimpleNamespace(
                run=lambda *_args, **_kwargs: SimpleNamespace(stdout=rnaplex_output)
            ),
        }
        exec(  # noqa: S102
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                source_path,
                "exec",
            ),
            namespace,
        )
        return namespace[name]

    candidate = {
        "siRNA": "target_0",
        "sense seq": "a" * 19,
        "anti seq": "u" * 19,
    }
    exact_rnaplex = f"{'(' * 19}&{')' * 19} 1,19 : 1,19 (-1.0)\n"
    short_rnaplex = f".{'(' * 19}&{')' * 19} 1,19 : 1,19 (-1.0)\n"
    long_rnaplex = f"{'(' * 19}&.{')' * 19} 1,19 : 1,19 (-1.0)\n"
    original_exact = source_function(original_source, "get_anti_start", exact_rnaplex)(
        None, candidate
    )
    assert original_exact is not None
    assert (
        source_function(original_source, "get_anti_start", short_rnaplex)(
            None, candidate
        )
        is None
    )
    assert (
        source_function(original_source, "get_anti_start", long_rnaplex)(
            None, candidate
        )
        is None
    )

    exec(app_info.get_pdb_runtime_patch, {})  # noqa: S102
    patched_source = source_path.read_text(encoding="utf-8")
    assert "def __init__(self,excel_dir,pdb_dir,num_cores=1):" in patched_source
    assert "parser.add_argument('--num-cores', type=int, default=1" in patched_source
    assert "Data_Prepare(filename,args.pdb_dir,args.num_cores).process()" in (
        patched_source
    )
    assert "processes=min(self.num_cores, len(chunks))," in patched_source
    assert (
        source_function(patched_source, "get_anti_start", exact_rnaplex)(
            None, candidate
        )
        == original_exact
    )
    short_positions, short_chain = source_function(
        patched_source, "get_anti_start", short_rnaplex
    )(None, candidate)
    long_positions, long_chain = source_function(
        patched_source, "get_anti_start", long_rnaplex
    )(None, candidate)
    expected_length = 61 + 19 + 19 + 1 + 1
    assert len(short_positions) == len(short_chain) == expected_length
    assert short_positions[-1] == short_positions[-2] - 1
    assert short_chain[-1] == 3
    assert len(long_positions) == len(long_chain) == expected_length
    assert long_positions[-1] == 1
    assert long_chain[-1] == 3

    fit_secstruct = source_function(patched_source, "_fit_secstruct")
    assert fit_secstruct("((..))", 6) == "((..))"
    assert fit_secstruct("((", 4) == "((.."
    assert fit_secstruct("..((..", 2) == "(("


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
    processed_dir = layout.outputs_dir / "mrna_processed"
    processed_dir.mkdir(parents=True)
    layout.outputs_dir.joinpath("mrna.json").write_text(
        '{"siRNA":"m_0"}\n{"siRNA":"m_1"}\n', encoding="utf-8"
    )
    part = processed_dir / "part_0.pkl"
    part.write_bytes(b"processed")
    processed_dir.joinpath("_metainfo").write_text(
        ('{"num_entry":2,"file_names":["' + str(part) + '"],"file_num_entries":[2]}'),
        encoding="utf-8",
    )
    write_prepared_marker(
        layout,
        cache_key=cache_key,
        candidate_count=2,
        chunk_count=3,
    )

    def fake_run_command(*args, **kwargs):
        raise AssertionError("prepare cache hit should not run commands")

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)

    result = ensirna_app.ensirna_prepare_inputs.get_raw_f()(
        mrna_fasta_bytes=fasta,
        max_prepare_jobs=8,
    )

    assert result.cached is True
    assert result.candidate_count == 2
    assert result.chunk_count == 3
    assert result.chunks == []
    assert result.prepared_dir == str(layout.run_root)
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 0


def test_completed_cache_rejects_missing_processed_parts(tmp_path: Path) -> None:
    cache_key = TEST_CACHE_KEY
    layout = ensirna_app.AppRunLayout.from_run_root(tmp_path / "prepared")
    processed_dir = layout.outputs_dir / "mrna_processed"
    processed_dir.mkdir(parents=True)
    layout.outputs_dir.joinpath("mrna.json").write_text(
        '{"siRNA":"target_0"}\n', encoding="utf-8"
    )
    processed_dir.joinpath("_metainfo").write_text(
        (
            '{"num_entry":1,"file_names":["'
            + str(processed_dir / "part_0.pkl")
            + '"],"file_num_entries":[1]}'
        ),
        encoding="utf-8",
    )
    layout.markers_dir.mkdir(parents=True)
    ensirna_app._prepared_marker_path(layout).write_text(
        (
            f'{{"schema_version":{ensirna_app.APP_INFO.cache_schema_version},'
            f'"cache_key":"{cache_key}","candidate_count":1,'
            '"chunk_count":1,"json_records":1}'
        ),
        encoding="utf-8",
    )

    assert (
        ensirna_app._cached_preparation_plan(cache_key=cache_key, layout=layout) is None
    )


def test_pdb_manifest_requires_candidate_specific_path(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdb"
    pdb_dir.mkdir()
    wrong_pdb = pdb_dir / "other.pdb"
    wrong_pdb.write_bytes(b"PDB")

    with pytest.raises(FileNotFoundError, match="PDB artifacts"):
        ensirna_app._validate_pdb_records(
            [{"siRNA": "target_0", "pdb_data_path": str(wrong_pdb)}],
            pdb_dir,
        )


def test_cache_builder_elects_one_writer_and_isolates_rebuilds(monkeypatch) -> None:
    values = {}

    class FakeDict:
        def get(self, key, default=None):
            return values.get(key, default)

        def put(self, key, value, *, skip_if_exists=False):
            if skip_if_exists and key in values:
                return False
            values[key] = value
            return True

    monkeypatch.setattr(
        ensirna_app.modal.Dict,
        "from_name",
        lambda *_args, **_kwargs: FakeDict(),
    )

    with ensirna_app._cache_build_lock("prepared", "identity") as owns_first:
        assert owns_first is True
    with ensirna_app._cache_build_lock("prepared", "identity") as owns_cached:
        assert owns_cached is False
    with ensirna_app._cache_build_lock(
        "prepared", "identity", rebuild=True
    ) as owns_rebuild:
        assert owns_rebuild is True


def test_cache_builder_waiters_share_one_repair_generation(monkeypatch) -> None:
    from threading import Event, Lock, Thread

    values = {}
    values_lock = Lock()
    waiter_observed_owner = Event()

    class FakeDict:
        def get(self, key, default=None):
            with values_lock:
                return values.get(key, default)

        def put(self, key, value, *, skip_if_exists=False):
            with values_lock:
                if skip_if_exists and key in values:
                    if key.endswith(":owner:1"):
                        waiter_observed_owner.set()
                    return False
                values[key] = value
                return True

    monkeypatch.setattr(
        ensirna_app.modal.Dict,
        "from_name",
        lambda *_args, **_kwargs: FakeDict(),
    )
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(ensirna_app.APP_INFO, cache_lock_poll_seconds=0.001),
    )
    first_entered = Event()
    release_first = Event()
    ownership = []

    with ensirna_app._cache_build_lock("prepared", "shared") as owns_initial:
        assert owns_initial is True

    def first_builder():
        with ensirna_app._cache_build_lock("prepared", "shared", rebuild=True) as owns:
            ownership.append(owns)
            first_entered.set()
            release_first.wait(timeout=2)

    def waiting_builder():
        with ensirna_app._cache_build_lock("prepared", "shared", rebuild=True) as owns:
            ownership.append(owns)

    first = Thread(target=first_builder)
    waiter = Thread(target=waiting_builder)
    first.start()
    assert first_entered.wait(timeout=2)
    waiter.start()
    assert waiter_observed_owner.wait(timeout=2)
    release_first.set()
    first.join(timeout=2)
    waiter.join(timeout=2)

    assert ownership == [True, False]
    assert any(key.endswith(":owner:1") for key in values)
    assert not any(key.endswith(":owner:2") for key in values)


def test_preparation_repairs_wholly_missing_completed_publication(
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
    lock_key, lock_values = install_completed_cache_generation(
        monkeypatch, "prepared", cache_key
    )
    plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key=cache_key,
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / "mrna.json"),
        processed_dir=str(layout.outputs_dir / "mrna_processed"),
        candidate_count=1,
        chunk_count=0,
        chunks=[],
        cached=False,
    )

    class FakePrepare:
        def remote(self, **_kwargs):
            return plan

    class FakeFinalize:
        def remote(self, stage_plan):
            return stage_plan

    class FakePreprocess:
        def remote(self, stage_plan, *, preprocess_shard_size):
            assert preprocess_shard_size == ensirna_app.APP_INFO.preprocess_shard_size
            layout.outputs_dir.mkdir(parents=True)
            Path(stage_plan.json_path).write_text('{"siRNA":"m_0"}\n', encoding="utf-8")
            processed_dir = Path(stage_plan.processed_dir)
            processed_dir.mkdir()
            part = processed_dir / "part_0.pkl"
            part.write_bytes(b"processed")
            processed_dir.joinpath("_metainfo").write_text(
                '{"num_entry":1,"file_names":["'
                + str(part)
                + '"],"file_num_entries":[1]}',
                encoding="utf-8",
            )
            ensirna_app._write_prepared_marker(
                layout=layout, plan=stage_plan, json_records=1
            )
            return stage_plan

    monkeypatch.setattr(ensirna_app, "ensirna_prepare_inputs", FakePrepare())
    monkeypatch.setattr(ensirna_app, "ensirna_finalize_prepared_inputs", FakeFinalize())
    monkeypatch.setattr(ensirna_app, "ensirna_preprocess_dataset", FakePreprocess())

    result = ensirna_app.build_ensirna_prepared_inputs.get_raw_f()(fasta)

    assert result.cache_key == cache_key
    assert lock_values[f"{lock_key}:status:1"]["state"] == "complete"
    assert not any(key.endswith(":owner:2") for key in lock_values)


def test_preparation_coordinator_preserves_partial_generation_on_retry(
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
    captured = {}
    plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key=ensirna_app._cache_key_for_fasta(b">m\nAUGCUAGCUAGCUAGCUAGC\n"),
        prepared_dir="/unused",
        json_path="/unused/mrna.json",
        processed_dir="/unused/mrna_processed",
        candidate_count=1,
        chunk_count=0,
        chunks=[],
        cached=False,
    )

    class FakeRemote:
        def __init__(self, result):
            self.result = result

        def remote(self, **kwargs):
            captured["prepare"] = kwargs
            return self.result

    class FakeStage:
        def remote(self, *_args, **kwargs):
            if "preprocess_shard_size" in kwargs:
                captured["preprocess_shard_size"] = kwargs["preprocess_shard_size"]
            return plan

    @contextmanager
    def owned_lock(*_args, **_kwargs):
        yield True

    monkeypatch.setattr(ensirna_app, "_cache_build_lock", owned_lock)
    monkeypatch.setattr(ensirna_app, "ensirna_prepare_inputs", FakeRemote(plan))
    monkeypatch.setattr(ensirna_app, "ensirna_finalize_prepared_inputs", FakeStage())
    monkeypatch.setattr(ensirna_app, "ensirna_preprocess_dataset", FakeStage())

    ensirna_app.build_ensirna_prepared_inputs.get_raw_f()(
        b">m\nAUGCUAGCUAGCUAGCUAGC\n",
        preprocess_shard_size=17,
    )

    assert captured["prepare"] == {
        "mrna_fasta_bytes": b">m\nAUGCUAGCUAGCUAGCUAGC\n",
        "max_prepare_jobs": 4,
        "force_generation": None,
    }
    assert captured["preprocess_shard_size"] == 17


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"prepare_workers": 0}, "prepare_workers must be between"),
        ({"pdb_cores": 0}, "pdb_cores must be between"),
        (
            {"prepare_workers": 3, "pdb_cores": 32},
            "prepare_workers * pdb_cores must not exceed",
        ),
        ({"preprocess_shard_size": 0}, "preprocess_shard_size must be at least 1"),
    ),
)
def test_preparation_coordinator_rejects_invalid_runtime_budget(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        ensirna_app.build_ensirna_prepared_inputs.get_raw_f()(
            b">m\nAUGCUAGCUAGCUAGCUAGC\n",
            **kwargs,
        )


def test_corrupt_published_part_is_invalidated_and_recomputed(
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
    processed_dir = layout.outputs_dir / "mrna_processed"
    processed_dir.mkdir(parents=True)
    json_path = layout.outputs_dir / "mrna.json"
    json_path.write_text('{"siRNA":"m_0"}\n', encoding="utf-8")
    part = processed_dir / "part_0.pkl"
    part.write_bytes(b"original")
    processed_dir.joinpath("_metainfo").write_text(
        '{"num_entry":1,"file_names":["' + str(part) + '"],"file_num_entries":[1]}',
        encoding="utf-8",
    )
    plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key=cache_key,
        prepared_dir=str(layout.run_root),
        json_path=str(json_path),
        processed_dir=str(processed_dir),
        candidate_count=1,
        chunk_count=0,
        chunks=[],
        cached=False,
    )
    ensirna_app._write_prepared_marker(layout=layout, plan=plan, json_records=1)
    part.write_bytes(b"corrupt!")
    captured = {}

    @contextmanager
    def owned_lock(*_args, **kwargs):
        captured["rebuild"] = kwargs["rebuild"]
        yield True

    class FakePrepare:
        def remote(self, **_kwargs):
            assert not processed_dir.exists()
            assert not ensirna_app._prepared_marker_path(layout).exists()
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text('{"siRNA":"m_0"}\n', encoding="utf-8")
            return plan

    class FakeFinalize:
        def remote(self, _plan):
            return _plan

    class FakePreprocess:
        def remote(self, _plan, *, preprocess_shard_size):
            assert preprocess_shard_size == ensirna_app.APP_INFO.preprocess_shard_size
            processed_dir.mkdir(parents=True)
            part.write_bytes(b"recomputed")
            processed_dir.joinpath("_metainfo").write_text(
                '{"num_entry":1,"file_names":["'
                + str(part)
                + '"],"file_num_entries":[1]}',
                encoding="utf-8",
            )
            ensirna_app._write_prepared_marker(layout=layout, plan=plan, json_records=1)
            return plan

    monkeypatch.setattr(ensirna_app, "_cache_build_lock", owned_lock)
    monkeypatch.setattr(ensirna_app, "ensirna_prepare_inputs", FakePrepare())
    monkeypatch.setattr(ensirna_app, "ensirna_finalize_prepared_inputs", FakeFinalize())
    monkeypatch.setattr(ensirna_app, "ensirna_preprocess_dataset", FakePreprocess())

    ensirna_app.build_ensirna_prepared_inputs.get_raw_f()(fasta)

    assert captured["rebuild"] is True
    assert part.read_bytes() == b"recomputed"
    assert (
        ensirna_app._cached_preparation_plan(cache_key=cache_key, layout=layout)
        is not None
    )


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


def test_prepare_inputs_rejects_unsafe_upstream_candidate_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ensirna_dir = tmp_path / "ENsiRNA"
    ensirna_dir.mkdir()
    monkeypatch.setattr(
        ensirna_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(), output_volume_mountpoint=str(tmp_path)
        ),
    )
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(ensirna_app.APP_INFO, ensirna_dir=ensirna_dir),
    )

    def fake_run_command(cmd, **_kwargs):
        Path(cmd[cmd.index("-o") + 1], "mrna.csv").write_text(
            "siRNA,anti seq,sense seq,mRNA_seq,position,efficacy\n"
            "target_0;touch pwned,GC,CG,AUGC,0,0\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)

    with pytest.raises(ValueError, match="candidate ID"):
        ensirna_app.ensirna_prepare_inputs.get_raw_f()(
            mrna_fasta_bytes=b">target\nAUGCUAGCUAGCUAGCUAGC\n",
        )


def test_prepare_inputs_regenerates_unmarked_truncated_csv_and_preserves_pdb_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
        ),
    )
    fasta = b">m\nAUGCUAGCUAGCUAGCUAGC\n"
    cache_key = ensirna_app._cache_key_for_fasta(fasta)
    layout = ensirna_app._layout_for_cache_key(cache_key)
    layout.outputs_dir.mkdir(parents=True)
    layout.outputs_dir.joinpath("mrna.csv").write_text(
        "siRNA,anti seq,sense seq,mRNA_seq,position,efficacy\nm_0,GC,CG,AUGC,0,0\n",
        encoding="utf-8",
    )
    pdb_dir = layout.outputs_dir / "mrna_pdb"
    pdb_dir.mkdir()
    pdb_dir.joinpath("m_0.pdb").write_text("PDB", encoding="utf-8")
    pdb_dir.joinpath("m_1").mkdir()
    chunk_dir = ensirna_app._pdb_prep_dir(layout)
    chunk_dir.mkdir(parents=True)
    chunk_dir.joinpath("chunk_0000.json").write_text(
        '{"siRNA":"m_0","pdb_data_path":"' + str(pdb_dir / "m_0.pdb") + '"}\n',
        encoding="utf-8",
    )

    calls = []

    def fake_run_command(cmd, **_kwargs):
        calls.append(cmd)
        output_dir = Path(cmd[cmd.index("-o") + 1])
        output_dir.joinpath("mrna.csv").write_text(
            "siRNA,anti seq,sense seq,mRNA_seq,position,efficacy\n"
            "m_0,GC,CG,AUGC,0,0\n"
            "m_1,GC,CG,AUGC,1,0\n"
            "m_2,GC,CG,AUGC,2,0\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)

    result = ensirna_app.ensirna_prepare_inputs.get_raw_f()(
        mrna_fasta_bytes=fasta,
        max_prepare_jobs=4,
    )

    assert result.cached is False
    assert len(calls) == 1
    assert result.candidate_count == 3
    assert result.chunk_count == 2
    assert [chunk.chunk_name for chunk in result.chunks] == [
        "chunk_0001",
        "chunk_0002",
    ]
    assert pdb_dir.joinpath("m_0.pdb").exists()
    assert not pdb_dir.joinpath("m_1").exists()
    assert chunk_dir.joinpath("chunk_0000.json").exists()
    chunk_csv_text = "\n".join(
        Path(chunk.csv_path).read_text(encoding="utf-8") for chunk in result.chunks
    )
    assert "m_0,GC,CG,AUGC,0,0" not in chunk_csv_text
    assert "m_1,GC,CG,AUGC,1,0" in chunk_csv_text
    assert "m_2,GC,CG,AUGC,2,0" in chunk_csv_text
    assert ensirna_app._candidate_csv_valid(
        layout=layout,
        cache_key=cache_key,
        input_sha256=ensirna_app._bytes_sha256(
            ensirna_app._sanitize_fasta_for_upstream(fasta)
        ),
    )
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_prepare_inputs_force_generation_isolated_from_normal_cache(
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
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
        ),
    )
    fasta = b">m\nAUGCUAGCUAGCUAGCUAGC\n"
    normal_cache_key = ensirna_app._cache_key_for_fasta(fasta)
    normal_layout = ensirna_app._layout_for_cache_key(normal_cache_key)
    force_generation = "0123456789abcdef0123456789abcdef"
    forced_cache_key = ensirna_app._cache_key_for_fasta(
        fasta, force_generation=force_generation
    )
    old_pdb = normal_layout.outputs_dir / "mrna_pdb" / "m_0.pdb"
    old_pdb.parent.mkdir(parents=True)
    old_pdb.write_text("PDB", encoding="utf-8")
    normal_layout.prep_dir.mkdir(parents=True)
    normal_layout.prep_dir.joinpath("chunk_0007.json").write_text(
        '{"siRNA":"m_0"}\n', encoding="utf-8"
    )

    def fake_run_command(cmd, **kwargs):
        captured["old_pdb_exists_during_get_sirna"] = old_pdb.exists()
        out_dir = Path(cmd[cmd.index("-o") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir.joinpath("mrna.csv").write_text(
            "siRNA,anti seq,sense seq,mRNA_seq,position,efficacy\nm_0,GC,CG,AUGC,0,0\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)

    result = ensirna_app.ensirna_prepare_inputs.get_raw_f()(
        mrna_fasta_bytes=fasta,
        max_prepare_jobs=4,
        force_generation=force_generation,
    )

    assert captured["old_pdb_exists_during_get_sirna"] is True
    assert old_pdb.exists()
    assert normal_layout.prep_dir.joinpath("chunk_0007.json").exists()
    assert result.cache_key == forced_cache_key
    assert result.chunk_count == 1
    assert [chunk.chunk_name for chunk in result.chunks] == ["chunk_0000"]
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_prepare_inputs_repairs_zero_byte_pdb_with_stale_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ensirna_dir = tmp_path / "ENsiRNA"
    ensirna_dir.mkdir()
    monkeypatch.setattr(
        ensirna_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(), output_volume_mountpoint=str(tmp_path)
        ),
    )
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(ensirna_app.APP_INFO, ensirna_dir=ensirna_dir),
    )
    fasta = b">m\nAUGCUAGCUAGCUAGCUAGC\n"
    layout = ensirna_app._layout_for_cache_key(ensirna_app._cache_key_for_fasta(fasta))
    layout.outputs_dir.mkdir(parents=True)
    layout.outputs_dir.joinpath("mrna.csv").write_text(
        "siRNA,anti seq,sense seq,mRNA_seq,position,efficacy\nm_0,GC,CG,AUGC,0,0\n",
        encoding="utf-8",
    )
    seal_candidate_csv(layout, fasta)
    pdb_dir = layout.outputs_dir / "mrna_pdb"
    pdb_dir.mkdir()
    pdb_path = pdb_dir / "m_0.pdb"
    pdb_path.touch()
    chunk_dir = layout.prep_dir / "pdb_chunks"
    chunk_dir.mkdir(parents=True)
    chunk_dir.joinpath("chunk_0000.json").write_text(
        '{"siRNA":"m_0","pdb_data_path":"' + str(pdb_path) + '"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ensirna_app,
        "run_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing candidate CSV should be reused")
        ),
    )

    result = ensirna_app.ensirna_prepare_inputs.get_raw_f()(
        mrna_fasta_bytes=fasta,
        max_prepare_jobs=1,
    )

    assert result.chunk_count == 1
    assert not pdb_path.exists()
    assert "m_0" in Path(result.chunks[0].csv_path).read_text(encoding="utf-8")


def test_prepare_pdb_chunk_runs_rosetta_on_cpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}
    ensirna_dir = tmp_path / "ENsiRNA"
    ensirna_dir.mkdir()
    output_volume = FakeVolume()
    layout = make_test_layout(tmp_path, monkeypatch, output_volume=output_volume)
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(ensirna_app.APP_INFO, ensirna_dir=ensirna_dir),
    )
    chunk_dir = ensirna_app._pdb_prep_dir(layout)
    chunk_dir.mkdir(parents=True)
    chunk_csv = chunk_dir / "chunk_0000.csv"
    chunk_csv.write_text("siRNA\nm_0\n", encoding="utf-8")
    pdb_dir = layout.outputs_dir / "mrna_pdb"
    pdb_dir.mkdir(parents=True)

    def fake_run_command(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        captured["kwargs"] = kwargs
        staged_pdb_dir = Path(cmd[cmd.index("-p") + 1])
        captured["staged_pdb_dir"] = staged_pdb_dir
        assert staged_pdb_dir != pdb_dir
        pdb_path = staged_pdb_dir / "m_0.pdb"
        pdb_path.write_text("PDB", encoding="utf-8")
        Path(cmd[cmd.index("-f") + 1]).with_suffix(".json").write_text(
            '{"siRNA":"m_0","pdb_data_path":"' + str(pdb_path) + '"}\n',
            encoding="utf-8",
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
    assert captured["cmd"][-2:] == ["--num-cores", "3"]
    assert "env" not in captured["kwargs"]
    assert (pdb_dir / "m_0.pdb").read_text(encoding="utf-8") == "PDB"
    assert ensirna_app._json_records(chunk_csv.with_suffix(".json"))[0][
        "pdb_data_path"
    ] == str(pdb_dir / "m_0.pdb")
    assert not captured["staged_pdb_dir"].exists()
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_prepare_pdb_chunk_does_not_publish_interrupted_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = make_test_layout(tmp_path, monkeypatch)
    chunk_dir = ensirna_app._pdb_prep_dir(layout)
    chunk_dir.mkdir(parents=True)
    chunk_csv = chunk_dir / "chunk_0000.csv"
    chunk_csv.write_text("siRNA\ntarget_0\n", encoding="utf-8")
    chunk_json = chunk_dir / "chunk_0000.json"
    pdb_dir = layout.outputs_dir / "mrna_pdb"
    pdb_dir.mkdir(parents=True)

    def fake_run_command(cmd, **_kwargs):
        staged_csv = Path(cmd[cmd.index("-f") + 1])
        staged_pdb_dir = Path(cmd[cmd.index("-p") + 1])
        staged_pdb_dir.joinpath("target_0.pdb").write_bytes(b"partial")
        staged_csv.with_suffix(".json").write_text("{", encoding="utf-8")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)

    with pytest.raises(RuntimeError, match="interrupted"):
        ensirna_app.ensirna_prepare_pdb_chunk.get_raw_f()(
            chunk=ensirna_app.EnsirnaPdbChunkSpec(
                "chunk_0000", str(chunk_csv), str(chunk_json), str(pdb_dir)
            )
        )

    assert not chunk_json.exists()
    assert not pdb_dir.joinpath("target_0.pdb").exists()
    assert not list(chunk_dir.glob(".*.tmp"))


def test_finalize_prepared_inputs_merges_json_only_on_cpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_volume = FakeVolume()
    layout = make_test_layout(tmp_path, monkeypatch, output_volume=output_volume)
    chunk_dir = ensirna_app._pdb_prep_dir(layout)
    chunk_dir.mkdir(parents=True)
    pdb_dir = layout.outputs_dir / "mrna_pdb"
    pdb_dir.mkdir(parents=True)
    pdb_a = pdb_dir / "m_0.pdb"
    pdb_b = pdb_dir / "m_1.pdb"
    pdb_a.write_text("PDB", encoding="utf-8")
    pdb_b.write_text("PDB", encoding="utf-8")
    chunk_a = chunk_dir / "chunk_0000.json"
    chunk_b = chunk_dir / "chunk_0001.json"
    csv_a = chunk_dir / "chunk_0000.csv"
    csv_b = chunk_dir / "chunk_0001.csv"
    csv_a.write_text("siRNA\nm_0\n", encoding="utf-8")
    csv_b.write_text("siRNA\nm_1\n", encoding="utf-8")
    chunk_a.write_text(
        '{"siRNA":"m_0","pdb_data_path":"' + str(pdb_a) + '"}\n',
        encoding="utf-8",
    )
    chunk_b.write_text(
        '{"siRNA":"m_1","pdb_data_path":"' + str(pdb_b) + '"}\n',
        encoding="utf-8",
    )
    plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key=TEST_CACHE_KEY,
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / "mrna.json"),
        processed_dir=str(layout.outputs_dir / "mrna_processed"),
        candidate_count=2,
        chunk_count=2,
        chunks=[
            ensirna_app.EnsirnaPdbChunkSpec(
                "chunk_0000", str(csv_a), str(chunk_a), str(pdb_dir)
            ),
            ensirna_app.EnsirnaPdbChunkSpec(
                "chunk_0001", str(csv_b), str(chunk_b), str(pdb_dir)
            ),
        ],
        cached=False,
    )

    def fake_run_command(*_args, **_kwargs):
        raise AssertionError("CPU finalize should not run dataset preprocessing")

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)
    result = ensirna_app.ensirna_finalize_prepared_inputs.get_raw_f()(plan)

    assert [
        record["siRNA"] for record in ensirna_app._json_records(Path(plan.json_path))
    ] == ["m_0", "m_1"]
    assert result.chunks == []
    assert result.cached is False
    assert not ensirna_app._prepared_marker_path(layout).exists()
    assert not Path(plan.processed_dir).exists()
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_finalize_rejects_json_without_complete_pdbs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = make_test_layout(tmp_path, monkeypatch)
    layout.outputs_dir.mkdir(parents=True)
    chunk_dir = ensirna_app._pdb_prep_dir(layout)
    chunk_dir.mkdir(parents=True)
    pdb_dir = layout.outputs_dir / "mrna_pdb"
    pdb_dir.mkdir()
    layout.outputs_dir.joinpath("mrna.csv").write_text(
        "siRNA,anti seq,sense seq,mRNA_seq,position,efficacy\n"
        "target_0,GC,CG,AUGC,0,0\n",
        encoding="utf-8",
    )
    chunk_csv = chunk_dir / "chunk_0000.csv"
    chunk_csv.write_text("siRNA\ntarget_0\n", encoding="utf-8")
    chunk_json = chunk_dir / "chunk_0000.json"
    chunk_json.write_text(
        '{"siRNA":"target_0","pdb_data_path":"'
        + str(pdb_dir / "target_0.pdb")
        + '"}\n',
        encoding="utf-8",
    )
    plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key=TEST_CACHE_KEY,
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / "mrna.json"),
        processed_dir=str(layout.outputs_dir / "mrna_processed"),
        candidate_count=1,
        chunk_count=1,
        chunks=[
            ensirna_app.EnsirnaPdbChunkSpec(
                "chunk_0000", str(chunk_csv), str(chunk_json), str(pdb_dir)
            )
        ],
        cached=False,
    )

    with pytest.raises(FileNotFoundError, match="PDB artifacts"):
        ensirna_app.ensirna_finalize_prepared_inputs.get_raw_f()(plan)


def test_finalize_prepared_inputs_merges_cached_and_new_json_in_csv_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_volume = FakeVolume()
    layout = make_test_layout(tmp_path, monkeypatch, output_volume=output_volume)
    layout.outputs_dir.mkdir(parents=True)
    chunk_dir = ensirna_app._pdb_prep_dir(layout)
    chunk_dir.mkdir(parents=True)
    pdb_dir = layout.outputs_dir / "mrna_pdb"
    pdb_dir.mkdir()
    pdb_paths = {name: pdb_dir / f"{name}.pdb" for name in ("m_0", "m_1", "m_2")}
    for path in pdb_paths.values():
        path.write_text("PDB", encoding="utf-8")
    layout.outputs_dir.joinpath("mrna.csv").write_text(
        "siRNA,anti seq,sense seq,mRNA_seq,position,efficacy\n"
        "m_0,GC,CG,AUGC,0,0\n"
        "m_1,GC,CG,AUGC,1,0\n"
        "m_2,GC,CG,AUGC,2,0\n",
        encoding="utf-8",
    )
    chunk_dir.joinpath("chunk_0000.json").write_text(
        '{"siRNA":"m_0","pdb_data_path":"' + str(pdb_paths["m_0"]) + '"}\n',
        encoding="utf-8",
    )
    chunk_dir.joinpath("chunk_0000.csv").write_text("siRNA\nm_0\n", encoding="utf-8")
    chunk_a = chunk_dir / "chunk_0001.json"
    chunk_b = chunk_dir / "chunk_0002.json"
    csv_a = chunk_dir / "chunk_0001.csv"
    csv_b = chunk_dir / "chunk_0002.csv"
    csv_a.write_text("siRNA\nm_2\n", encoding="utf-8")
    csv_b.write_text("siRNA\nm_1\n", encoding="utf-8")
    chunk_a.write_text(
        '{"siRNA":"m_2","pdb_data_path":"' + str(pdb_paths["m_2"]) + '"}\n',
        encoding="utf-8",
    )
    chunk_b.write_text(
        '{"siRNA":"m_1","pdb_data_path":"' + str(pdb_paths["m_1"]) + '"}\n',
        encoding="utf-8",
    )
    plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key=TEST_CACHE_KEY,
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / "mrna.json"),
        processed_dir=str(layout.outputs_dir / "mrna_processed"),
        candidate_count=3,
        chunk_count=2,
        chunks=[
            ensirna_app.EnsirnaPdbChunkSpec(
                "chunk_0001", str(csv_a), str(chunk_a), str(pdb_dir)
            ),
            ensirna_app.EnsirnaPdbChunkSpec(
                "chunk_0002", str(csv_b), str(chunk_b), str(pdb_dir)
            ),
        ],
        cached=False,
    )

    def fake_run_command(*_args, **_kwargs):
        raise AssertionError("CPU finalize should not run dataset preprocessing")

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)
    ensirna_app.ensirna_finalize_prepared_inputs.get_raw_f()(plan)

    assert [
        record["siRNA"] for record in ensirna_app._json_records(Path(plan.json_path))
    ] == ["m_0", "m_1", "m_2"]
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_preprocess_dataset_runs_rnafm_on_gpu_and_marks_cache(
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
    layout = make_test_layout(tmp_path, monkeypatch, output_volume=output_volume)
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
            rnafm_cache_path=rnafm_cache,
        ),
    )
    layout.outputs_dir.mkdir(parents=True)
    active_inference = ensirna_app._inference_prep_dir(layout) / "active.tmp"
    active_inference.parent.mkdir(parents=True)
    active_inference.write_bytes(b"active")
    layout.outputs_dir.joinpath("mrna.json").write_text(
        '{"siRNA":"m_0"}\n{"siRNA":"m_1"}\n',
        encoding="utf-8",
    )
    plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key=TEST_CACHE_KEY,
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / "mrna.json"),
        processed_dir=str(layout.outputs_dir / "mrna_processed"),
        candidate_count=2,
        chunk_count=2,
        chunks=[],
        cached=False,
    )

    def fake_run_command(cmd, **kwargs):
        assert not ensirna_app._prepared_marker_path(layout).exists()
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        save_dir = Path(cmd[cmd.index("--save_dir") + 1])
        save_dir.mkdir(parents=True)
        part = save_dir / "part_0.pkl"
        part.write_bytes(b"processed")
        save_dir.joinpath("_metainfo").write_text(
            (
                '{"num_entry":2,"file_names":["'
                + str(part)
                + '"],"file_num_entries":[2]}'
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)
    result = ensirna_app.ensirna_preprocess_dataset.get_raw_f()(plan)

    assert result.chunks == []
    assert result.cached is False
    assert ensirna_app._prepared_marker_path(layout).exists()
    assert '"json_records":2' in ensirna_app._prepared_marker_path(layout).read_text(
        encoding="utf-8"
    )
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
    assert captured["env"] == {ensirna_app.APP_INFO.rnafm_device_env: "cuda"}
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 3
    assert active_inference.read_bytes() == b"active"


def test_preprocess_dataset_checkpoints_independent_rnafm_shards(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_volume = FakeVolume()
    ensirna_dir = tmp_path / "ENsiRNA"
    ensirna_dir.mkdir()
    rnafm_cache = tmp_path / "models" / "RNA-FM_pretrained.pth"
    rnafm_cache.parent.mkdir()
    rnafm_cache.write_bytes(b"weights")
    layout = make_test_layout(tmp_path, monkeypatch, output_volume=output_volume)
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
            rnafm_cache_path=rnafm_cache,
        ),
    )
    layout.outputs_dir.mkdir(parents=True)
    layout.outputs_dir.joinpath("mrna.json").write_text(
        '{"siRNA":"target_0"}\n{"siRNA":"target_1"}\n',
        encoding="utf-8",
    )
    plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key=TEST_CACHE_KEY,
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / "mrna.json"),
        processed_dir=str(layout.outputs_dir / "mrna_processed"),
        candidate_count=2,
        chunk_count=1,
        chunks=[],
        cached=False,
    )
    calls = []

    def fake_run_command(cmd, **_kwargs):
        calls.append(cmd)
        save_dir = Path(cmd[cmd.index("--save_dir") + 1])
        save_dir.mkdir(parents=True)
        part = save_dir / "part_0.pkl"
        part.write_bytes(b"processed")
        save_dir.joinpath("_metainfo").write_text(
            (
                '{"num_entry":1,"file_names":["'
                + str(part)
                + '"],"file_num_entries":[1]}'
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)

    ensirna_app.ensirna_preprocess_dataset.get_raw_f()(plan, preprocess_shard_size=1)

    assert len(calls) == 2
    assert ensirna_app._processed_manifest_valid(Path(plan.processed_dir), 2)
    assert output_volume.commit_count >= 3


def test_preprocess_dataset_recomputes_same_size_corrupted_partial_shard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ensirna_dir = tmp_path / "ENsiRNA"
    ensirna_dir.mkdir()
    rnafm_cache = tmp_path / "models" / "RNA-FM_pretrained.pth"
    rnafm_cache.parent.mkdir()
    rnafm_cache.write_bytes(b"weights")
    layout = make_test_layout(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
            rnafm_cache_path=rnafm_cache,
        ),
    )
    layout.outputs_dir.mkdir(parents=True)
    layout.outputs_dir.joinpath("mrna.json").write_text(
        '{"siRNA":"target_0"}\n{"siRNA":"target_1"}\n',
        encoding="utf-8",
    )
    plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key=TEST_CACHE_KEY,
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / "mrna.json"),
        processed_dir=str(layout.outputs_dir / "mrna_processed"),
        candidate_count=2,
        chunk_count=1,
        chunks=[],
        cached=False,
    )

    def interrupted_run(cmd, **_kwargs):
        input_path = Path(cmd[cmd.index("--dataset") + 1])
        if "target_1" in input_path.read_text(encoding="utf-8"):
            raise RuntimeError("simulated interruption")
        save_dir = Path(cmd[cmd.index("--save_dir") + 1])
        save_dir.mkdir(parents=True)
        part = save_dir / "part_0.pkl"
        part.write_bytes(b"processed")
        save_dir.joinpath("_metainfo").write_text(
            (
                '{"num_entry":1,"file_names":["'
                + str(part)
                + '"],"file_num_entries":[1]}'
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ensirna_app, "run_command", interrupted_run)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        ensirna_app.ensirna_preprocess_dataset.get_raw_f()(
            plan, preprocess_shard_size=1
        )

    first_part = (
        Path(plan.processed_dir) / "shards" / "shard_0000" / "processed" / "part_0.pkl"
    )
    assert first_part.read_bytes() == b"processed"
    assert ensirna_app._processed_shard_marker_path(first_part.parent).is_file()
    first_part.write_bytes(b"corrupted")
    retry_inputs = []

    def retry_run(cmd, **_kwargs):
        input_path = Path(cmd[cmd.index("--dataset") + 1])
        retry_inputs.append(input_path.read_text(encoding="utf-8"))
        save_dir = Path(cmd[cmd.index("--save_dir") + 1])
        save_dir.mkdir(parents=True)
        part = save_dir / "part_0.pkl"
        part.write_bytes(b"recomputed")
        save_dir.joinpath("_metainfo").write_text(
            (
                '{"num_entry":1,"file_names":["'
                + str(part)
                + '"],"file_num_entries":[1]}'
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ensirna_app, "run_command", retry_run)
    ensirna_app.ensirna_preprocess_dataset.get_raw_f()(plan, preprocess_shard_size=1)

    assert len(retry_inputs) == 2
    assert "target_0" in retry_inputs[0]
    assert "target_1" in retry_inputs[1]
    assert first_part.read_bytes() == b"recomputed"
    assert ensirna_app._processed_manifest_valid(Path(plan.processed_dir), 2)


def test_run_ensirna_inference_returns_result_xlsx_bytes(
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
    layout = make_test_layout(tmp_path, monkeypatch, output_volume=output_volume)
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
            checkpoint_dir=checkpoint_dir,
        ),
    )
    processed_dir = layout.outputs_dir / "mrna_processed"
    processed_dir.mkdir(parents=True)
    layout.outputs_dir.joinpath("mrna.json").write_text(
        '{"siRNA":"m_0"}\n', encoding="utf-8"
    )
    part = processed_dir / "part_0.pkl"
    part.write_bytes(b"processed")
    processed_dir.joinpath("_metainfo").write_text(
        ('{"num_entry":1,"file_names":["' + str(part) + '"],"file_num_entries":[1]}'),
        encoding="utf-8",
    )
    write_prepared_marker(
        layout,
        candidate_count=1,
        chunk_count=1,
    )
    lock_rebuilds = []
    run_count = 0

    def fake_run_command(cmd, **kwargs):
        nonlocal run_count
        run_count += 1
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        Path(cmd[cmd.index("--save_dir") + 1], "mrna_result.xlsx").write_bytes(b"xlsx")

    @contextmanager
    def owned_lock(*_args, **kwargs):
        lock_rebuilds.append(kwargs["rebuild"])
        yield True

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)
    monkeypatch.setattr(ensirna_app, "_cache_build_lock", owned_lock)
    result = ensirna_app.run_ensirna_inference.get_raw_f()(
        prepared_dir=str(layout.run_root)
    )

    assert result == b"xlsx"
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
    assert Path(
        captured["cmd"][captured["cmd"].index("--save_dir") + 1]
    ).parent == ensirna_app._inference_prep_dir(layout)
    assert captured["cmd"][captured["cmd"].index("--gpu") + 1] == "0"
    assert captured["cmd"][captured["cmd"].index("--id") + 1] == "mrna"
    assert captured["cwd"] == ensirna_dir
    for filename in ensirna_app.APP_INFO.checkpoint_filenames:
        assert (ensirna_dir / "pkl" / filename).resolve() == checkpoint_dir / filename
    assert output_volume.reload_count == 2
    assert output_volume.commit_count == 1

    (layout.outputs_dir / "mrna_result.xlsx").write_bytes(b"oops")
    repaired = ensirna_app.run_ensirna_inference.get_raw_f()(
        prepared_dir=str(layout.run_root)
    )

    assert repaired == b"xlsx"
    assert lock_rebuilds == [True, True]
    assert run_count == 2


def test_inference_repairs_wholly_missing_completed_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_volume = FakeVolume()
    ensirna_dir = tmp_path / "ENsiRNA"
    checkpoint_dir = tmp_path / "models" / "pkl"
    (ensirna_dir / "pkl").mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    for filename in ensirna_app.APP_INFO.checkpoint_filenames:
        checkpoint_dir.joinpath(filename).write_bytes(b"checkpoint")
    layout = make_test_layout(tmp_path, monkeypatch, output_volume=output_volume)
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
            checkpoint_dir=checkpoint_dir,
        ),
    )
    processed_dir = layout.outputs_dir / "mrna_processed"
    processed_dir.mkdir(parents=True)
    layout.outputs_dir.joinpath("mrna.json").write_text(
        '{"siRNA":"m_0"}\n', encoding="utf-8"
    )
    part = processed_dir / "part_0.pkl"
    part.write_bytes(b"processed")
    processed_dir.joinpath("_metainfo").write_text(
        '{"num_entry":1,"file_names":["' + str(part) + '"],"file_num_entries":[1]}',
        encoding="utf-8",
    )
    write_prepared_marker(layout, candidate_count=1, chunk_count=1)
    lock_key, lock_values = install_completed_cache_generation(
        monkeypatch, "inference", TEST_CACHE_KEY
    )

    def fake_run_command(cmd, **_kwargs):
        Path(cmd[cmd.index("--save_dir") + 1], "mrna_result.xlsx").write_bytes(b"xlsx")

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)

    result = ensirna_app.run_ensirna_inference.get_raw_f()(
        prepared_dir=str(layout.run_root)
    )

    assert result == b"xlsx"
    assert lock_values[f"{lock_key}:status:1"]["state"] == "complete"
    assert not any(key.endswith(":owner:2") for key in lock_values)


def test_submit_ensirna_writes_local_xlsx(tmp_path: Path, monkeypatch) -> None:
    input_fasta = tmp_path / "target.fa"
    input_fasta.write_text(">m\nAUGCUAGCUAGCUAGCUAGC\n", encoding="utf-8")
    captured = {}

    class FakeDownload:
        def remote(self, *, force: bool):
            captured["download_force"] = force

    class FakeRun:
        def remote(self, **kwargs):
            captured["run"] = kwargs
            return b"xlsx"

    monkeypatch.setattr(ensirna_app, "download_ensirna_models", FakeDownload())
    preprocessed_plan = ensirna_app.EnsirnaPreparationPlan(
        cache_key="abc123",
        prepared_dir="/remote/prepared",
        json_path="/remote/outputs/mrna.json",
        processed_dir="/remote/outputs/mrna_processed",
        candidate_count=1,
        chunk_count=1,
        chunks=[],
        cached=False,
    )

    class FakeBuild:
        def remote(self, **kwargs):
            captured["build"] = kwargs
            return preprocessed_plan

    monkeypatch.setattr(ensirna_app, "build_ensirna_prepared_inputs", FakeBuild())
    monkeypatch.setattr(ensirna_app, "run_ensirna_inference", FakeRun())
    raw_f = ensirna_app.submit_ensirna_task.info.raw_f
    assert raw_f is not None

    raw_f(
        mrna_fasta=str(input_fasta),
        out_dir=str(tmp_path),
        run_name="demo",
        prepare_workers=2,
        pdb_cores=3,
        preprocess_shard_size=17,
    )

    assert captured["download_force"] is False
    assert captured["build"] == {
        "mrna_fasta_bytes": b">m\nAUGCUAGCUAGCUAGCUAGC\n",
        "prepare_workers": 2,
        "pdb_cores": 3,
        "preprocess_shard_size": 17,
        "force_generation": None,
    }
    assert captured["run"] == {"prepared_dir": "/remote/prepared", "force": False}
    assert (tmp_path / "demo.xlsx").read_bytes() == b"xlsx"
