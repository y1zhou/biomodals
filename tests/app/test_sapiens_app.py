"""Contracts for the Sapiens humanization app."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pandas as pd
import polars as pl
import pytest

from biomodals.app.design.sapiens import app as sapiens_app
from biomodals.app.design.sapiens.execution import (
    HUMANIZE_NODE,
    SapiensExecutionRequest,
    _SapiensHumanizeNode,
)
from biomodals.execution import RunStatus
from biomodals.execution.nodes import NodeRunContext
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE

VALID_CSV = (
    b"id,vh,vl\n"
    b"pair-1,QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYGMNWVRQAPGQGLEWMG,"
    b"DIQMTQSPSSLSASVGDRVTITCRASQSI\n"
)


def _request() -> SapiensExecutionRequest:
    return SapiensExecutionRequest(
        run_name="example",
        csv_bytes=VALID_CSV,
        iterations=1,
        numbering_scheme="kabat",
        cdr_definition="kabat",
        mutate_cdrs=False,
        app_version="sapiens-source",
        model_vh_revision="vh-revision",
        model_vl_revision="vl-revision",
        tokenizer_revision="tokenizer-revision",
        runtime_identity="runtime-identity",
    )


def test_modal_group_tag_does_not_depend_on_source_path() -> None:
    assert sapiens_app.CONF.tags == {"group": "design"}


def test_parse_sapiens_csv_accepts_complete_unique_pairs() -> None:
    frame = sapiens_app.parse_sapiens_csv(VALID_CSV)

    assert frame.schema == {"id": pl.String, "vh": pl.String, "vl": pl.String}
    assert frame.to_dicts()[0]["id"] == "pair-1"


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (b"id,vh\na,AAAA\n", "exactly these columns"),
        (b"id,vh,vl\na,AAAA,CCCC\na,DDDD,EEEE\n", "duplicate id"),
        (b"id,vh,vl\na,AAAA,\n", "vl must be non-empty"),
        (b"id,vh,vl\na,AAAa,CCCC\n", "non-canonical or lowercase"),
    ),
)
def test_parse_sapiens_csv_rejects_invalid_batches(
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sapiens_app.parse_sapiens_csv(content)


def test_execution_request_roundtrips_and_plans_one_cpu_node() -> None:
    request = _request()

    assert SapiensExecutionRequest.from_bytes(request.to_bytes()) == request
    assert request.execution_plan.nodes[0].node_key == HUMANIZE_NODE
    call = _SapiensHumanizeNode(request).prepare_remote(cast(NodeRunContext, None))
    assert call.function_name == "sapiens_humanize"
    assert call.uses_gpu is False
    assert call.kwargs["csv_bytes"] == VALID_CSV


def test_execution_request_rejects_boolean_iterations() -> None:
    with pytest.raises(ValueError, match="iterations"):
        replace(_request(), iterations=True)


class FakePosition:
    def __init__(self, index: int) -> None:
        self.index = index

    def format(self) -> str:
        return f"H{self.index + 1}"

    def get_region(self) -> str:
        return "CDR1" if self.index == 1 else "FR1"


class FakeChain:
    def __init__(
        self,
        sequence: str,
        *,
        name: str,
        scheme: str,
        cdr_definition: str,
    ) -> None:
        del scheme, cdr_definition
        self.seq = sequence
        self.name = name
        self.chain_type = "H"
        self.positions = [FakePosition(index) for index in range(len(sequence))]

    def clone(self, replace_seq: str | None = None):
        return FakeChain(
            replace_seq or self.seq,
            name=self.name,
            scheme="kabat",
            cdr_definition="kabat",
        )

    def graft_cdrs_onto(self, other, *, backmutate_vernier: bool):
        assert backmutate_vernier is False
        sequence = list(other.seq)
        sequence[1] = self.seq[1]
        return self.clone("".join(sequence))

    def is_heavy_chain(self) -> bool:
        return True

    def is_light_chain(self) -> bool:
        return False


def _scores(argmax_sequence: str) -> pd.DataFrame:
    rows = []
    for argmax in argmax_sequence:
        row = {amino_acid: 0.0 for amino_acid in sapiens_app.AMINO_ACIDS}
        row[argmax] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_humanize_chain_preserves_parental_cdr_each_iteration(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "abnumber", SimpleNamespace(Chain=FakeChain))

    def fake_predict(chain: FakeChain, model_root) -> pd.DataFrame:
        del model_root
        return _scores("CCCC" if chain.seq == "AAAA" else "DDDD")

    monkeypatch.setattr(sapiens_app, "_predict_scores", fake_predict)

    chain, score_frames, mutations = sapiens_app._humanize_chain(
        identifier="pair",
        chain_label="vh",
        sequence="AAAA",
        iterations=2,
        numbering_scheme="kabat",
        cdr_definition="kabat",
        mutate_cdrs=False,
        model_root=sapiens_app.MODEL_ROOT,
    )

    assert chain.seq == "DADD"
    assert [frame["endpoint"][0] for frame in score_frames] == ["input", "final"]
    assert {row["iteration"] for row in mutations} == {1, 2}
    assert all(row["numbered_position"] != "H2" for row in mutations)


def test_result_bundle_writes_typed_mutation_history(monkeypatch) -> None:
    expected_schema = {
        "id": pl.String,
        "chain": pl.String,
        "iteration": pl.Int64,
        "sequence_index": pl.Int64,
        "numbered_position": pl.String,
        "region": pl.String,
        "from_aa": pl.String,
        "to_aa": pl.String,
    }

    def fake_package_outputs(result_dir: Path, *, num_threads: int) -> bytes:
        assert (result_dir / "humanized.fasta").read_text().splitlines()[::2] == [
            ">pair_1_VH",
            ">pair_1_VL",
        ]
        assert num_threads == 2
        assert (
            pl.read_parquet(result_dir / "mutation_history.parquet").schema
            == expected_schema
        )
        assert not (result_dir / "mutation_history.csv").exists()
        assert b'"schema_version": 2' in (result_dir / "manifest.json").read_bytes()
        return b"archive"

    monkeypatch.setattr(sapiens_app, "package_outputs", fake_package_outputs)
    archive = sapiens_app._write_result_bundle(
        run_name="demo",
        input_frame=pl.DataFrame([{"id": "pair 1", "vh": "AAAA", "vl": "CCCC"}]),
        humanized_rows=[{"id": "pair 1", "vh": "AAAA", "vl": "CCCC"}],
        score_frames=[pl.DataFrame({"id": ["pair 1"]})],
        mutation_rows=[],
        iterations=1,
        numbering_scheme="kabat",
        cdr_definition="kabat",
        mutate_cdrs=False,
    )

    assert archive == b"archive"


def test_workflow_function_returns_inline_archive(monkeypatch) -> None:
    monkeypatch.setattr(
        sapiens_app,
        "_run_sapiens_humanization",
        lambda **kwargs: (b"archive", 3, 7, 1.25),
    )

    result = sapiens_app.sapiens_humanize.get_raw_f()(
        run_name="../example",
        csv_bytes=VALID_CSV,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert result.metrics == {
        "pair_count": 3,
        "mutation_count": 7,
        "iterations": 1,
        "elapsed_seconds": 1.25,
    }
    assert result.outputs[0].storage == InlineBytes(
        data=b"archive",
        filename="example_sapiens.tar.zst",
        media_type=ZSTD_MEDIA_TYPE,
    )


def test_scoring_only_keeps_sequences_and_averages_observed_not_argmax(monkeypatch):
    seen = []

    def number_chain(**kwargs):
        assert kwargs["numbering_scheme"] == kwargs["cdr_definition"] == "imgt"
        return FakeChain(
            kwargs["sequence"],
            name=kwargs["identifier"],
            scheme="imgt",
            cdr_definition="imgt",
        )

    def predict(chain, model_root):
        seen.append(chain.seq)
        # Deliberately non-normalized over the canonical subset: omitted special
        # token mass must not be redistributed by the wrapper.
        rows = []
        for index, residue in enumerate(chain.seq):
            row = {amino_acid: 0.0 for amino_acid in sapiens_app.AMINO_ACIDS}
            row[residue] = 0.1 * (index + 1)
            row["W"] = 0.5
            rows.append(row)
        return pd.DataFrame(rows)

    monkeypatch.setattr(sapiens_app, "_number_chain", number_chain)
    monkeypatch.setattr(sapiens_app, "_predict_scores", predict)
    monkeypatch.setattr(
        sapiens_app,
        "_humanize_chain",
        lambda **kwargs: pytest.fail("Scoring must not humanize"),
    )
    frame, summary, profile = sapiens_app._score_sapiens_pairs(b"id,vh,vl\na,ACD,EF\n")
    assert seen == ["ACD", "EF"]
    assert frame["vh"].to_list() == ["ACD"]
    assert summary["vh_mean_probability"][0] == pytest.approx(0.2)
    assert summary["vl_mean_probability"][0] == pytest.approx(0.15)
    assert profile["observed_aa"].to_list() == list("ACDEF")
    assert profile["argmax_aa"].to_list() == ["W"] * 5
    assert profile["endpoint"].unique().to_list() == ["candidate"]


def test_scoring_operation_returns_summary_and_detail_archive(monkeypatch):
    from biomodals.workflow.humanization import scoring as humanization

    frame = pl.DataFrame({"id": ["a"], "vh": ["ACD"], "vl": ["EFG"]})
    summary = pl.DataFrame({
        "id": ["a"],
        "vh_mean_probability": [0.2],
        "vl_mean_probability": [0.3],
    })
    detail = pl.DataFrame({"id": ["a"], "observed_aa": ["A"]})
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None)
    )
    monkeypatch.setattr(
        sapiens_app, "_score_sapiens_pairs", lambda content: (frame, summary, detail)
    )

    def package(root, *, num_threads):
        assert pl.read_csv(root / "summary.csv").equals(summary)
        assert pl.read_parquet(root / "residue_scores.parquet").equals(detail)
        assert not (root / "humanized.csv").exists()
        assert b'"operation":"sapiens_score"' in (root / "manifest.json").read_bytes()
        return b"scores"

    monkeypatch.setattr(humanization, "package_outputs", package)
    result = sapiens_app.sapiens_score.get_raw_f()(csv_bytes=VALID_CSV)
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].name == "sapiens_scores"
    assert result.outputs[0].storage.data == b"scores"


def test_local_entrypoint_stages_kernel_run_and_writes_archive(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "pairs.csv"
    input_path.write_bytes(VALID_CSV)
    output_volume = object()
    monkeypatch.setattr(
        sapiens_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume,
            repo_commit_hash="source",
            version="1.1.0",
            name="Sapiens",
        ),
    )
    calls = {}

    def fake_stage(volume, execution_run_id, request):
        calls["stage"] = (volume, execution_run_id, request)

    overview = SimpleNamespace(run=SimpleNamespace(execution_run_id=UUID(int=1)))

    def fake_submit(volume, **kwargs):
        calls["submit"] = (volume, kwargs)
        return overview

    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="sapiens_humanization",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(
                    data=b"archive",
                    filename="pairs_sapiens.tar.zst",
                    media_type=ZSTD_MEDIA_TYPE,
                ),
            )
        ],
    )
    monkeypatch.setattr(sapiens_app, "stage_execution_request", fake_stage)
    monkeypatch.setattr(sapiens_app, "submit_staged_execution_run", fake_submit)
    monkeypatch.setattr(sapiens_app, "result_from_overview", lambda *args: result)

    raw_f = sapiens_app.submit_sapiens_task.info.raw_f
    assert raw_f is not None
    raw_f(input_csv=str(input_path), output_dir=str(tmp_path))

    assert calls["stage"][0] is output_volume
    assert calls["stage"][2].max_active_gpu_provider_calls == 0
    assert calls["submit"][1]["accepted_statuses"] == (RunStatus.SUCCEEDED,)
    assert (tmp_path / "pairs_sapiens.tar.zst").read_bytes() == b"archive"

    calls.clear()
    with pytest.raises(FileExistsError):
        raw_f(input_csv=str(input_path), output_dir=str(tmp_path))
    assert calls == {}
    assert (tmp_path / "pairs_sapiens.tar.zst").read_bytes() == b"archive"
