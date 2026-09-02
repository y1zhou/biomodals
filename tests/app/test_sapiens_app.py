"""Contracts for the Sapiens humanization app."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID

import pandas as pd
import polars as pl
import pytest

from biomodals.app.design import sapiens_app
from biomodals.app.design.sapiens_execution import (
    HUMANIZE_NODE,
    SapiensExecutionRequest,
    _SapiensHumanizeNode,
)
from biomodals.execution import RunStatus
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
    call = _SapiensHumanizeNode(request).prepare_remote(None)  # type: ignore[arg-type]
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
