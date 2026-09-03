"""Contracts for the p-AbNatiV2 humanization app."""

# ruff: noqa: D101,D102,D103

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from biomodals.app.design.pabnativ2 import app as pabnativ2_app
from biomodals.app.design.pabnativ2 import patches as pabnativ2_patches
from biomodals.app.design.pabnativ2.execution import (
    HUMANIZE_NODE,
    PAbNatiV2ExecutionRequest,
    _PAbNatiV2HumanizeNode,
)
from biomodals.app.design.pabnativ2.models import (
    IDENTITY,
    PAIRED_MODEL,
    STRUCTURE_MODEL_ARCHIVE,
)
from biomodals.app.design.pabnativ2.patches import _replace_once
from biomodals.execution.nodes import NodeRunContext
from biomodals.schema import AppRunResult, AppRunStatus

VALID_CSV = (
    b"id,vh,vl\n"
    b"pair-1,QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYGMNWVRQAPGQGLEWMG,"
    b"DIQMTQSPSSLSASVGDRVTITCRASQSI\n"
)


def _request() -> PAbNatiV2ExecutionRequest:
    return PAbNatiV2ExecutionRequest(
        run_name="example",
        csv_bytes=VALID_CSV,
        mutate_cdrs=False,
        fixed_vh_positions="27,28",
        fixed_vl_positions="",
        residue_score_threshold=0.98,
        rasa_threshold=0.15,
        max_relative_pairing_score_decrease=0.1,
        forbidden_residues="C,M",
        seed=7,
        source_commit=IDENTITY.abnativ_commit,
        paired_model_md5=PAIRED_MODEL.md5_hex,
        structure_archive_md5=STRUCTURE_MODEL_ARCHIVE.md5_hex,
        runtime_identity="runtime",
    )


def test_modal_group_tag_does_not_depend_on_source_path() -> None:
    assert pabnativ2_app.CONF.tags == {"group": "design"}


def test_parse_pabnativ2_csv_accepts_complete_unique_pairs() -> None:
    frame = pabnativ2_app.parse_pabnativ2_csv(VALID_CSV)

    assert frame.schema == {"id": pl.String, "vh": pl.String, "vl": pl.String}
    assert frame.item(0, "id") == "pair-1"


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (b"id,vh\na,AAAA\n", "exactly these columns"),
        (b"id,vh,vl\na,AAAA,CCCC\na,DDDD,EEEE\n", "duplicate id"),
        (b"id,vh,vl\na,AAAA,\n", "vl must be non-empty"),
        (b"id,vh,vl\na,AAAa,CCCC\n", "non-canonical or lowercase"),
    ),
)
def test_parse_pabnativ2_csv_rejects_invalid_batches(
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pabnativ2_app.parse_pabnativ2_csv(content)


def test_controls_normalize_positions_and_forbidden_residues() -> None:
    parameters = pabnativ2_app._validate_parameters(
        mutate_cdrs=True,
        fixed_vh_positions="27, 149",
        fixed_vl_positions="",
        residue_score_threshold=0.98,
        rasa_threshold=0.15,
        max_relative_pairing_score_decrease=0.1,
        forbidden_residues="C,M",
        seed=0,
    )

    assert parameters.fixed_vh_positions == (27, 149)
    assert parameters.fixed_vl_positions == ()
    assert parameters.forbidden_residues == ("C", "M")
    assert 27 not in pabnativ2_app._allowed_positions(True, (27,))
    assert 27 not in pabnativ2_app._allowed_positions(False, ())


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"fixed_vh_positions": "27,27"}, "duplicate"),
        ({"fixed_vl_positions": "150"}, "between 1 and 149"),
        ({"forbidden_residues": "C,C"}, "duplicates"),
        ({"forbidden_residues": "c"}, "uppercase canonical"),
        ({"residue_score_threshold": float("nan")}, "finite"),
        ({"seed": True}, "seed"),
    ),
)
def test_controls_reject_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "mutate_cdrs": False,
        "fixed_vh_positions": "",
        "fixed_vl_positions": "",
        "residue_score_threshold": 0.98,
        "rasa_threshold": 0.15,
        "max_relative_pairing_score_decrease": 0.1,
        "forbidden_residues": "C,M",
        "seed": 0,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        pabnativ2_app._validate_parameters(**values)  # type: ignore[arg-type]


def test_pair_seed_is_order_independent_and_input_specific() -> None:
    first = pabnativ2_app._pair_seed(4, "id", "AAAA", "CCCC")

    assert first == pabnativ2_app._pair_seed(4, "id", "AAAA", "CCCC")
    assert first != pabnativ2_app._pair_seed(4, "other", "AAAA", "CCCC")


def test_execution_request_roundtrips_and_plans_one_cpu_node() -> None:
    request = _request()

    assert PAbNatiV2ExecutionRequest.from_bytes(request.to_bytes()) == request
    assert request.execution_plan.nodes[0].node_key == HUMANIZE_NODE
    call = _PAbNatiV2HumanizeNode(request).prepare_remote(cast(NodeRunContext, None))
    assert call.function_name == "pabnativ2_humanize"
    assert call.uses_gpu is True
    assert call.kwargs["seed"] == 7


def test_execution_request_rejects_boolean_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        replace(_request(), seed=True)


def test_cpu_worker_returns_common_app_result(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = AppRunResult(status=AppRunStatus.SUCCEEDED)
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> AppRunResult:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(pabnativ2_app, "_run_pabnativ2", fake_run)

    result = pabnativ2_app.pabnativ2_humanize.get_raw_f()("-unsafe/name", VALID_CSV)

    assert result is expected
    assert captured["run_name"] == "unsafe_name"


def test_compatibility_patches_are_guarded_and_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.py"
    path.write_text("return data[ranges]\n", encoding="utf-8")

    _replace_once(path, "return data[ranges]", "return data[tuple(ranges)]")
    _replace_once(path, "return data[ranges]", "return data[tuple(ranges)]")

    assert path.read_text(encoding="utf-8") == "return data[tuple(ranges)]\n"


def test_missing_pssm_is_hash_verified_and_installed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"pinned pssm"
    monkeypatch.setattr(
        pabnativ2_patches,
        "PSSM_SHA256",
        {"VH2_pssm.npy": sha256(content).hexdigest()},
    )
    monkeypatch.setattr(pabnativ2_patches, "_package_root", lambda _package: tmp_path)
    monkeypatch.setattr(
        pabnativ2_patches,
        "urlopen",
        lambda _url, timeout: BytesIO(content),  # noqa: ARG005
    )

    pabnativ2_patches.install_missing_abnativ_pssms()

    assert (tmp_path / "humanisation/pssms/VH2_pssm.npy").read_bytes() == content
