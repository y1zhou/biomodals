"""Tests for PPIFlow coordinator helpers."""

# ruff: noqa: D103

import pytest

from biomodals.workflow.ppiflow import coordinators


def test_candidate_concurrency_from_config_defaults_and_validates() -> None:
    assert coordinators.candidate_concurrency_from_config({}) == 4
    assert (
        coordinators.candidate_concurrency_from_config(
            {"candidate_concurrency": 2},
            {"candidate_concurrency": 8},
        )
        == 2
    )
    with pytest.raises(ValueError, match="at least 1"):
        coordinators.candidate_concurrency_from_config({"candidate_concurrency": 0})
