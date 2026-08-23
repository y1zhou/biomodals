"""Modal billing attribution contracts."""

# ruff: noqa: D103

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from biomodals.service.billing import _summarize


def test_billing_uses_stable_tool_tags_and_keeps_other_usage() -> None:
    rows = [
        SimpleNamespace(
            cost=Decimal("1.25"),
            environment_name="main",
            tags={"biomodals_tool": "alphafold3"},
            interval_start=datetime(2026, 8, 1, tzinfo=UTC),
        ),
        SimpleNamespace(
            cost=Decimal("0.75"),
            environment_name="main",
            tags={},
            interval_start=datetime(2026, 8, 1, tzinfo=UTC),
        ),
    ]

    summary = _summarize(rows)

    assert summary.total == Decimal("2.00")
    assert summary.tools[0].name == "alphafold3"
    assert summary.other == Decimal("0.75")
