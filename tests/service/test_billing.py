"""Modal billing attribution contracts."""

# ruff: noqa: D103

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from biomodals.service.billing import _summarize
from biomodals.service.tools import TOOLS


@pytest.mark.parametrize("tool", [tool.key for tool in TOOLS])
def test_billing_uses_stable_tool_tags_and_keeps_other_usage(tool) -> None:
    rows = [
        SimpleNamespace(
            cost=Decimal("1.25"),
            environment_name="main",
            tags={"biomodals_tool": tool},
            interval_start=datetime(2026, 8, 1, tzinfo=UTC),
        ),
        SimpleNamespace(
            cost=Decimal("0.75"),
            environment_name="main",
            tags={},
            interval_start=datetime(2026, 8, 1, tzinfo=UTC),
        ),
        SimpleNamespace(
            cost=Decimal("0.50"),
            environment_name="main",
            tags={"biomodals_tool": "unknown"},
        ),
    ]

    summary = _summarize(rows)

    assert summary.total == Decimal("2.50")
    assert summary.tools[0].name == tool
    assert summary.tools[0].cost == Decimal("1.25")
    assert summary.other == Decimal("1.25")
    assert summary.environments[0].cost == summary.total
