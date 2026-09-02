"""Optional five-minute cache over Modal workspace billing reports."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import modal


@dataclass(frozen=True, slots=True)
class CostGroup:
    """One billing label and its exact decimal cost."""

    name: str
    cost: Decimal


@dataclass(frozen=True, slots=True)
class BillingSummary:
    """Workspace, Environment, and stable Tool-tag attribution."""

    total: Decimal
    environments: tuple[CostGroup, ...]
    tools: tuple[CostGroup, ...]
    other: Decimal


class BillingService:
    """Fetch optional billing data without affecting service readiness."""

    def __init__(self, workspace: Any | None = None) -> None:
        """Accept an optional test double without resolving Modal at startup."""
        self.workspace = workspace
        self._cache: dict[
            tuple[datetime, datetime, str], tuple[float, BillingSummary]
        ] = {}
        self._lock = asyncio.Lock()

    async def report(
        self,
        *,
        start: datetime,
        end: datetime,
        resolution: str,
        refresh: bool = False,
    ) -> BillingSummary:
        """Return one tagged report, caching identical requests for five minutes."""
        key = (start, end, resolution)
        now = time.monotonic()
        cached = self._cache.get(key)
        if not refresh and cached is not None and now - cached[0] < 300:
            return cached[1]
        async with self._lock:
            cached = self._cache.get(key)
            if not refresh and cached is not None and now - cached[0] < 300:
                return cached[1]
            workspace = self.workspace or modal.Workspace.from_context()
            rows = await workspace.billing.report.aio(
                start=start,
                end=end,
                resolution=resolution,
                tag_names=["biomodals_tool"],
            )
            summary = _summarize(rows)
            self._cache[key] = (time.monotonic(), summary)
            return summary


def _summarize(rows: list[Any]) -> BillingSummary:
    total = Decimal(0)
    environments: dict[str, Decimal] = {}
    tools: dict[str, Decimal] = {}
    other = Decimal(0)
    for row in rows:
        cost = Decimal(row.cost)
        total += cost
        environments[row.environment_name] = (
            environments.get(row.environment_name, Decimal(0)) + cost
        )
        tool = row.tags.get("biomodals_tool")
        if tool in {"gromacs", "alphafold3"}:
            tools[tool] = tools.get(tool, Decimal(0)) + cost
        else:
            other += cost
    return BillingSummary(
        total=total,
        environments=tuple(
            CostGroup(name, cost) for name, cost in sorted(environments.items())
        ),
        tools=tuple(CostGroup(name, cost) for name, cost in sorted(tools.items())),
        other=other,
    )
