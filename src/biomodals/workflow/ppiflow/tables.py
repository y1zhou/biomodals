"""Pure table helpers for the PPIFlow workflow."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path


def candidate_key(file_name: str) -> str:
    """Return the original structure stem from a collision-safe artifact name."""
    return Path(file_name).stem.rsplit("__", 1)[-1].lower()


def row_passes_filters(
    row: Mapping[str, object], filters: Mapping[str, object]
) -> bool:
    """Return whether a score-table row satisfies all configured filters."""
    comparisons = {
        ">": lambda value, threshold: value > threshold,
        ">=": lambda value, threshold: value >= threshold,
        "<": lambda value, threshold: value < threshold,
        "<=": lambda value, threshold: value <= threshold,
        "==": lambda value, threshold: value == threshold,
        "!=": lambda value, threshold: value != threshold,
    }
    for metric, condition in filters.items():
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid or missing filter metric {metric!r}") from exc

        clauses: list[tuple[str, float]] = []
        if isinstance(condition, Mapping):
            clauses = [
                (str(op), float(threshold)) for op, threshold in condition.items()
            ]
        elif isinstance(condition, str):
            for raw_clause in condition.split(","):
                match = re.fullmatch(
                    r"\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*",
                    raw_clause,
                )
                if match is None:
                    raise ValueError(f"Invalid filter clause: {raw_clause!r}")
                clauses.append((match.group(1), float(match.group(2))))
        else:
            clauses = [(">=", float(condition))]

        for op, threshold in clauses:
            comparison = comparisons.get(op)
            if comparison is None:
                raise ValueError(f"Unsupported filter operator: {op}")
            if not comparison(value, threshold):
                return False
    return True
