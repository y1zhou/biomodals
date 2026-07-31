"""Workload-owned immutable output claim chaining."""

from __future__ import annotations

from hashlib import sha256
from typing import Any


def acquire_output_claim(
    claim_store: Any,
    *,
    claim_key: str,
    owner: str,
    replace_owner: str | None = None,
) -> None:
    """Atomically elect one owner, or its explicit successor, to publish output."""
    if not claim_key or not owner:
        raise ValueError("Output claim key and owner cannot be empty")
    root_key = f"{claim_key}:root"
    if claim_store.put(root_key, owner, skip_if_exists=True):
        return
    current_owner = _current_owner(claim_store, claim_key)
    if current_owner == owner:
        return
    if replace_owner is None or current_owner != replace_owner:
        raise RuntimeError("Output is already claimed by another Provider Call")
    successor_key = _successor_key(claim_key, current_owner)
    if claim_store.put(successor_key, owner, skip_if_exists=True):
        return
    if _current_owner(claim_store, claim_key) != owner:
        raise RuntimeError("Output is already claimed by another Provider Call")


def _current_owner(claim_store: Any, claim_key: str) -> str:
    current = claim_store.get(f"{claim_key}:root", None)
    if not isinstance(current, str) or not current:
        raise RuntimeError("Output claim has invalid ownership")
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        successor = claim_store.get(_successor_key(claim_key, current), None)
        if successor is None:
            return current
        if not isinstance(successor, str) or not successor:
            raise RuntimeError("Output claim has invalid ownership")
        current = successor
    raise RuntimeError("Output claim contains an ownership cycle")


def _successor_key(claim_key: str, owner: str) -> str:
    return f"{claim_key}:successor:{sha256(owner.encode()).hexdigest()}"
