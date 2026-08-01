"""Workload-owned immutable output claim chaining."""

from __future__ import annotations

from hashlib import sha256
from typing import Any


def register_output_claim_successor(
    claim_store: Any,
    *,
    owner: str,
    predecessor: str,
) -> None:
    """Record immutable Run lineage even when a successor only reads cache."""
    if not owner or not predecessor or owner == predecessor:
        raise ValueError("Output claim successor lineage is invalid")
    key = _predecessor_key(owner)
    if claim_store.put(key, predecessor, skip_if_exists=True):
        return
    if claim_store.get(key, None) != predecessor:
        raise RuntimeError("Output claim successor lineage is immutable")


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
    if replace_owner is None or (
        current_owner != replace_owner
        and not _replacement_ancestor(
            claim_store,
            owner=owner,
            declared_predecessor=replace_owner,
            current_owner=current_owner,
        )
    ):
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


def _predecessor_key(owner: str) -> str:
    return f"biomodals:output-claim-predecessor:{sha256(owner.encode()).hexdigest()}"


def _replacement_ancestor(
    claim_store: Any,
    *,
    owner: str,
    declared_predecessor: str,
    current_owner: str,
) -> bool:
    predecessor = claim_store.get(_predecessor_key(owner), None)
    if predecessor != declared_predecessor:
        return False
    visited = {owner}
    while isinstance(predecessor, str) and predecessor not in visited:
        if predecessor == current_owner:
            return True
        visited.add(predecessor)
        predecessor = claim_store.get(_predecessor_key(predecessor), None)
    return False
