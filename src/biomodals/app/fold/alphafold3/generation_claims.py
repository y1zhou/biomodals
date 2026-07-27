"""Append-only generation claims for resumable AlphaFold 3 work.

Claims coordinate writers; they are never completion evidence. Callers must
validate their stage-specific marker before reusing a publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Protocol, cast

from biomodals.app.fold.alphafold3.sharding import utc_now

_TERMINAL_STATUSES = frozenset({"complete", "failed", "abandoned"})


class ClaimStore(Protocol):
    """Minimal insert-only key-value interface used by generation claims."""

    def put(
        self,
        key: str,
        value: object,
        *,
        skip_if_exists: bool = False,
    ) -> bool:
        """Store a value and report whether an insert-only write succeeded."""
        ...

    def get(self, key: str, default: object = None) -> object:
        """Return a stored value or the supplied default."""
        ...


@dataclass(frozen=True, slots=True)
class GenerationClaim:
    """One elected writer generation."""

    scope_key: str
    generation_id: str
    owner: dict[str, object]


class ActiveGenerationError(RuntimeError):
    """Raised when another non-stale generation owns a claim."""

    def __init__(self, scope_key: str, owner: dict[str, object]) -> None:
        """Record the conflicting scope and current owner."""
        self.scope_key = scope_key
        self.owner = owner
        generation_id = owner["generation_id"]
        super().__init__(
            f"Claim {scope_key!r} is owned by active generation {generation_id!r}"
        )


class LostGenerationError(RuntimeError):
    """Raised when a superseded writer attempts to publish."""


def _root_key(scope_key: str) -> str:
    return f"claim:{scope_key}:root"


def _successor_key(scope_key: str, generation_id: str) -> str:
    return f"claim:{scope_key}:after:{generation_id}"


def _status_key(scope_key: str, generation_id: str) -> str:
    return f"status:{scope_key}:{generation_id}"


def _validate_scope_key(scope_key: str) -> str:
    if not isinstance(scope_key, str) or not scope_key:
        raise ValueError("scope_key must be a non-empty string")
    return scope_key


def _validate_generation_id(generation_id: str) -> str:
    if not isinstance(generation_id, str) or not generation_id or ":" in generation_id:
        raise ValueError("generation_id must be a non-empty colon-free string")
    return generation_id


def _validate_owner(scope_key: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("scope_key") != scope_key:
        raise RuntimeError(f"Claim {scope_key!r} has an invalid owner")
    generation_id = value.get("generation_id")
    _validate_generation_id(cast(str, generation_id))
    started_at = value.get("started_at_epoch_seconds")
    maximum_age = value.get("maximum_age_seconds")
    if (
        isinstance(started_at, bool)
        or not isinstance(started_at, int | float)
        or isinstance(maximum_age, bool)
        or not isinstance(maximum_age, int | float)
        or maximum_age <= 0
    ):
        raise RuntimeError(f"Claim {scope_key!r} has invalid timing metadata")
    identity = value.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"Claim {scope_key!r} has an invalid identity")
    return cast(dict[str, object], value)


def latest_generation_owner(
    claims: ClaimStore,
    scope_key: str,
) -> dict[str, object] | None:
    """Follow append-only successors to the current generation owner."""
    selected_scope = _validate_scope_key(scope_key)
    current = claims.get(_root_key(selected_scope), None)
    if current is None:
        return None
    seen: set[str] = set()
    while True:
        owner = _validate_owner(selected_scope, current)
        generation_id = cast(str, owner["generation_id"])
        if generation_id in seen:
            raise RuntimeError(f"Claim {selected_scope!r} contains a cycle")
        seen.add(generation_id)
        successor = claims.get(
            _successor_key(selected_scope, generation_id),
            None,
        )
        if successor is None:
            return owner
        current = successor


def generation_status(
    claims: ClaimStore,
    scope_key: str,
    generation_id: str,
) -> dict[str, object] | None:
    """Return a validated terminal status for one generation."""
    selected_scope = _validate_scope_key(scope_key)
    selected_generation = _validate_generation_id(generation_id)
    value = claims.get(_status_key(selected_scope, selected_generation), None)
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("status") not in _TERMINAL_STATUSES:
        raise RuntimeError(
            f"Claim {selected_scope!r} generation {selected_generation!r} "
            "has an invalid terminal status"
        )
    return cast(dict[str, object], value)


def acquire_generation_claim(
    claims: ClaimStore,
    *,
    scope_key: str,
    generation_id: str,
    identity: dict[str, object],
    container_id: str,
    maximum_age_seconds: int | float,
    now_epoch_seconds: int | float | None = None,
    now_text: str | None = None,
) -> GenerationClaim:
    """Elect a writer after fencing a terminal or conservatively stale owner."""
    selected_scope = _validate_scope_key(scope_key)
    selected_generation = _validate_generation_id(generation_id)
    if not isinstance(identity, dict):
        raise TypeError("identity must be a dictionary")
    if not isinstance(container_id, str) or not container_id:
        raise ValueError("container_id must be a non-empty string")
    if (
        isinstance(maximum_age_seconds, bool)
        or not isinstance(maximum_age_seconds, int | float)
        or maximum_age_seconds <= 0
    ):
        raise ValueError("maximum_age_seconds must be positive")
    observed_epoch = time() if now_epoch_seconds is None else now_epoch_seconds
    if isinstance(observed_epoch, bool) or not isinstance(observed_epoch, int | float):
        raise TypeError("now_epoch_seconds must be numeric")
    observed_text = utc_now() if now_text is None else now_text
    if not isinstance(observed_text, str) or not observed_text:
        raise ValueError("now_text must be a non-empty string")

    owner: dict[str, object] = {
        "scope_key": selected_scope,
        "generation_id": selected_generation,
        "identity": identity,
        "container_id": container_id,
        "started_at": observed_text,
        "started_at_epoch_seconds": observed_epoch,
        "maximum_age_seconds": maximum_age_seconds,
    }
    if claims.put(_root_key(selected_scope), owner, skip_if_exists=True):
        return GenerationClaim(selected_scope, selected_generation, owner)

    while True:
        predecessor = latest_generation_owner(claims, selected_scope)
        if predecessor is None:
            raise RuntimeError(f"Claim {selected_scope!r} root disappeared")
        predecessor_generation = cast(str, predecessor["generation_id"])
        predecessor_status = generation_status(
            claims,
            selected_scope,
            predecessor_generation,
        )
        if predecessor_status is None:
            started_at = cast(
                int | float,
                predecessor["started_at_epoch_seconds"],
            )
            maximum_age = cast(
                int | float,
                predecessor["maximum_age_seconds"],
            )
            age_seconds = float(observed_epoch) - float(started_at)
            if age_seconds <= float(maximum_age):
                raise ActiveGenerationError(selected_scope, predecessor)
            claims.put(
                _status_key(selected_scope, predecessor_generation),
                {
                    "status": "abandoned",
                    "abandoned_at": observed_text,
                    "age_seconds": age_seconds,
                },
                skip_if_exists=True,
            )
            predecessor_status = generation_status(
                claims,
                selected_scope,
                predecessor_generation,
            )
            if predecessor_status is None:
                raise RuntimeError(
                    f"Claim {selected_scope!r} stale owner was not fenced"
                )

        successor = owner | {
            "predecessor_generation_id": predecessor_generation,
            "predecessor_status": predecessor_status["status"],
        }
        if claims.put(
            _successor_key(selected_scope, predecessor_generation),
            successor,
            skip_if_exists=True,
        ):
            return GenerationClaim(
                selected_scope,
                selected_generation,
                successor,
            )


def assert_generation_current(
    claims: ClaimStore,
    claim: GenerationClaim,
) -> None:
    """Fail closed unless ``claim`` remains the live latest generation."""
    owner = latest_generation_owner(claims, claim.scope_key)
    if owner is None or owner.get("generation_id") != claim.generation_id:
        raise LostGenerationError(
            f"Generation {claim.generation_id!r} no longer owns "
            f"claim {claim.scope_key!r}"
        )
    if generation_status(claims, claim.scope_key, claim.generation_id) is not None:
        raise LostGenerationError(
            f"Generation {claim.generation_id!r} is already terminal"
        )


def finish_generation_claim(
    claims: ClaimStore,
    claim: GenerationClaim,
    *,
    status: str,
    detail: dict[str, object],
    now_text: str | None = None,
) -> None:
    """Append a terminal status without deleting claim history."""
    if status not in {"complete", "failed"}:
        raise ValueError("status must be 'complete' or 'failed'")
    if not isinstance(detail, dict):
        raise TypeError("detail must be a dictionary")
    observed_text = utc_now() if now_text is None else now_text
    created = claims.put(
        _status_key(claim.scope_key, claim.generation_id),
        {
            "status": status,
            "finished_at": observed_text,
            **detail,
        },
        skip_if_exists=True,
    )
    if created:
        return
    existing = generation_status(
        claims,
        claim.scope_key,
        claim.generation_id,
    )
    if existing is None or existing.get("status") != status:
        raise LostGenerationError(
            f"Generation {claim.generation_id!r} already has a different "
            "terminal status"
        )
