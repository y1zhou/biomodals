"""Deterministic workflow DAG hashing helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import orjson
from pydantic import BaseModel

from biomodals.workflow.core.builder import WorkflowDefinition
from biomodals.workflow.core.nodes import WorkflowNode


def dag_hash(definition: WorkflowDefinition) -> str:
    """Return a deterministic hash for semantic workflow DAG identity."""
    payload = {
        "name": definition.name,
        "nodes": {
            node_id: node_hash_payload(spec.node)
            | {
                "inputs": {
                    input_name: selector.model_dump(mode="json")
                    for input_name, selector in sorted(spec.inputs.items())
                },
                "control_dependencies": sorted(spec.control_dependencies),
                "dependencies": sorted(definition.dependencies[node_id]),
            }
            for node_id, spec in sorted(definition.nodes.items())
        },
    }
    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(encoded).hexdigest()


def node_hash_payload(node: WorkflowNode) -> dict[str, object]:
    """Return the semantic hash payload for one workflow node."""
    payload: dict[str, object] = {
        "class": f"{node.__class__.__module__}.{node.__class__.__qualname__}",
        "execution_policy": node.execution_policy.value,
    }
    if is_dataclass(node):
        payload["dataclass"] = stable_json_value(node)
    return payload


def stable_json_value(value: object) -> object:
    """Normalize a node config value into deterministic JSON-compatible data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", round_trip=True)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {
            "bytes_sha256": hashlib.sha256(value).hexdigest(),
            "size_bytes": len(value),
        }
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: stable_json_value(getattr(value, field.name))
            for field in fields(value)
            if field.metadata.get("dag_hash") is not False
        }
    if isinstance(value, Mapping):
        return {
            str(key): stable_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [stable_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        stable_items = [stable_json_value(item) for item in value]
        return sorted(
            stable_items,
            key=lambda item: orjson.dumps(item, option=orjson.OPT_SORT_KEYS),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Unsupported DAG hash value type: {type(value).__module__}."
        f"{type(value).__qualname__}"
    )
