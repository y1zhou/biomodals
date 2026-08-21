"""Invoke a Modal local entrypoint as a plain local CLI client."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import types
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints
from uuid import UUID


def invoke_local_entrypoint(
    *,
    module_name: str,
    entrypoint_name: str,
    flags: Sequence[str],
    overrides: Mapping[str, Any],
    program_name: str,
    environment_name: str | None = None,
) -> Any:
    """Parse CLI flags and call one Modal local entrypoint in this process."""
    with _modal_environment(environment_name):
        module = importlib.import_module(module_name)
        local_entrypoint = getattr(module, entrypoint_name, None)
        callback = getattr(getattr(local_entrypoint, "info", None), "raw_f", None)
        if not callable(callback):
            raise ValueError(
                f"{module_name}::{entrypoint_name} is not a Modal local entrypoint"
            )
        unknown_overrides = set(overrides).difference(
            inspect.signature(callback).parameters
        )
        if unknown_overrides:
            names = ", ".join(sorted(unknown_overrides))
            raise ValueError(
                f"Entrypoint does not accept injected parameter(s): {names}"
            )
        kwargs = _parse_entrypoint_flags(
            callback,
            flags,
            hidden_parameters=frozenset(overrides),
            program_name=program_name,
        )
        return callback(**kwargs, **overrides)


@contextmanager
def _modal_environment(environment_name: str | None):
    """Temporarily bind lazily hydrated Modal handles to one Environment."""
    if environment_name is None:
        yield
        return
    previous = os.environ.get("MODAL_ENVIRONMENT")
    os.environ["MODAL_ENVIRONMENT"] = environment_name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MODAL_ENVIRONMENT", None)
        else:
            os.environ["MODAL_ENVIRONMENT"] = previous


def _parse_entrypoint_flags(
    callback: Callable[..., Any],
    flags: Sequence[str],
    *,
    hidden_parameters: frozenset[str],
    program_name: str,
) -> dict[str, Any]:
    """Parse Modal-style scalar entrypoint flags without starting a Modal app."""
    signature = inspect.signature(callback)
    type_hints = get_type_hints(callback)
    parser = argparse.ArgumentParser(prog=program_name)
    required_positionals: list[tuple[str, Callable[[str], Any]]] = []

    for name, parameter in signature.parameters.items():
        if name in hidden_parameters:
            continue
        annotation = type_hints.get(name, parameter.annotation)
        converter, choices = _argument_type(annotation)
        option = f"--{name.replace('_', '-')}"
        if converter is bool:
            if parameter.default is inspect.Parameter.empty:
                raise TypeError(
                    f"Boolean entrypoint parameter '{name}' needs a default"
                )
            parser.add_argument(
                option,
                action=argparse.BooleanOptionalAction,
                default=parameter.default,
                dest=name,
            )
            continue

        default = (
            None if parameter.default is inspect.Parameter.empty else parameter.default
        )
        parser.add_argument(
            option,
            choices=choices,
            default=default,
            dest=name,
            type=converter,
        )
        if parameter.default is inspect.Parameter.empty:
            required_positionals.append((name, converter))

    parser.add_argument("_required_positionals", nargs="*")
    parsed = vars(parser.parse_args(list(flags)))
    positional_values = parsed.pop("_required_positionals")
    missing = [item for item in required_positionals if parsed[item[0]] is None]
    if len(positional_values) > len(missing):
        parser.error("too many positional arguments")
    for raw_value, (name, converter) in zip(
        positional_values,
        missing,
        strict=False,
    ):
        parsed[name] = converter(raw_value)
    missing_names = [
        name for name, _converter in required_positionals if parsed[name] is None
    ]
    if missing_names:
        parser.error(
            "the following arguments are required: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing_names)
        )
    return parsed


def _argument_type(
    annotation: Any,
) -> tuple[Callable[[str], Any], tuple[Any, ...] | None]:
    """Return an argparse converter and optional choices for one scalar type."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        members = tuple(
            member for member in get_args(annotation) if member is not type(None)
        )
        if len(members) != 1:
            raise TypeError(f"Unsupported entrypoint parameter type: {annotation}")
        return _argument_type(members[0])
    if origin is Literal:
        choices = get_args(annotation)
        if not choices:
            raise TypeError("Literal entrypoint parameter has no choices")
        converter = type(choices[0])
        if not all(type(choice) is converter for choice in choices):
            raise TypeError(f"Unsupported mixed Literal parameter type: {annotation}")
        return converter, choices
    if annotation in (str, int, float, bool, Path, UUID):
        return annotation, None
    raise TypeError(f"Unsupported entrypoint parameter type: {annotation}")
