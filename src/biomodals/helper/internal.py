"""Internal utility scripts for other helpers."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from time import time
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


# Note: this syntax is only supported in Python 3.12+. If we want to support Python 3.10+, we can switch to using `typing_extensions` and `typing.ParamSpec` instead of `typing.P` and `typing.R`.
# def timed_function[**P, R](f: Callable[P, R]) -> Callable[P, R]:
def timed_function(f: Callable[P, R]) -> Callable[P, R]:  # noqa: UP047
    """Decorator to time a function and print its execution time.

    Credit: https://stackoverflow.com/a/27737385
    """

    @wraps(f)
    def wrap(*args: P.args, **kwargs: P.kwargs) -> R:
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        f_name = getattr(f, "__name__", type(f).__name__)
        print(f"func:{f_name!r}[{args!r}, {kwargs!r}] took: {te - ts:.3f} seconds")
        return result

    return wrap
