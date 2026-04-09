"""Internal utility scripts for other helpers."""

from collections.abc import Callable
from functools import wraps
from time import time


def timed_function[**P, R](f: Callable[P, R]) -> Callable[P, R]:
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
