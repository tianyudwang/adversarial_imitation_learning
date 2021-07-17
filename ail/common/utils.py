import random
import dataclasses
from time import sleep
from itertools import zip_longest
from typing import Tuple, Dict, Any, Iterable

import numpy as np
import torch as th


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def countdown(t_sec) -> None:
    while t_sec:
        mins, secs = divmod(t_sec, 60)
        time_format = f"{mins: 02d}:{secs: 02d}"
        print(time_format, end="\r")
        sleep(1)
        t_sec -= 1
    print("Done!!")


def get_stats(x: np.ndarray) -> Tuple[np.ndarray, ...]:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return x.mean(), x.std(), x.min(), x.max()  # noqa


def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)  # noqa
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def dataclass_quick_asdict(dataclass_instance) -> Dict[str, Any]:
    """
    Extract dataclass to items using `dataclasses.fields` + dict comprehension.
    This is a quick alternative to `dataclasses.asdict`, which expensively and
    undocumentedly deep-copies every numpy array value.
    See https://stackoverflow.com/a/52229565/1091722.
    """
    obj = dataclass_instance
    d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return d


def zip_strict(*iterables: Iterable) -> Iterable:
    """
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.
    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo