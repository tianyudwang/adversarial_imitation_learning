import random
from time import sleep
from typing import Tuple

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


def asarray_shape2d(x):
    return np.asarray(x, dtype=np.float32).reshape(1, -1)
