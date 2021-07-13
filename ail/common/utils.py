import random
from datetime import timedelta
from time import time
from typing import Tuple

import numpy as np
import torch as th


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def duration(start_time) -> str:
    return str(timedelta(seconds=int(time() - start_time)))


def get_statistics(x: np.ndarray) -> Tuple[np.ndarray, ...]:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return x.mean(), x.std(), x.min(), x.max()  # noqa


def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)  # noqa
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def asarray_shape2(x):
    return np.asarray(x, dtype=np.float32).reshape(1, -1)
