import random

import numpy as np
import torch as th


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)  # noqa
    return (length, shape) if np.isscalar(shape) else (length, *shape)