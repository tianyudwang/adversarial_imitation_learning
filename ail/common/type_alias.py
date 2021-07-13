from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import Adam, AdamW


# Gym type
GymEnv = gym.Env
GymWrapper = gym.Wrapper
GymSpace = gym.spaces.Space
GymDict = gym.spaces.Dict
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]

# tensor type
TensorDict = Dict[Union[str, int], th.Tensor]
Activation = Union[str, nn.Module]

# ----------------------------------------------------------------
# string to object naming conventions

_str_to_activation = {
    "relu": nn.ReLU(),
    "relu_inplace": nn.ReLU(inplace=True),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}

OPT = {
    "adam": Adam,
    "adamw": AdamW,
}


# ----------------------------------------------------------------
# Buffer shape and dtype
@dataclass(frozen=True, eq=False)
class Extra_shape:
    advs: Tuple[int, ...] = (1,)
    rets: Tuple[int, ...] = (1,)
    vals: Tuple[int, ...] = (1,)
    log_pis: Tuple[int, ...] = (1,)


@dataclass(frozen=True, eq=False)
class Extra_dtypes:
    advs: np.dtype = np.float32
    rets: np.dtype = np.float32
    vals: np.dtype = np.float32
    log_pis: np.dtype = np.float32


extra_shapes = Extra_shape()
extra_dtypes = Extra_dtypes()
