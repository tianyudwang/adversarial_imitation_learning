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
