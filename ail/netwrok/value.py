from abc import ABC, abstractmethod
from typing import Sequence, Union, Optional

import torch as th
from torch import nn

from ail.common.pytorch_util import build_mlp


class BaseValue(nn.Module, ABC):
    """
    Basic class of a general Value or Q function
    :param state_dim: State dimensions of the environment
    :param action_dim: Action dimensions of the environment
    """

    def __init__(self, state_dim: int, action_dim: Optional[int] = None):
        super(BaseValue, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None

    @abstractmethod
    def forward(self, *args, **kwargs) -> th.Tensor:
        """
        Output of Value Network
        """
        raise NotImplementedError()

    @abstractmethod
    def get_value(self, *args, **kwargs) -> th.Tensor:
        """
        Squeeze Output of Value Network
        """
        raise NotImplementedError()


class StateFunction(BaseValue):
    """
    Basic implementation of a general state function (value function)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_units: Sequence[int] = (64, 64),
        activation: Union[str, nn.Module] = nn.ReLU(inplace=True),
        use_spectral_norm=False,
        **kwargs
    ):
        super().__init__(obs_dim)
        self.net = build_mlp(
            [obs_dim] + list(hidden_units) + [1],
            activation,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, state: th.Tensor):
        """self.net()"""
        # TODO: Critical to ensure v has right shape.
        return self.net(state)

    def get_value(self, state: th.Tensor) -> th.Tensor:
        return self.forward(state).squeeze(-1)


class StateActionFunction(BaseValue):
    """
    Basic implementation of a general state-action function (Q function)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_units: Sequence[int] = (64, 64),
        activation: Union[str, nn.Module] = nn.ReLU(inplace=True),
        use_spectral_norm=False,
        **kwargs
    ):
        super().__init__(obs_dim, act_dim)
        self.net = build_mlp(
            [obs_dim + act_dim] + list(hidden_units) + [1],
            activation,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, state, action):
        # TODO: Critical to ensure q has right shape.
        return self.net(th.cat([state, action], dim=-1))

    def get_value(self, state, action):
        return self.net(th.cat([state, action], dim=-1)).squeeze(-1)


def mlp_value(
    state_dim: int,
    action_dim: int,
    val_type: str,
    value_layers: Sequence[int],
    activation: str = "relu",
    **kwargs
):
    if val_type in  ["V", "v", "Vs"]:
        return StateFunction(state_dim, value_layers, activation, **kwargs)
    elif val_type in ["Qsa", "Q"]:
        return StateActionFunction(
            state_dim, action_dim, value_layers, activation, **kwargs
        )
    else:
        raise ValueError
