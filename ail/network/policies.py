import torch as th
from torch import nn

from ail.common.pytorch_util import build_mlp
from ail.common.math import reparameterize, evaluate_lop_pi


class StateIndependentPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_units=(64, 64),
        hidden_activation=nn.Tanh(),
    ):
        super().__init__()

        self.net = build_mlp(
            sizes=[obs_dim] + list(hidden_units) + [act_dim],
            activation=hidden_activation,
        )
        # TODO: allow log_std init
        self.log_stds = nn.Parameter(th.zeros(1, act_dim))

    def forward(self, states):
        return th.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_units=(256, 256),
        hidden_activation=nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.net = build_mlp(
            sizes=[obs_dim] + list(hidden_units) + [2 * act_dim],
            activation=hidden_activation,
        )

    def forward(self, states: th.Tensor):
        return th.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states: th.Tensor):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp(-20, 2))

    def evaluate_log_pi(self, states: th.Tensor, actions: th.Tensor):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return evaluate_lop_pi(means, log_stds, actions)