from abc import ABC, abstractmethod
from typing import Optional, Sequence, Any, Dict
from enum import Enum, auto

import torch as th
from torch import nn
import torch.nn.functional as F

from ail.netwrok.value import StateFunction, StateActionFunction
from ail.common.type_alias import Activation


class Arch(Enum):
    """Arch types of Discriminator"""
    s = auto()
    sa = auto()
    ss = auto()
    sas = auto()


class DiscrimNet(nn.Module, ABC):
    """
    Abstract base class for discriminator, used in AIRL and GAIL.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: Optional[int] = None,
        hidden_units: Sequence[int] = (128, 128),
        hidden_activation: Activation = nn.ReLU(inplace=True),
        gamma: Optional[float] = None,
        disc_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(DiscrimNet, self).__init__()
        if disc_kwargs is None:
            disc_kwargs = {}

        # Net input
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

        # Discount factor
        self.gamma = gamma

        # Regularization
        self.spectral_norm = disc_kwargs.get("spectral_norm", False)
        self.dropout = disc_kwargs.get(
            "dropout", False
        )  # TODO: apply drop out between hidden layers

        # Init Discriminator
        self.init_model = disc_kwargs.get("init_model", True)
        if self.init_model:
            self._init_model(disc_kwargs)

    def _init_model(self, disc_kwargs):
        disc_type = disc_kwargs.get("disc_type", "")
        if disc_type == Arch.s:
            self.hidden_units_r = self.disc_kwargs.get(
                "hidden_units_r", self.hidden_units
            )
            self.hidden_units_v = self.disc_kwargs.get(
                "hidden_units_v", self.hidden_units
            )

            self.g = StateFunction(
                self.state_dim,
                self.hidden_units_r,
                self.hidden_activation,
                self.spectral_norm,
            )
            self.h = StateFunction(
                self.state_dim,
                self.hidden_units_v,
                self.hidden_activation,
                self.spectral_norm,
            )

        elif disc_type == Arch.sa:
            self.f = StateActionFunction(
                self.state_dim,
                self.action_dim,
                self.hidden_units,
                self.hidden_activation,
                self.spectral_norm,
            )

        elif disc_type == Arch.ss:
            raise NotImplementedError()
        elif disc_type == Arch.sas:
            raise NotImplementedError()
        else:
            raise NotImplementedError(
                f"Type {self.disc_type} is not supported or arch not provide in dist_kwargs"
            )

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def calculate_reward(self, *args, **kwargs):
        raise NotImplementedError()

    def _calculate_reward(self, state: th.Tensor, **kwargs):
        with th.no_grad():
            return -F.logsigmoid(-self.forward(state, **kwargs))


class GAILDiscrim(DiscrimNet):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Sequence[int],
        hidden_activation: Activation,
        disc_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if disc_kwargs is None:
            disc_kwargs = {}
        super(GAILDiscrim, self).__init__(
            state_dim, action_dim, hidden_units, hidden_activation, None, disc_kwargs
        )

    def forward(self, state, action):
        # naming `f` to keep consistent with base DiscrimNet
        return self.f(th.cat([state, action], dim=-1))

    def calculate_reward(self, state, action):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with th.no_grad():
            return -F.logsigmoid(-self.forward(state, action))


class AIRLStateDiscrim(DiscrimNet):
    """
    Discriminator used in AIRL with disentangled reward.
    f_{θ,φ} (s, a, s') = g_θ (s, a) + \gamma h_φ (s') − h_φ (s)
    """

    def __init__(
        self,
        state_dim: int,
        gamma: float,
        hidden_units: Sequence[int],
        hidden_activation: Activation,
        disc_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if disc_kwargs is None:
            disc_kwargs = {"disc_type", Arch.s}

        super(AIRLStateDiscrim, self).__init__(
            state_dim, None, hidden_units, hidden_activation, gamma, disc_kwargs
        )

    def f(self, state: th.Tensor, done: th.FloatTensor, next_state: th.Tensor):
        """
        f(s, a, s' ) = g_θ (s) + \gamma h_φ (s') − h_φ (s)

        f: recover to the advantage
        g: state-only reward function approximator
        h: shaping term
        """
        r_s = self.g(state)
        v_s = self.h(state)
        next_vs = self.h(next_state)
        # reshape (1-done) from (n,) to (n,1) to prevent shape mismatch
        return r_s + self.gamma * (1 - done).view(-1, 1) * next_vs - v_s

    def forward(self, state, done, next_state, log_pi=None, **kwargs):
        """
        Policy Objective
        \hat{r}_t = log[D_θ(s,a)] - log[1-D_θ(s,a)]
        = log[exp{f_θ} /(exp{f_θ} + \pi)] - log[\pi / (exp{f_θ} + \pi)]
        = f_θ (s,a) - log \pi (a|s)
        """
        # TODO: verify this in paper
        if log_pi is not None:
            # Discriminator's output is sigmoid(f - log_pi).
            # reshape log_pi to prevent size mismatch
            return self.f(state, done, next_state) - log_pi.view(-1, 1)
        else:
            return self.f(state, done, next_state)

    def calculate_reward(
        self, state, done, next_state, log_pi: Optional[th.Tensor] = None
    ):
        """
        Calculate GAN reward (can pass all data at once)
        """
        # logits = self.forward(state, done, log_pi, next_state)
        # return -F.logsigmoid(-logits)
        kwargs = {
            "state": state,
            "done": done,
            "next_state": next_state,
            "log_pi": log_pi,
        }
        return super()._calculate_reward(state, **kwargs)

    """
    The objective of the discriminator is to minimize cross-entropy loss between expert demonstrations
    and generated samples:
        L(θ) = \sum_{t=0}{T} -E_D [log {D_θ (s_t , a_t )}] − E_πt [log{1 − Dθ (s_t , a_t )}]
    
    DISCRIMINATOR Objective
    => d L(θ) 
        =   \sum_{t=0}{T} -E_D [f_θ (s_t , a_t )] 
            − E_{\mu} [exp{f_θ (s_t , a_t )} / \mu_t (s_t , a_t )}]
    
    Policy Objective
        \hat{r}_t = log[D_θ(s,a)] - log[1-D_θ(s,a)]
        = log[exp{f_θ} /(exp{f_θ} + \pi)] - log[\pi / (exp{f_θ} + \pi)]
        = f_θ (s,a) - log \pi (a|s)        
    """  # noqa


class AIRLStateActionDiscrim(DiscrimNet):
    """
    Discriminator used in AIRL with entangled reward.
    As in the trajectory-centric case,
        f* (s, a) = log π∗ (a|s) = A∗ (s, a)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Sequence[int],
        hidden_activation,
        gamma: float,
        disc_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if disc_kwargs is None:
            disc_kwargs = {"disc_type", Arch.sa}
        super(AIRLStateActionDiscrim, self).__init__(
            state_dim, action_dim, hidden_units, hidden_activation, gamma, disc_kwargs
        )

    def forward(self, obs, act, log_pi=None, **kwargs):
        if log_pi is not None:
            # Discriminator's output is sigmoid(f - log_pi).
            # reshape log_pi to prevent size mismatch
            return self.f(th.cat([obs, act], dim=-1)) - log_pi.view(-1, 1)
        else:
            return self.f(th.cat([obs, act], dim=-1))

    def calculate_reward(
        self, state: th.Tensor, action: th.Tensor, log_pi: Optional[th.Tensor] = None
    ):
        # logits = self.forward(state, action, log_pi)
        # return -F.logsigmoid(-logits)
        kwargs = {
            "state": state,
            "action": action,
            "log_pi": log_pi,
        }
        return super()._calculate_reward(state, **kwargs)
