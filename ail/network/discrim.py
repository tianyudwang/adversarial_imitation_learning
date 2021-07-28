from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence
from enum import Enum, auto

import torch as th
from torch import nn
import torch.nn.functional as F

from ail.network.value import StateFunction, StateActionFunction
from ail.common.type_alias import Activation
from ail.common.pytorch_util import count_vars


class ArchType(Enum):
    """Arch types of Discriminator"""

    s = auto()
    sa = auto()
    ss = auto()
    sas = auto()


class RewardType(Enum):
    airl = "airl"
    AIRL = "airl"
    gail = "gail"
    GAIL = "gail"


class ChoiceType(Enum):
    logit = "logit"
    logsigmoid = "logsigmoid"
    log_sigmoid = "log_sigmoid"
    softplus = "softplus"
    soft_plus = "soft_plus"


class DiscrimNet(nn.Module, ABC):
    """
    Abstract base class for discriminator, used in AIRL and GAIL.

    D = sigmoid(f)
    D(s, a) = sigmoid(f(s, a))
    D(s, a) = exp{f(s,a)} / (exp{f(s,a) + \pi(a|s)}
    where f is a discriminator logit (a learnable function represented as MLP)

    Choice of reward function:
    • r(s, a) = − ln(1 − D) = softplus(h) (used in the original GAIL paper),
    • r(s, a) = ln D − ln(1 − D) = h (introduced in AIRL).
    • r(s, a) = ln D = −softplus(−h),       # ! Currently not working.
    • r(s, a) = −h exp(h) (introduced in FAIRL) # ! Not Implemented.
    # TODO: clip rewards with the absolute values higher than max reward magnitude
    # * The GAIL paper uses the inverse convention in which
    # * D denotes the probability as being classified as non-expert.

    The objective of the discriminator is to
    minimize cross-entropy loss
    between expert demonstrations and generated samples:

    L = \sum[ -E_{D} log(D) - E_{\pi} log(1 - D)]

    Write the negative loss to turn the minimization problem into maximization:
    -L = \sum[ -E_{D} log(D) + E_{\pi} log(1 - D)]

    """

    def __init__(
        self,
        disc_type: ArchType,
        state_dim: int,
        action_dim: Optional[int] = None,
        hidden_units: Sequence[int] = (128, 128),
        hidden_activation: Activation = nn.ReLU(inplace=True),
        init_model=True,
        **disc_kwargs,
    ):
        super().__init__()
        if disc_kwargs is None:
            disc_kwargs = {}

        # Net input
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

        # Regularization
        self.spectral_norm = disc_kwargs.get("spectral_norm", False)
        self.dropout = disc_kwargs.get(
            "dropout", False
        )  # TODO: apply drop out between hidden layers

        # Init Discriminator
        if init_model:
            self._init_model(disc_type)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def _init_model(self, disc_type: ArchType) -> None:
        if disc_type == ArchType.s:
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

        elif disc_type == ArchType.sa:
            self.f = StateActionFunction(
                self.state_dim,
                self.action_dim,
                self.hidden_units,
                self.hidden_activation,
                self.spectral_norm,
            )

        elif disc_type == ArchType.ss:
            raise NotImplementedError(f"disc_type: {disc_type} not implemented.")

        elif disc_type == ArchType.sas:
            raise NotImplementedError(f"disc_type: {disc_type} not implemented.")

        else:
            raise NotImplementedError(
                f"Type {self.disc_type} is not supported or arch not provide in dist_kwargs."
            )

    @abstractmethod
    def forward(self, *args, **kwargs) -> th.Tensor:
        """Output logits of discriminator."""
        raise NotImplementedError()

    @abstractmethod
    def calculate_rewards(self, *args, **kwargs) -> th.Tensor:
        """Calculate learning rewards based on choice of reward formulation."""
        raise NotImplementedError()

    def reward_fn(self, rew_type: str, choice: str) -> Callable[[th.Tensor], th.Tensor]:
        """
        The learning rewards formulation.
        (GAIL):r(s, a) = − ln(1 − D) = softplus(h)
        (AIRL): r(s, a) = ln D − ln(1 − D) = h

        Paper:"What Matters for Adversarial Imitation Learning?" Appendix C.2.
        See: https://arxiv.org/abs/2106.00672

        :param rew_type: airl or gail
        :param choice: logsigmoid, sofplus, logit
        Note logit only available in airl and returns itself without any transformation.

        LHS equation and RHS equation are mathmatically identical why implement both?
        Because Pytorch's logsigmoid and softplus behaves differently in the same reward function.
        Might due to the threshold value in softplus.
        Refer to https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
        """
        rew_types = {"gail", "airl"}
        choices = {"logsigmoid", "softplus", "logit"}

        rew_type = RewardType[rew_type.lower()].value
        choice = ChoiceType[choice.lower()].value

        if rew_type == "gail":
            # * (1)  − ln(1 − D) = softplus(h)
            if choice == "logsigmoid":
                return self.gail_logsigmoid
            elif choice == "softplus":
                return self.gail_softplus
            elif choice == "logit":
                raise ValueError(f"Choice logit not supported for Gail.")
            else:
                raise ValueError(
                    f"Choice {choices} not supported with rew_type gail. "
                    f"Valid choices are {choices}."
                )

        elif rew_type == "airl":
            # * (2)  ln D − ln(1 − D) = h = −softplus(-h) + softplus(h)
            if choice == "logsigmoid":
                return self.airl_logsigmoid
            elif choice == "softplus":
                return self.airl_softplus
            elif choice == "logit":
                return self.airl_logit
            else:
                raise ValueError(
                    f"Choice {choices} not supported. Valid choices are {choices}."
                )

        else:
            raise ValueError(
                f"Reward type {rew_type} not supported. "
                f"Valid rew_types: {rew_types}"
            )

    @staticmethod
    def gail_logsigmoid(x: th.Tensor) -> th.Tensor:
        """
        (GAIL):r(s, a) = − ln(1 − D)
        :param x: logits
        """
        return -F.logsigmoid(-x)

    @staticmethod
    def gail_softplus(x: th.Tensor) -> th.Tensor:
        """
        (GAIL):r(s, a) = softplus(h)
        :param x: logits
        """
        return F.softplus(x)

    @staticmethod
    def airl_logsigmoid(x: th.Tensor) -> th.Tensor:
        """
        (AIRL): r(s, a) = ln D − ln(1 − D)
        :param x: logits
        """
        return F.logsigmoid(x) - F.logsigmoid(-x)

    @staticmethod
    def airl_softplus(x: th.Tensor) -> th.Tensor:
        """
        (AIRL): r(s, a) = -softplus(-x) + softplus(x)
        :param x: logits
        """
        return -F.softplus(-x) + F.softplus(x)

    @staticmethod
    def airl_logit(x: th.Tensor) -> th.Tensor:
        """
        (AIRL): r(s, a) = ln D − ln(1 − D) = h
        where h is the logits. Output of f net/function.
        :param x: logits
        """
        return x


class DiscrimTag(Enum):
    GAIL_DISCRIM = auto()
    AIRL_STATE_ONLY_DISCRIM = auto()
    AIRL_STATE_ACTION_DISCRIM = auto()


class GAILDiscrim(DiscrimNet):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Sequence[int],
        hidden_activation: Activation,
        **disc_kwargs,
    ):
        if disc_kwargs is None:
            disc_kwargs = {}

        super().__init__(
            ArchType.sa,
            state_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            disc_kwargs=disc_kwargs,
        )
        self._tag = DiscrimTag.GAIL_DISCRIM

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}: {self.f}, Total params: {count_vars(self.f)}"
        )
    
    @property
    def tag(self):
        return self._tag

    def forward(self, obs: th.Tensor, acts: th.Tensor, **kwargs):
        """
        Output logits of discriminator.
        Naming `f` to keep consistent with base DiscrimNet.
        """
        return self.f(obs, acts)

    def calculate_rewards(
        self, obs: th.Tensor, acts: th.Tensor, choice="logit", **kwargs
    ):
        """
        (GAIL) is to maximize E_{\pi} [-log(1 - D)].
        r(s, a) = − ln(1 − D) = softplus(h)
        """
        with th.no_grad():
            reward_fn = self.reward_fn("gail", choice)
            logits = self.forward(obs, acts, **kwargs)
            rews = reward_fn(logits)
        return rews


class AIRLStateDiscrim(DiscrimNet):
    """
    Discriminator used in AIRL with disentangled reward.
    f_{θ,φ} (s, a, s') = g_θ (s, a) + \gamma h_φ (s') − h_φ (s)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_units: Sequence[int],
        hidden_activation: Activation,
        **disc_kwargs,
    ):
        if disc_kwargs is None:
            disc_kwargs = {}

        super().__init__(
            ArchType.s,
            state_dim,
            action_dim=None,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            disc_kwargs=disc_kwargs,
        )
        self._tag = DiscrimTag.AIRL_STATE_ONLY_DISCRIM

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"{self.g}, Total params: {count_vars(self.net1)}\n"
            f"{self.h}, Total params: {count_vars(self.net2)}"
        )

    @property
    def tag(self):
        return self._tag    
    
    def f(
        self, obs: th.Tensor, dones: th.FloatTensor, next_obs: th.Tensor, gamma: float
    ) -> th.Tensor:
        """
        f(s, a, s' ) = g_θ (s) + \gamma h_φ (s') − h_φ (s)

        f: recover to the advantage
        g: state-only reward function approximator
        h: shaping term
        """
        r_s = self.g(obs)
        v_s = self.h(obs)
        next_vs = self.h(next_obs)
        # * Reshape (1-done) to (n,1) to prevent boardcasting mismatch in case done is (n,).
        return r_s + gamma * (1 - dones).view(-1, 1) * next_vs - v_s

    def forward(
        self,
        obs: th.Tensor,
        dones: th.Tensor,
        next_obs: th.Tensor,
        gamma: float,
        log_pis: Optional[th.Tensor] = None,
        subtract_logp: bool = True,
        **kwargs,
    ) -> th.Tensor:
        """
        Policy Objective.
        \hat{r}_t = log[D_θ(s,a)] - log[1-D_θ(s,a)]
        = log[exp{f_θ} /(exp{f_θ} + \pi)] - log[\pi / (exp{f_θ} + \pi)]
        = f_θ (s,a) - log \pi (a|s)
        """
        if log_pis is not None and subtract_logp:
            # reshape log_pi to prevent size mismatch
            return self.f(obs, dones, next_obs, gamma) - log_pis.view(-1, 1)
        elif log_pis is None and subtract_logp:
            raise ValueError("log_pis is None! Can not subtract None.")
        else:
            return self.f(obs, dones, next_obs, gamma)

    def calculate_rewards(
        self,
        obs: th.Tensor,
        dones: th.Tensor,
        next_obs: th.Tensor,
        gamma: float,
        log_pis: Optional[th.Tensor] = None,
        subtract_logp: bool = True,
        choice="logit",
        **kwargs,
    ) -> th.Tensor:
        """
        Calculate GAN reward.
        """
        kwargs = {
            "dones": dones,
            "next_obs": next_obs,
            "log_pis": log_pis,
            "subtract_logp": subtract_logp,
            "gamma": gamma,
        }
        with th.no_grad():
            reward_fn = self.reward_fn(rew_type="airl", choice=choice)
            logits = self.forward(obs, **kwargs)
            rews = reward_fn(logits)
        return rews


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
        hidden_activation: Activation,
        **disc_kwargs,
    ):
        if disc_kwargs is None:
            disc_kwargs = {}

        super().__init__(
            ArchType.sa,
            state_dim,
            action_dim,
            hidden_units,
            hidden_activation,
            **disc_kwargs,
        )
        self._tag = DiscrimTag.AIRL_STATE_ACTION_DISCRIM

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}: {self.f}, Total params: {count_vars(self.f)}"
        )

    @property
    def tag(self):
        return self._tag    
    
    def forward(
        self,
        obs: th.Tensor,
        acts: th.Tensor,
        log_pis: Optional[th.Tensor] = None,
        subtract_logp: bool = True,
        **kwargs,
    ) -> th.Tensor:
        if log_pis is not None and subtract_logp:
            # Reshape log_pi to prevent size mismatch.
            return self.f(obs, acts) - log_pis.view(-1, 1)
        elif log_pis is None and subtract_logp:
            raise ValueError("log_pis is None! Can not subtract None.")
        else:
            return self.f(obs, acts)

    def calculate_rewards(
        self,
        obs: th.Tensor,
        acts: th.Tensor,
        log_pis: Optional[th.Tensor] = None,
        subtract_logp: bool = True,
        choice="logit",
        **kwargs,
    ) -> th.Tensor:
        kwargs = {
            "acts": acts,
            "log_pis": log_pis,
            "subtract_logp": subtract_logp,
        }
        # TODO: apply reward bound
        with th.no_grad():
            reward_fn = self.reward_fn(rew_type="airl", choice=choice)
            logits = self.forward(obs, **kwargs)
            rews = reward_fn(logits)
        return rews

class DiscrimType(Enum):
    gail = GAILDiscrim
    airl_so = AIRLStateDiscrim
    airl_sa = AIRLStateActionDiscrim
