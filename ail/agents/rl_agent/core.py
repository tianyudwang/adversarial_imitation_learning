import os
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
from functools import partial
from math import sqrt

import torch as th
from torch.nn.utils import clip_grad_norm_


from ail.agents.base import BaseAgent
from ail.network.policies import StateIndependentPolicy
from ail.network.value import mlp_value
from ail.buffer.buffer_irl import BUFFER

from ail.common.utils import dataclass_quick_asdict
from ail.common.pytorch_util import to_numpy, orthogonal_init
from ail.common.type_alias import GymSpace, EXTRA_SHAPES, EXTRA_DTYPES


class BaseRLAgent(BaseAgent, ABC):
    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        gamma: float,
        max_grad_norm: Optional[float],
        batch_size: int,
        buffer_size: int,
        optim_kwargs: Optional[Dict[str, Any]],
        buffer_kwargs: Optional[Dict[str, Any]],
    ):
        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            optim_kwargs,
        )
        
        # Buffer kwargs.
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer_kwargs = {} if buffer_kwargs is None else buffer_kwargs
        
        # Other parameters.
        self.learning_steps = 0
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.clipping = max_grad_norm is not None

    def explore(self, state: th.Tensor):
        assert isinstance(state, th.Tensor)
        with th.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))
        return to_numpy(action)[0], log_pi

    def exploit(self, state):
        assert isinstance(state, th.Tensor)

        with th.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return to_numpy(action)[0]

    @abstractmethod
    def is_update(self, step):
        raise NotImplementedError()

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _init_models_componet(self, policy_kwargs: Dict[str, Any]):
        """Check if the core componet exits in policy kwargs."""
        assert policy_kwargs is not None, "policy_kwargs cannot be None."
        assert isinstance(
            policy_kwargs, dict
        ), "policy_kwargs must be a Dict[str, Any]."
        assert len(policy_kwargs) > 0, "policy_kwargs cannot be empty."

        assert "pi" in policy_kwargs, "Missing `pi` key in policy_kwargs."
        assert "vf" in policy_kwargs, "Missing `vf` key in policy_kwargs."
        assert (
            "activation" in policy_kwargs
        ), "Missing `activation` key in policy_kwargs."
        assert (
            "critic_type" in policy_kwargs
        ), "Missing `critic_type` key in policy_kwargs."
        assert "lr_actor" in policy_kwargs, "Missing `lr_actor` key in policy_kwargs."
        assert "lr_critic" in policy_kwargs, "Missing `lr_critic` key in policy_kwargs."

        self.units_actor = policy_kwargs["pi"]
        self.units_critic = policy_kwargs["vf"]
        self.hidden_activation = policy_kwargs["activation"]
        self.critic_type = policy_kwargs["critic_type"]
        self.lr_actor = policy_kwargs["lr_actor"]
        self.lr_critic = policy_kwargs["lr_critic"]

    def _init_buffer(self, buffer_type: str):
        """Initialize the rollout/replay buffer."""
        data = self.buffer_kwargs.get("extra_data", [])
        if not isinstance(data, (list, tuple)):
            data = [data]

        shape_dict = dataclass_quick_asdict(EXTRA_SHAPES)
        dtypes_dict = dataclass_quick_asdict(EXTRA_DTYPES)

        extra_shapes = {k: shape_dict[k] for k in data if k in shape_dict}
        extra_dtypes = {k: dtypes_dict[k] for k in data if k in dtypes_dict}

        buffer_cls = BUFFER[buffer_type.lower()]

        self.buffer = buffer_cls(
            capacity=self.buffer_size,
            device=self.device,
            obs_shape=self.state_shape,
            act_shape=self.action_shape,
            with_reward=self.buffer_kwargs.get("with_reward", True),
            extra_shapes=extra_shapes,
            extra_dtypes=extra_dtypes,
        )

    # Helper methods
    def weight_initiation(self):
        # Originally from openai/baselines (default gains/init_scales).
        module_gains = {
            self.actor: sqrt(2),
            self.critic: sqrt(2),
        }
        for module, gain in module_gains.items():
            module.apply(partial(orthogonal_init, gain=gain))

    def one_gradient_step(self, loss, opt, net):
        """take one gradient step with grad clipping.(AMP support)"""
        if self.fp16:
            # AMP.
            self.scaler.scale(loss).backward()
            if self.clipping:
                self.scaler.unscale_(opt)
                clip_grad_norm_(net.parameters(), max_norm=self.max_grad_norm)
            self.scaler.step(opt)
            self.scaler.update()
        else:
            # Unscale_update.
            loss.backward()
            if self.clipping:
                clip_grad_norm_(net.parameters(), max_norm=self.max_grad_norm)
            opt.step()


class OnPolicyAgent(BaseRLAgent):
    def __init__(
        self,
        state_space,
        action_space,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        gamma: float,
        max_grad_norm: Optional[float],
        batch_size: int,
        buffer_size: int,
        policy_kwargs: Dict[str, Any],
        optim_kwargs: Optional[Dict[str, Any]],
        buffer_kwargs: Optional[Dict[str, Any]],
        init_buffer: bool,
        init_models: bool,
        **kwargs
    ):
        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            gamma,
            max_grad_norm,
            batch_size,
            buffer_size,
            optim_kwargs,
            buffer_kwargs,
        )

        # Rollout Buffer.
        if init_buffer:
            self._init_buffer(buffer_type="rollout")

        # Policy kwargs.
        self._init_models_componet(policy_kwargs)

        # Build actor and critic and initialize optimizer.
        if init_models:
            self._setup_models()
            # Optinally weight initialization
            orthogonal_init = policy_kwargs.get("orthogonal_init", False)
            if orthogonal_init:
                self.weight_initiation()
            self.optim_actor = self.optim_cls(self.actor.parameters(), lr=self.lr_actor)
            self.optim_critic = self.optim_cls(
                self.critic.parameters(), lr=self.lr_critic
            )

    def _setup_models(self):
        """Build model for actor and critic."""
        # Actor.
        self.actor = StateIndependentPolicy(
            self.obs_dim,
            self.act_dim,
            self.units_actor,
            self.hidden_activation,
        ).to(self.device)

        # Critic.
        self.critic = mlp_value(
            self.obs_dim,
            self.act_dim,
            self.units_critic,
            self.hidden_activation,
            self.critic_type,
        ).to(self.device)


class OffPolicyAgent(BaseRLAgent):
    def __init__(
        self,
        state_space,
        action_space,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        gamma: float,
        max_grad_norm: Optional[float],
        batch_size: int,
        buffer_size: int,
        policy_kwargs: Dict[str, Any],
        optim_kwargs: Optional[Dict[str, Any]],
        buffer_kwargs: Optional[Dict[str, Any]],
        init_buffer: bool,
        init_models: bool,  # TODO: not implemented
        **kwargs
    ):
        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            gamma,
            max_grad_norm,
            batch_size,
            buffer_size,
            optim_kwargs,
            buffer_kwargs,
        )

        # Replay Buffer.
        if init_buffer:
            self._init_buffer(buffer_type="replay")

        # Policy kwargs.
        self._init_models_componet(policy_kwargs)

        # # Build actor and critic and initialize optimizer.
        # if init_models:
        #     self._init_models()
        #     # Optinally weight initialization
        #     orthogonal_init = policy_kwargs.get("orthogonal_init", False)
        #     if orthogonal_init:
        #         self.weight_initiation()
        #     self.optim_actor = self.optim_cls(self.actor.parameters(), lr=self.lr_actor)
        #     self.optim_critic = self.optim_cls(
        #         self.critic.parameters(), lr=self.lr_critic
        #     )