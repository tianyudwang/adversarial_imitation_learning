from abc import ABC, abstractmethod
from typing import Union, Optional
from functools import partial
from math import sqrt

from gym.spaces import Box
import torch as th
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

from ail.common.utils import set_random_seed
from ail.common.env_utils import get_obs_shape, get_flat_obs_dim, get_act_dim
from ail.common.pytorch_util import init_gpu, to_numpy, orthogonal_init


class RLAgent(nn.Module, ABC):
    # Modified to inherited from nn.Module, so I can use algo.train and algo.eval
    def __init__(
        self,
        state_space,
        action_space,
        device: Union[th.device, str],
        seed: int,
        gamma: float,
        max_grad_norm: Optional[float] = None,
        fp16: bool = False,
        **kwargs
    ):
        super().__init__()
        set_random_seed(seed)

        self.state_space = state_space
        self.action_space = action_space

        self.state_shape = get_obs_shape(state_space)
        if isinstance(action_space, Box):
            self.action_shape = action_space.shape
        else:
            raise NotImplementedError()

        self.obs_dim = get_flat_obs_dim(state_space)
        self.act_dim = get_act_dim(action_space)
        self.act_low = action_space.low
        self.act_high = action_space.high

        # Device management
        self.device = init_gpu(use_gpu=(device == "cuda"))
        self.fp16 = fp16 and th.cuda.is_available() and device == "cuda"
        self.scaler = GradScaler() if self.fp16 else None

        self.learning_steps = 0
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.clipping = max_grad_norm is not None

    def explore(self, state: th.Tensor):
        assert isinstance(state, th.Tensor)
        with th.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))
        return to_numpy(action), log_pi.item()

    def exploit(self, state):
        assert isinstance(state, th.Tensor)

        with th.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return to_numpy(action)

    @abstractmethod
    def is_update(self, step):
        raise NotImplementedError()

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save_models(self, save_dir):
        raise NotImplementedError()

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
            # AMP
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
