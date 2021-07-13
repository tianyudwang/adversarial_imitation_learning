from abc import ABC, abstractmethod
from typing import Union

from gym.spaces import Box
import torch as th
from torch import nn


from ail.common.utils import set_random_seed
from ail.common.env_utils import get_obs_shape, get_flat_obs_dim, get_act_dim
from ail.common.pytorch_util import init_gpu


class BaseAgent(nn.Module, ABC):
    def __init__(
        self,
        state_space,
        action_space,
        device: Union[th.device, str],
        seed: int,
        gamma: float,
        **kwargs
    ):
        super(BaseAgent, self).__init__()
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

        self.learning_steps = 0
        self.gamma = gamma

    def explore(self, state: th.Tensor):
        assert isinstance(state, th.Tensor)
        with th.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        assert isinstance(state, th.Tensor)

        with th.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        pass
