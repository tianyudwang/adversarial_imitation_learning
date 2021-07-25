from typing import Union, Optional, Dict, Any

from gym.spaces import Box
import torch as th
from torch import nn
from torch.cuda.amp import GradScaler


from ail.common.type_alias import OPT, GymSpace
from ail.common.utils import set_random_seed
from ail.common.pytorch_util import init_gpu
from ail.common.env_utils import get_obs_shape, get_flat_obs_dim, get_act_dim


class BaseAgent(nn.Module):
    """
    Base class for all agents.
    :param state_space: state space.
    :param action_space: action space.
    :param device: PyTorch device to which the values will be converted.
    :param fp16: Whether to use float16 mixed precision training.
    :param seed: random seed.
    :optim_kwargs: arguments to be passed to the optimizer.
        eg. : {
            "optim_cls": adam,
            "optim_set_to_none": True, # which set grad to None instead of zero.
            }
    """

    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        optim_kwargs: Optional[Dict[str, Any]],
    ):
        super().__init__()

        # RNG.
        if not isinstance(seed, int):
            raise ValueError("seed must be integer.")
        self.seed = seed
        set_random_seed(self.seed)

        # env spaces.
        self.state_space = state_space
        self.action_space = action_space

        # shapes of space useful for buffer.
        self.state_shape = get_obs_shape(state_space)
        if isinstance(action_space, Box):
            self.action_shape = action_space.shape
        else:
            raise NotImplementedError()

        # Space dimension and action dimension.
        self.obs_dim = get_flat_obs_dim(state_space)
        self.act_dim = get_act_dim(action_space)

        # Action limits.
        self.act_low = action_space.low
        self.act_high = action_space.high

        # Device management.
        self.device = init_gpu(use_gpu=(device == "cuda"))
        # Use automatic mixed precision training in GPU
        self.fp16 = all([fp16, th.cuda.is_available(), device == "cuda"])
        self.scaler = GradScaler() if self.fp16 else None

        # Optimizer kwargs.
        self.optim_kwargs = {} if optim_kwargs is None else optim_kwargs

        optim_cls = self.optim_kwargs.get("optim_cls", "adam")
        if isinstance(optim_cls, str):
            self.optim_cls = OPT[optim_cls.lower()].value
        elif isinstance(optim_cls, th.optim.Optimizer):
            self.optim_cls = optim_cls
        else:
            raise ValueError("optim_cls must be a string or an torch. optim.Optimizer.")

        self.optim_set_to_none = self.optim_kwargs.get("optim_set_to_none", False)
