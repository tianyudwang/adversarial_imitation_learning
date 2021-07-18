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
        set_random_seed(seed)

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
        self.fp16 = fp16 and th.cuda.is_available() and device == "cuda"
        self.scaler = GradScaler() if self.fp16 else None

        # Optimizer kwargs.
        self.optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        self.optim_cls = OPT[self.optim_kwargs.get("optim_cls", "adam").lower()]
        self.optim_set_to_none = self.optim_kwargs.get("optim_set_to_none", False)

        
