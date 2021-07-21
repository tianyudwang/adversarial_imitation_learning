from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any

import torch as th

from ail.agents import ALGO
from ail.agents.base import BaseAgent
from ail.agents.rl_agent.rl_core import OnPolicyAgent, OffPolicyAgent
from ail.buffer import ReplayBuffer, BufferType
from ail.common.type_alias import GymSpace
from ail.network.discrim import DiscrimNet



class BaseIRLAgent(BaseAgent, ABC):
    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        batch_size: int,
        buffer_exp: Union[ReplayBuffer, str],
        buffer_kwargs: Dict[str, Any],
        gen_algo:Union[OnPolicyAgent, OffPolicyAgent, str],
        gen_kwargs: Dict[str, Any],
        optim_kwargs: Optional[Dict[str, Any]],
    ):
        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            optim_kwargs,
        )

        if buffer_kwargs is None:
            buffer_kwargs = {}

        # Expert's buffer.
        self.batch_size = batch_size
        if isinstance(buffer_exp, ReplayBuffer):
            self.buffer_exp = buffer_exp
        elif isinstance(buffer_exp, str):
            assert (
                len(buffer_kwargs) > 0
            ), "Need specifies buffer_kwargs for replay buffer."
            self.buffer_exp = BufferType[buffer_exp].from_data(**buffer_kwargs)
        else:
            raise ValueError(f"Unsupported buffer type: {buffer_exp}")
    
        # Generator
        gen_cls = ALGO[gen_algo] if isinstance(gen_algo, str) else gen_algo
        self.gen = gen_cls(**gen_kwargs)
            
        
        

    @abstractmethod
    def train_generator(self):
        """Train generator (RL policy)"""
        raise NotImplementedError()

    def train_generator(self):
        """Train discriminator"""
        raise NotImplementedError()
