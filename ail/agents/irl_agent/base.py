from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any

import torch as th

from ail.agents.base import BaseAgent
from ail.common.type_alias import GymSpace


class BaseIRLAgent(BaseAgent, ABC):
    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device,
        fp16,
        seed,
        buffer_exp,
        batch_size,
        gen,
        gen_kwargs: Dict[str, Any],
        disc,
        disc_kwargs,
        lr_disc,
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
        
        self.gen_algo = gen(**gen_kwargs)
        
        self.disc = disc(**disc_kwargs)
        self.lr_disc = lr_disc
        self.optim_disc = self.optim_cls(self.disc.parameters(), lr=self.lr_disc)
        
        # Expert's buffer.
        self.batch_size = batch_size
        self.buffer_exp = buffer_exp
        