from typing import Union, Optional, Dict, Any

import torch as th

from ail.agents.irl_agent.irl_core import BaseIRLAgent
from ail.agents.rl_agent.rl_core import OnPolicyAgent, OffPolicyAgent
from ail.buffer import ReplayBuffer
from ail.common.type_alias import GymSpace
from ail.network.discrim import DiscrimNet, DiscrimType


class AIRL(BaseIRLAgent):
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
        gen_algo: Union[OnPolicyAgent, OffPolicyAgent, str],
        gen_kwargs: Dict[str, Any],
        disc_cls: Union[DiscrimNet, str],
        disc_kwargs: Dict[str, Any],
        lr_disc: float,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            batch_size,
            buffer_exp,
            buffer_kwargs,
            gen_algo,
            gen_kwargs,
            optim_kwargs,
        )

        if disc_cls is None:
            disc_cls = "airl"

        if disc_kwargs is None:
            disc_kwargs = {}  # * hidden, activation,

        # Discriminator
        if isinstance(disc_cls, str):
            assert (
                disc_cls.lower() in ["airl", "airl_so", "airl_sa"],
                "AIRL has two discrim type: ``airl_so`` and ``airl_sa``. "
                "Default airl will assign to airl_so",
            )
            disc_cls = DiscrimType[disc_cls.lower()].value

        self.disc = disc_cls(self.obs_dim, self.act_dim, **disc_kwargs)
        self.lr_disc = lr_disc
        self.optim_disc = self.optim_cls(self.disc.parameters(), lr=self.lr_disc)

    def train_discriminator(self):
        pass

    def train_generator(self):
        pass
