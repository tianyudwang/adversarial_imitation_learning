from typing import Union, Optional, Dict, Any

import torch as th
import torch.nn.functional as F

from ail.agents.irl_agent.irl_core import BaseIRLAgent
from ail.agents.rl_agent.rl_core import OnPolicyAgent, OffPolicyAgent
from ail.buffer import ReplayBuffer
from ail.common.type_alias import GymSpace, TensorDict
from ail.network.discrim import DiscrimNet, DiscrimType


class AIRL(BaseIRLAgent):
    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        epoch_disc: int,
        replay_batch_size: int,
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
            replay_batch_size,
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

        self.learning_steps_disc = 0
        self.epoch_disc = epoch_disc

        # ? Create alias?

    def __repr__(self):
        return "AIRL"

    def update(self, log_this_batch: bool = False) -> Dict[str, Any]:

        # Main loop
        # 1. Interact with the environment using the current generator
        #    and store the experience in a replay buffer
        # 2. Update discriminator
        # 3. Update generator

        for _ in range(self.epoch_disc):
            # * samples from *current* policy's trajectories
            # TODO: find a way to work with off policies
            # * get current trajectories
            data_gen = self.gen.buffer.get(self.replay_batch_size)
            # Samples from expert's demonstrations.
            data_exp = self.buffer_exp.sample(self.replay_batch_size)

            # Calculate log probabilities of agent and expert actions.
            with th.no_grad():
                # ? maybe we don't need to calculate the log probs of the expert actions
                data_exp["log_pis"] = self.gen.actor.evaluate_log_pi(
                    data_exp["obs"], data_exp["acts"]
                )
            # Update discriminator.
            disc_logs = self.update_discriminator(data_gen, data_exp, log_this_batch)
            # TODO: a deafult dict to handle multiple logs
        # Calculate rewards:
        # # TODO: buffer.get or sample()
        #     states,
        #     actions,
        #     dones,
        #     log_pis,
        #     next_states,
        rollout_data = self.buffer.get()  # TODO: n_samples
        rollout_data["rews"] = self.disc.calculate_rewards(...)
        assert rollout_data["rews"].shape[0] == rollout_data["obs"].shape[0]

        # Update generator using estimated rewards.
        gen_logs = self.update_generator(rollout_data, log_this_batch)

    def update_generator(self, log_this_batch: bool = False) -> Dict[str, Any]:
        """Update generator algo."""
        return self.gen.update(log_this_batch)

    def update_discriminator(
        self, data_gen: TensorDict, data_exp: TensorDict, log_this_batch: bool = False
    ) -> Dict[str, Any]:
        """
        Update discriminator.

        Discriminator is to maximize E_{exp} [log(D)] + E_{\pi} [log(1 - D)]
        D = sigmoid(f)
        Output of disc is logits `f` in range (-inf, inf), not [0, 1].
        :param data_gen: batch of data from the current policy
        :param data_exp: batch of data from demonstrations
        """
        # Obtain logits of the discriminator.
        logits_gen = self.disc(**data_gen)
        logits_exp = self.disc(**data_exp)

        # E_{exp} [log(D)] + E_{\pi} [log(1 - D)]
        # E_{exp} [log(sigmoid(f))] + E_{\pi} [log(1 - sigmoid(f))]
        # *Note: S(x) = 1 - S(-x) -> S(-x) = 1 - S(x)
        loss_pi = -F.logsigmoid(-logits_gen).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        # TODO: one_gradient_step
        self.optim_disc.zero_grad(self.optim_set_to_none)
        loss_disc.backward()
        self.optim_disc.step()

        if log_this_batch:
            acc_gen = (logits_gen.detach() < 0).float().mean()
            acc_exp = (logits_exp.detach() > 0).float().mean()
            # TODO: avoid print(cuda_tensor)
            ic(acc_gen)
            ic(acc_exp)
            return {
                "disc_loss": loss_disc.detach(),
                "acc_gen": acc_gen,
                "acc_exp": acc_exp,
            }
        else:
            return {}
