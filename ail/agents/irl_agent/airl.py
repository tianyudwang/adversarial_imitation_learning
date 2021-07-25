from typing import Union, Optional, Dict, Any

import torch as th
import torch.nn.functional as F

from ail.agents.irl_agent.irl_core import BaseIRLAgent
from ail.buffer import ReplayBuffer, BufferTag
from ail.common.type_alias import GymSpace, TensorDict
from ail.network.discrim import DiscrimNet, DiscrimType, ArchType


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
        gen_algo,
        gen_kwargs: Dict[str, Any],
        disc_cls: Union[DiscrimNet, str],
        disc_kwargs: Dict[str, Any],
        lr_disc: float,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
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

        if disc_kwargs is None:
            disc_kwargs = {}  # * hidden, activation,

        if disc_cls is None or disc_cls == "airl_so":
            disc_cls = "airl"
            disc_kwargs["disc_type"] = ArchType.s
        elif disc_cls == "airl_sa":
            disc_kwargs["disc_type"] = ArchType.sa

        # Discriminator
        if isinstance(disc_cls, str):
            disc_cls = DiscrimType[disc_cls.lower()].value

        self.disc = disc_cls(self.obs_dim, self.act_dim, **disc_kwargs)
        self.lr_disc = lr_disc
        self.optim_disc = self.optim_cls(self.disc.parameters(), lr=self.lr_disc)

        self.learning_steps_disc = 0
        self.epoch_disc = epoch_disc

        self.acc_gen = []
        self.acc_exp = []

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

            # Not needed. Delete to save memory.
            del data_gen, data_exp
            # TODO: a deafult dict to handle multiple logs
        # Calculate rewards:
        # # TODO: buffer.get or sample()
        #     states,
        #     actions,
        #     dones,
        #     log_pis,
        #     next_states,

        if self.gen.buffer.tag == BufferTag.ROLLOUT:
            # Obtain entire batch of transitions from rollout buffer.
            data = self.gen.buffer.get()
            # Clear buffer after getting entire buffer.
            self.gen.buffer.reset()

        elif self.gen.buffer.tag == BufferTag.REPLAY:
            # Random uniform sampling a batch of transitions from agent's replay buffer
            data = self.gen.buffer.sample(self.gen.batch_size)

        # Claculate learning rewards
        data["rews"] = self.disc.calculate_rewards(**data)
        assert data["rews"].shape[0] == data["obs"].shape[0]

        # Update generator using estimated rewards.
        gen_logs = self.update_generator(data, log_this_batch)

        return {}

    def update_generator(
        self, data: TensorDict, log_this_batch: bool = False
    ) -> Dict[str, Any]:
        """Update generator algo."""
        return self.gen.update_algo(data, log_this_batch)

    def update_discriminator(
        self, data_gen: TensorDict, data_exp: TensorDict, log_this_batch: bool = False
    ) -> Dict[str, Any]:
        """
        Update discriminator.
        Let D denote the probability that a state-action pair (s, a) is classified as expert
        by the discriminator while f is the discriminator logit.

        The objective of the discriminator is to minimize cross-entropy loss
        between expert demonstrations and generated samples:

        L = sum( -E_{exp} [log(D)] - E_{\pi} [log(1 - D)] )

        We write the ``negative`` loss to turn the ``minimization`` problem into ``maximization``.

        -L = sum( E_{exp} [log(D)] + E_{\pi} [log(1 - D)] )

        D = sigmoid(f)
        Output of self.disc() is logits `f` in range (-inf, inf), not [0, 1].
        :param data_gen: batch of data from the current policy
        :param data_exp: batch of data from demonstrations
        """
        # Obtain logits of the discriminator.
        logits_gen = self.disc(**data_gen)
        logits_exp = self.disc(**data_exp)

        """
        E_{exp} [log(D)] + E_{\pi} [log(1 - D)]
        E_{exp} [log(sigmoid(f))] + E_{\pi} [log(1 - sigmoid(f))]
        *Note: S(x) = 1 - S(-x) -> S(-x) = 1 - S(x)
        """
        loss_pi = -F.logsigmoid(-logits_gen).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        # TODO: one_gradient_step
        self.optim_disc.zero_grad(self.optim_set_to_none)
        loss_disc.backward()
        self.optim_disc.step()

        # TODO: remove this
        if log_this_batch:
            # Discriminator's accuracies.
            with th.no_grad():
                acc_gen = (logits_gen.detach() < 0).float().mean().item()
                acc_exp = (logits_exp.detach() > 0).float().mean().item()
                self.acc_gen.append(acc_gen)
                self.acc_exp.append(acc_exp)
            if len(self.acc_gen) == self.epoch_disc:
                import numpy as np

                acc_gen = round(np.mean(self.acc_gen), 4)
                acc_exp = round(np.mean(self.acc_exp), 4)
                self.acc_gen.clear()
                self.acc_exp.clear()
                ic(acc_gen)
                ic(acc_exp)
            return {
                "disc_loss": loss_disc.detach(),
                "acc_gen": acc_gen,
                "acc_exp": acc_exp,
            }
        else:
            return {}
