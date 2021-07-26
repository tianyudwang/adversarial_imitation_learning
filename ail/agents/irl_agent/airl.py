from typing import Union, Optional, Dict, Any
from collections import defaultdict

import torch as th
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ail.agents.irl_agent.irl_core import BaseIRLAgent
from ail.buffer import ReplayBuffer, BufferTag
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
        max_grad_norm: Optional[float],
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
        subtract_logp: bool = True,
        rew_type: str = "airl",
        rew_input_choice: str = "logit",
        **kwargs,
    ):

        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            max_grad_norm,
            replay_batch_size,
            buffer_exp,
            buffer_kwargs,
            gen_algo,
            gen_kwargs,
            optim_kwargs,
        )

        if disc_kwargs is None:
            disc_kwargs = {}

        # Discriminator
        if isinstance(disc_cls, str):
            disc_cls = disc_cls.lower()
            if disc_cls not in ["airl", "airl_so", "airl_sa"]:
                raise ValueError(
                    f"No string string conversion of discriminator class {disc_cls}."
                )
            disc_cls = DiscrimType[disc_cls].value

        else:
            if isinstance(disc_cls, DiscrimNet):
                disc_cls = DiscrimNet
            else:
                raise ValueError(f"Unknown discriminator class: {disc_cls}.")
        
        self.disc = disc_cls(self.obs_dim, self.act_dim, **disc_kwargs).to(self.device)
        self.lr_disc = lr_disc
        self.optim_disc = self.optim_cls(self.disc.parameters(), lr=self.lr_disc)

        self.learning_steps_disc = 0
        self.epoch_disc = epoch_disc

        # Reward function args
        self.subtract_logp = subtract_logp
        self.rew_type = rew_type
        self.rew_input_choice = rew_input_choice

    def __repr__(self) -> str:
        return "AIRL"

    def update(self, log_this_batch: bool = False) -> Dict[str, Any]:
        """
        Main loop
         1. Interact with the environment using the current generator/ policy.
            and store the experience in a replay buffer (implementing in step()).
         2. Update discriminator.
         3. Update generator.
        """
        # TODO: find a way to work with off policies
        disc_logs = defaultdict(list)
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1
            # * Sample transitions from ``curret`` policy.
            data_gen = self.gen.buffer.get(self.replay_batch_size)
            # Samples transitions from expert's demonstrations.
            data_exp = self.buffer_exp.sample(self.replay_batch_size)

            # Calculate log probabilities of geberator's actions.
            # And evaluate log probabilities of expert actions.
            # Based on current generator's action distribution.
            with th.no_grad():
                # ? maybe we don't need to calculate the log probs of the expert actions?
                data_exp["log_pis"] = self.gen.actor.evaluate_log_pi(
                    data_exp["obs"], data_exp["acts"]
                )
            # Update discriminator.
            disc_info = self.update_discriminator(data_gen, data_exp, log_this_batch)
            if log_this_batch:
                for k in disc_info.keys():
                    disc_logs[k].append(disc_info[k])
                disc_logs.update({
                    "lr_disc": self.lr_disc,
                    "learn_steps_disc":self.learning_steps_disc
                })
        
        # TODO: buffer.get or sample()
        # Calculate rewards:
        if self.gen.buffer.tag == BufferTag.ROLLOUT:
            # Obtain entire batch of transitions from rollout buffer.
            data = self.gen.buffer.get()
            # Clear buffer after getting entire buffer.
            self.gen.buffer.reset()

        elif self.gen.buffer.tag == BufferTag.REPLAY:
            # Random uniform sampling a batch of transitions from agent's replay buffer
            data = self.gen.buffer.sample(self.gen.batch_size)

        else:
            raise ValueError(f"Unknown generator buffer type: {self.gen.buffer}.")

        # Calculate learning rewards.
        data["rews"] = self.disc.calculate_rewards(
            rew_type=self.rew_type, choice=self.rew_input_choice, **data
        )
        # Sanity check length of data are equal.
        assert data["rews"].shape[0] == data["obs"].shape[0]

        # Update generator using estimated rewards.
        gen_logs = self.update_generator(data, log_this_batch)
        
        train_logs = {**gen_logs, **disc_logs}
        return train_logs

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
        self.optim_disc.zero_grad(self.optim_set_to_none)
        with autocast(enabled=self.fp16):
            # Obtain logits of the discriminator.
            logits_gen = self.disc(subtract_logp=self.subtract_logp, **data_gen)
            logits_exp = self.disc(subtract_logp=self.subtract_logp, **data_exp)

            """
            E_{exp} [log(D)] + E_{\pi} [log(1 - D)]
            E_{exp} [log(sigmoid(f))] + E_{\pi} [log(1 - sigmoid(f))]
            *Note: S(x) = 1 - S(-x) -> S(-x) = 1 - S(x)
            """
            loss_pi = F.logsigmoid(-logits_gen).mean()
            loss_exp = F.logsigmoid(logits_exp).mean()
            loss_disc = -(loss_pi + loss_exp)

        self.one_gradient_step(loss_disc, self.optim_disc, self.disc)        
        
        disc_logs ={}
        if log_this_batch:
            # Discriminator's accuracies.
            with th.no_grad():
                acc_gen = (logits_gen.detach() < 0).float().mean()
                acc_exp = (logits_exp.detach() > 0).float().mean()

            disc_logs.update({
                "disc_loss": loss_disc.detach(),
                "acc_gen": acc_gen,
                "acc_exp": acc_exp,
            })
        return disc_logs 
