from typing import Union, Optional, Dict, Any
from collections import OrderedDict

import torch as th
from torch import nn
from torch.cuda.amp import autocast
from torch.distributions import Bernoulli

from ail.agents.irl_agent.irl_core import BaseIRLAgent
from ail.buffer import ReplayBuffer, BufferTag
from ail.common.math import normalize
from ail.common.type_alias import GymSpace, TensorDict
from ail.network.discrim import DiscrimNet


class Adversarial(BaseIRLAgent):
    """
    Base class for adversarial imitation learning algorithms like GAIL and AIRL.

    :param state_space: state space.
    :param action_space: action space.
    :param device: PyTorch device to which the values will be converted.
    :param seed: random seed.
    :param max_grad_norm: Maximum norm for the gradient clipping
    :param epoch_disc: Number of epoch when update the discriminator
    :param replay_batch_size: Replay batch size for training the discriminator
    :param buffer_exp: Replay buffer that store expert demostrations
    :param buffer_kwargs: Arguments to be passed to the buffer.
        eg. : {
            with_reward: False,
            extra_data: ["log_pis"]
            }
    :param gen_algo: RL algorithm for the generator.
    :param gen_kwargs: Kwargs to be passed to the generator.
    :param disc_cls: Class for DiscrimNet,
    :param disc_kwargs: Expected kwargs to be passed to the DiscrimNet.
    :param lr_disc: learning rate for the discriminator
    :param optim_kwargs: arguments to be passed to the optimizer.
        eg. : {
            "optim_cls": adam,
            "optim_set_to_none": True, # which set grad to None instead of zero.
            }
    :param subtract_logp: wheteher to subtract log_pis from the learning reward.
    :param rew_type: GAIL or AIRL flavor of learning reward.
    :param rew_input_choice: Using logit or logsigmoid or softplus to calculate reward function
    """

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
        disc_cls: DiscrimNet,
        disc_kwargs: Dict[str, Any],
        lr_disc: float,
        disc_ent_coef: float,
        optim_kwargs: Optional[Dict[str, Any]],
        subtract_logp: bool,
        rew_type: str,
        rew_input_choice: str,
        rew_clip: bool,
        max_rew_magnitude: float,
        obs_normalization: Optional[str],
        **kwargs,
    ):
        assert max_rew_magnitude > 0, "max_rew_magnitude must be greater than 0"
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
        # DiscrimNet
        self.disc = disc_cls(self.obs_dim, self.act_dim, **disc_kwargs).to(self.device)
        self.lr_disc = lr_disc
        self.optim_disc = self.optim_cls(self.disc.parameters(), lr=self.lr_disc)

        # Lables for the discriminator(Assuming same batch size for gen and exp)
        self.disc_labels = th.cat(
            [
                th.zeros(self.replay_batch_size, dtype=th.float32),
                th.ones(self.replay_batch_size, dtype=th.float32),
            ]
        ).reshape(-1, 1)

        # loss function for the discriminator
        self.disc_criterion = nn.BCEWithLogitsLoss(reduction="mean")

        # Coeffient for entropy bonus
        assert disc_ent_coef >= 0, "disc_ent_coef must be non-negative."
        self.disc_ent_coef = disc_ent_coef

        self.learning_steps_disc = 0
        self.epoch_disc = epoch_disc

        # Reward function args
        self.subtract_logp = subtract_logp
        self.rew_type = rew_type
        self.rew_input_choice = rew_input_choice
        self.rew_clip = rew_clip
        self.max_rew_magnitude = max_rew_magnitude

        if obs_normalization is not None:
            assert isinstance(
                obs_normalization, str
            ), "obs_normalization should be a string"
            if obs_normalization == "fixed":
                self.normalize_obs = True
                self.normalize_mode = "fixed"
            elif obs_normalization == "online":
                self.normalize_obs = True
                self.normalize_mode = "online"
                raise NotImplementedError()
            else:
                raise ValueError(
                    f"Valid inputs of obs_normalization: ['fixed', 'online']."
                )
        else:
            self.normalize_obs = False

        self.absorbing_states = th.zeros(self.obs_dim + 1).reshape(1, -1)
        self.absorbing_states[:, -1] = 1.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def update(self, log_this_batch: bool = False) -> Dict[str, Any]:
        """
        Main loop
         1. Interact with the environment using the current generator/ policy.
            and store the experience in a replay buffer (implementing in step()).
         2. Update discriminator.
         3. Update generator.
        """
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1
            # * Sample transitions from ``current`` policy.
            if self.gen.buffer.tag == BufferTag.ROLLOUT:
                data_gen = self.gen.buffer.sample(self.replay_batch_size)

            elif self.gen.buffer.tag == BufferTag.REPLAY:
                data_gen = self.gen.buffer.get(self.replay_batch_size, last_n=True)

            else:
                raise ValueError(f"Unknown generator buffer type: {self.gen.buffer}.")

            # Samples transitions from expert's demonstrations.
            data_exp = self.buffer_exp.sample(self.replay_batch_size)

            # self.make_absorbing_states(data_gen["obs"], data_gen["dones"])

            if self.normalize_obs:

                data_gen["obs"] = self.fix_normalize_obs(
                    data_gen["obs"], data_exp["obs"]
                )
                data_exp["obs"] = self.fix_normalize_obs(
                    data_exp["obs"], data_exp["obs"]
                )
                data_gen["next_obs"] = self.fix_normalize_obs(
                    data_gen["next_obs"], data_exp["next_obs"]
                )
                data_exp["next_obs"] = self.fix_normalize_obs(
                    data_exp["next_obs"], data_exp["next_obs"]
                )
                # ic(data_gen["obs"].mean(0), data_gen["next_obs"].mean(0))
                # ic(data_gen["obs"].std(0), data_gen["next_obs"].std(0))
                # ic(data_exp["obs"].mean(0), data_exp["next_obs"].mean(0))
                # ic(data_exp["obs"].std(0), data_exp["next_obs"].std(0))

            # Calculate log probabilities of generator's actions.
            # And evaluate log probabilities of expert actions.
            # Based on current generator's action distribution.
            # See: https://arxiv.org/pdf/1710.11248.pdf appendix A.2
            if self.subtract_logp:
                with th.no_grad():
                    data_exp["log_pis"] = self.gen.actor.evaluate_log_pi(
                        data_exp["obs"], data_exp["acts"]
                    )
            # Update discriminator.
            disc_logs = self.update_discriminator(data_gen, data_exp, log_this_batch)
            if log_this_batch:
                disc_logs.update(
                    {
                        "lr_disc": self.lr_disc,
                        "learn_steps_disc": self.learning_steps_disc,
                    }
                )
                disc_logs = dict(disc_logs)
            del data_gen, data_exp

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

        # if self.normalize_obs:
        #     data["obs"] = normalize(data["obs"], exp_obs_mean, exp_obs_std)
        #     data["next_obs"] = normalize(
        #         data["next_obs"], exp_next_obs_mean, exp_next_obs_std
        #     )

        # Calculate learning rewards.
        data["rews"] = self.disc.calculate_rewards(choice=self.rew_input_choice, **data)
        # Sanity check length of data are equal.
        assert data["rews"].shape[0] == data["obs"].shape[0]

        # Reward Clipping
        if self.rew_clip:
            ic("clipped")
            data["rews"].clamp_(-self.max_rew_magnitude, self.max_rew_magnitude)

        # Update generator using estimated rewards.
        gen_logs = self.update_generator(data, log_this_batch)

        # train_logs = {}
        return gen_logs, disc_logs

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
            disc_logits_gen = self.disc(subtract_logp=self.subtract_logp, **data_gen)
            disc_logits_exp = self.disc(subtract_logp=self.subtract_logp, **data_exp)

            """
            E_{exp} [log(D)] + E_{\pi} [log(1 - D)]
            E_{exp} [log(sigmoid(f))] + E_{\pi} [log(1 - sigmoid(f))]
            *Note: S(x) = 1 - S(-x) -> S(-x) = 1 - S(x)
            
            # ! Deprecated:
            Implmentation below is correct, but using BCEWithLogitsLoss
            is more numerically stable than using a plain Sigmoid followed by a BCELoss
            
            loss_gen = F.logsigmoid(-logits_gen).mean()
            loss_exp = F.logsigmoid(logits_exp).mean()
            loss_disc = -(loss_gen + loss_exp)
            """
            # Check dimensions of logits.
            assert disc_logits_gen.shape[0] == disc_logits_exp.shape[0]
            disc_logits = th.vstack([disc_logits_gen, disc_logits_exp])
            loss_disc = self.disc_criterion(disc_logits, self.disc_labels)

            if self.disc_ent_coef > 0:
                label_dist = Bernoulli(logits=disc_logits)
                entropy = th.mean(label_dist.entropy())
                loss_disc -= self.disc_ent_coef * entropy

        self.one_gradient_step(loss_disc, self.optim_disc, self.disc)

        disc_logs = {}
        if log_this_batch:
            # Discriminator's statistics.
            disc_logs = self.compute_disc_stats(
                disc_logits, self.disc_labels, loss_disc
            )

        return disc_logs

    @staticmethod
    def fix_normalize_obs(input_obs: th.Tensor, obs_exp: th.Tensor) -> th.Tensor:
        """
        Normalize expert's observations.
        :param obs_exp: expert's observations which has shape (batch_size, obs_dim)
        :return: normalized input_obs with approximately zero mean and std one
        """
        exp_obs_mean, exp_obs_std = obs_exp.mean(axis=0), obs_exp.std(axis=0)
        return normalize(input_obs, exp_obs_mean, exp_obs_std)

    def make_absorbing_states(self, obs: th.Tensor, dones: th.Tensor) -> th.Tensor:
        combined_states = th.hstack([obs, dones])
        is_done = th.all(combined_states, dim=-1, keepdims=True)
        absorbing_obs = th.where(is_done, self.absorbing_states, combined_states)
        return absorbing_obs

    @staticmethod
    def compute_disc_stats(
        disc_logits: th.Tensor,
        labels: th.Tensor,
        disc_loss: th.Tensor,
    ) -> Dict[str, float]:
        """
        Train statistics for GAIL/AIRL discriminator, or other binary classifiers.
        :param disc_logits: discriminator logits where expert is 1 and generated is 0
        :param labels: integer labels describing whether logit was for an
                expert (1) or generator (0) sample.
        :param disc_loss: discriminator loss.
        :returns stats: dictionary mapping statistic names for float values.
        """
        with th.no_grad():
            bin_is_exp_pred = disc_logits > 0
            bin_is_exp_true = labels > 0
            bin_is_gen_true = th.logical_not(bin_is_exp_true)

            int_is_exp_pred = bin_is_exp_pred.long()
            int_is_exp_true = bin_is_exp_true.long()

            n_labels = float(len(labels))
            n_exp = float(th.sum(int_is_exp_true))
            n_gen = n_labels - n_exp

            percent_gen = n_gen / float(n_labels) if n_labels > 0 else float("NaN")
            n_gen_pred = int(n_labels - th.sum(int_is_exp_pred))

            if n_labels > 0:
                percent_gen_pred = n_gen_pred / float(n_labels)
            else:
                percent_gen_pred = float("NaN")

            correct_vec = th.eq(bin_is_exp_pred, bin_is_exp_true)
            disc_acc = th.mean(correct_vec.float())

            _n_pred_gen = th.sum(th.logical_and(bin_is_gen_true, correct_vec))
            if n_gen < 1:
                gen_acc = float("NaN")
            else:
                # float() is defensive, since we cannot divide Torch tensors by
                # Python ints
                gen_acc = _n_pred_gen / float(n_gen)

            _n_pred_exp = th.sum(th.logical_and(bin_is_exp_true, correct_vec))
            _n_exp_or_1 = max(1, n_exp)
            exp_acc = _n_pred_exp / float(_n_exp_or_1)

            label_dist = Bernoulli(logits=disc_logits)
            entropy = th.mean(label_dist.entropy())

        pairs = [
            ("disc_loss", float(th.mean(disc_loss))),
            # Accuracy, as well as accuracy on *just* expert examples and *just*
            # generated examples
            ("disc_acc", float(disc_acc)),
            ("disc_acc_gen", float(gen_acc)),
            ("disc_acc_exp", float(exp_acc)),
            # Entropy of the predicted label distribution, averaged equally across
            # both classes (if this drops then disc is very good or has given up)
            ("disc_entropy", float(entropy)),
            # True number of generators and predicted number of generators
            ("proportion_gen_true", float(percent_gen)),
            ("proportion_gen_pred", float(percent_gen_pred)),
        ]
        return OrderedDict(pairs)
