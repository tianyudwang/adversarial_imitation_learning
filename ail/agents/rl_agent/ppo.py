from typing import Union, Optional, Dict, Any

import torch as th
from torch import nn
from torch.cuda.amp import autocast

from ail.agents.rl_agent.base import OnPolicyAgent
from ail.common.math import normalize
from ail.common.type_alias import TensorDict, GymSpace
from ail.common.pytorch_util import asarray_shape2d

# TODO: test performance with scipy.filter and torch-discount-cumsum
def calculate_gae(rewards, dones, values, next_values, gamma, lambd, normal=True):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = th.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    targets = values + gaes

    if normal:
        return targets, normalize(gaes, gaes.mean(), gaes.std())
    else:
        return targets, gaes


class PPO(OnPolicyAgent):
    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        seed: int,
        batch_size: int,
        policy_kwargs: Dict[str, Any],
        epoch_ppo: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_eps: float = 0.2,
        coef_ent: float = 0.01,
        max_grad_norm: Optional[float] = None,
        fp16: bool = False,
        optim_kwargs: Optional[dict] = None,
        buffer_kwargs: Optional[Dict[str, Any]] = None,
        init_buffer: bool = True,
        init_models: bool = True,
        **kwargs,
    ):
        super().__init__(
            state_space,
            action_space,
            device,
            seed,
            gamma,
            max_grad_norm,
            fp16,
            batch_size,
            policy_kwargs,
            optim_kwargs,
            buffer_kwargs,
            init_buffer,
            init_models,
        )

        # learning rate scheduler.
        # TODO: add learning rate scheduler.
        # ? Is there one suitable for RL?

        """alpha_t = alpha_0 (1 - t/T)"""
        # schedule = lambda epoch: 1 - epoch/(self.param.evaluation['total_timesteps'] // self.batch_size)
        # self.scheduler_actor = optim.lr_scheduler.LambdaLR(self.optim_actor, schedule)
        # self.scheduler_critic = optim.lr_scheduler.LambdaLR(self.optim_critic, schedule)

        self.learning_steps_ppo = 0
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.coef_ent = coef_ent

    def is_update(self, step):
        return step % self.batch_size == 0

    def step(self, env, state, t, step):
        """Intereact with environment and store the transition."""
        t += 1
        action, log_pi = self.explore(state)
        next_state, reward, done, info = env.step(action)
        # TODO: may remove mask
        # * intuitively, mask make sence that agent keeps alive which is not done by env
        # ! mask = False if t == env._max_episode_steps else done

        data = {
            "obs": asarray_shape2d(state),
            "acts": asarray_shape2d(action),
            "rews": asarray_shape2d(reward),
            "dones": asarray_shape2d(done),
            "log_pis": asarray_shape2d(log_pi),
            "next_obs": asarray_shape2d(next_state),
        }

        # Store transition. not allow size larger than buffer capcity.
        self.buffer.store(data, truncate_ok=False)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        data = self.buffer.get()
        self.buffer.reset()
        train_logs = self.update_ppo(data)
        return train_logs

    def update_ppo(self, data: TensorDict):
        states, actions, next_states, log_pis = (
            data["obs"],
            data["acts"],
            data["next_obs"],
            data["log_pis"],
        )

        with th.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        rewards, dones = data["rews"], data["dones"]

        targets, gaes = calculate_gae(
            rewards, dones, values, next_values, self.gamma, self.gae_lambda
        )

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            loss_critic = self.update_critic(states, targets)
            loss_actor, pi_info = self.update_actor(states, actions, log_pis, gaes)

        # return log changes(key used for logging name).
        return {
            "actor_loss": loss_actor.item(),
            "critic_loss": loss_critic.item(),
            "approx_kl": pi_info["kl"].item(),
            "entropy": pi_info["ent"].item(),
            "clip_fraction": pi_info["cf"].item(),
            "pi_lr": self.lr_actor,
            "vf_lr": self.lr_critic,
            "learn_steps_ppo": self.learning_steps_ppo,
        }

    def update_critic(self, states, targets):

        self.optim_critic.zero_grad(set_to_none=self.optim_set_to_none)
        with autocast():
            loss_critic = (self.critic(states) - targets).pow_(2).mean()
        self.one_gradient_step(loss_critic, self.optim_critic, self.critic)
        return loss_critic

    def update_actor(self, states, actions, log_pis_old, gaes):
        log_pis = self.actor.evaluate_log_pi(states, actions)

        # * Since we bounded the mean action with tanh(), there is no analytical form of entropy
        # approximate entropy.
        approx_ent = -log_pis.mean()

        log_ratios = log_pis - log_pis_old
        ratios = (log_ratios).exp()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -th.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes
        loss_actor = th.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad(set_to_none=self.optim_set_to_none)
        with autocast():
            loss_actor_ent = loss_actor - self.coef_ent * approx_ent
        self.one_gradient_step(loss_actor_ent, self.optim_actor, self.actor)

        # Useful extra info
        """
        Calculate approximate form of reverse KL Divergence for early stopping.
        See issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        and Schulman blog: https://joschu.net/blog/kl-approx.html
        KL(q||p): (r-1) - log(r), where r = p(x)/q(x)
        """
        # ! Deprecated:
        # ! Naive version: approx_kl = (log_pi_old - log_pi).mean().item()
        # ! This is an unbiased estimator, but it has large variance.
        # ! Since it can take on negative values. (as opposed to the actual KL Divergence measure)
        with th.no_grad():
            approx_kl = ((ratios - 1) - log_ratios).mean()
            clipped = ratios.gt(1 + self.clip_eps) | ratios.lt(1 - self.clip_eps)
            clip_frac = th.as_tensor(clipped, dtype=th.float32).mean()
            pi_info = {"kl": approx_kl, "ent": approx_ent, "cf": clip_frac}
        return loss_actor, pi_info

    def save_models(self, save_dir):
        pass
