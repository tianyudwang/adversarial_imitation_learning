from typing import Union, Sequence

import numpy as np
import torch as th
from torch import nn
from torch.nn.utils import clip_grad_norm_


from ail.agents.base_agent import BaseAgent
from ail.common.math import normalize
from ail.common.type_alias import OPT, TensorDict, GymSpace
from ail.buffer.buffer_irl import RolloutBuffer
from ail.network.policies import StateIndependentPolicy
from ail.network.value import mlp_value
from ail.common.utils import asarray_shape2d


def calculate_gae(data, values, next_values, gamma, lambd, normal=True):
    rewards = data["rews"]
    dones = data["dones"]
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


class PPO(BaseAgent):
    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        seed: int,
        batch_size: int,
        lr_actor: float,
        lr_critic: float,
        units_actor: Sequence[int],
        units_critic: Sequence[int],
        epoch_ppo: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_eps: float = 0.2,
        coef_ent: float = 0.01,
        max_grad_norm: float = 0.5,
        optim_kwargs=None,
    ):
        super().__init__(state_space, action_space, device, seed, gamma)

        if optim_kwargs is None:
            optim_kwargs = {}

        # Rollout
        self.buffer = RolloutBuffer(
            capacity=batch_size,
            obs_shape=self.state_shape,
            act_shape=self.action_shape,
            device=self.device,
            with_reward=True,
            extra_shapes={"log_pis": (1,)},
            extra_dtypes={"log_pis": np.float32},
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            self.obs_dim,
            self.act_dim,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh(),
        ).to(self.device)

        # Critic.
        self.critic = mlp_value(
            self.obs_dim,
            self.act_dim,
            val_type="V",
            value_layers=units_critic,
            activation=nn.Tanh(),
        ).to(self.device)

        self.optim_cls = OPT[optim_kwargs.get("optim_cls", "adam").lower()]

        self.optim_actor = self.optim_cls(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = self.optim_cls(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.batch_size = batch_size
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, step):
        return step % self.batch_size == 0

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        # TODO: may remove mask
        mask = False if t == env._max_episode_steps else done

        data = {
            "obs": asarray_shape2d(state),
            "acts": asarray_shape2d(action),
            "rews": asarray_shape2d(reward),
            "dones": asarray_shape2d(mask),
            "log_pis": asarray_shape2d(log_pi),
            "next_obs": asarray_shape2d(next_state),
        }

        self.buffer.store(data, truncate_ok=False)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        data = self.buffer.get()
        self.buffer.reset()
        self.update_ppo(data)

    def update_ppo(self, data: TensorDict):
        states, actions, rewards, dones, log_pis, next_states = (
            data["obs"],
            data["acts"],
            data["rews"],
            data["dones"],
            data["log_pis"],
            data["next_obs"],
        )

        with th.no_grad():
            values = self.critic(data["obs"])
            next_values = self.critic(data["next_obs"])

        targets, gaes = calculate_gae(
            data, values, next_values, self.gamma, self.gae_lambda
        )

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets)
            self.update_actor(states, actions, log_pis, gaes)

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, gaes):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -th.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes
        loss_actor = th.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

    def save_models(self, save_dir):
        pass
