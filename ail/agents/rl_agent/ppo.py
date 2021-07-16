from typing import Union, Sequence, Optional

import torch as th
from torch import nn
from torch.cuda.amp import autocast

from ail.agents.rl_agent.base import RLAgent
from ail.common.math import normalize
from ail.common.type_alias import OPT, TensorDict, GymSpace, extra_shapes, extra_dtypes
from ail.buffer.buffer_irl import RolloutBuffer
from ail.network.policies import StateIndependentPolicy
from ail.network.value import mlp_value
from ail.common.utils import asarray_shape2d

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


class PPO(RLAgent):
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
        max_grad_norm: Optional[float] = None,
        optim_kwargs: Optional[dict] = None,
    ):
        super().__init__(state_space, action_space, device, seed, gamma, max_grad_norm)

        if optim_kwargs is None:
            optim_kwargs = {}

        # Rollout
        self.buffer = RolloutBuffer(
            capacity=batch_size,
            obs_shape=self.state_shape,
            act_shape=self.action_shape,
            device=self.device,
            with_reward=True,
            extra_shapes={"log_pis": extra_shapes.log_pis},
            extra_dtypes={"log_pis": extra_dtypes.log_pis},
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

        # Orthogonal Initialize.
        self.weight_initiation()

        # Learning rate.
        
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        # learning rate scheduler.
        # TODO: add learning rate scheduler.
        # ? Is there one suitable for RL?
        
        '''alpha_t = alpha_0 (1 - t/T)'''
        # schedule = lambda epoch: 1 - epoch/(self.param.evaluation['total_timesteps'] // self.batch_size)
        # self.scheduler_actor = optim.lr_scheduler.LambdaLR(self.optim_actor, schedule)
        # self.scheduler_critic = optim.lr_scheduler.LambdaLR(self.optim_critic, schedule)
        
        self.optim_cls = OPT[optim_kwargs.get("optim_cls", "adam").lower()]
        self.optim_actor = self.optim_cls(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = self.optim_cls(self.critic.parameters(), lr=self.lr_critic)
        self.optim_set_to_none = optim_kwargs.get("optim_set_to_none", False)

        self.learning_steps_ppo = 0
        self.batch_size = batch_size
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
        next_state, reward, done, _ = env.step(action)
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
