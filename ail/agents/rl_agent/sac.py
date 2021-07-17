from typing import Union, Optional, Dict, Any

import torch as th
from torch.cuda.amp import autocast

from ail.agents.rl_agent.base import OffPolicyAgent
from ail.common.type_alias import TensorDict, GymEnv, GymSpace
from ail.common.pytorch_util import asarray_shape2d, disable_gradient, soft_update

from ail.network.policies import StateDependentPolicy
from ail.network.value import mlp_value


class SAC(OffPolicyAgent):
    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        seed: int,
        batch_size: int,
        buffer_size: int,
        policy_kwargs: Dict[str, Any],
        lr_alpha: float,
        start_steps: int = 10_000,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_grad_norm: Optional[float] = None,
        fp16: bool = False,
        optim_kwargs=None,
        buffer_kwargs: Optional[Dict[str, Any]] = None,
        init_buffer: bool = True,
        init_models: bool = True,
    ):
        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            gamma,
            max_grad_norm,
            batch_size,
            buffer_size,
            policy_kwargs,
            optim_kwargs,
            buffer_kwargs,
            init_buffer,
            init_models,
        )
        
        # TODO: (Yifan) Build the model inside off policy class latter.
        # Actor.
        self.actor = StateDependentPolicy(
            self.obs_dim, self.act_dim, self.units_actor, self.hidden_activation
        ).to(self.device)

        # Critic.
        self.critic = mlp_value(
            self.obs_dim,
            self.act_dim,
            self.units_critic,
            self.hidden_activation,
            self.critic_type,
        ).to(self.device)

        self.critic_target = (
            mlp_value(
                self.obs_dim,
                self.act_dim,
                self.units_critic,
                self.hidden_activation,
                self.critic_type,
            )
            .to(self.device)
            .eval()
        )

        soft_update(self.critic_target, self.critic, 1.0)
        disable_gradient(self.critic_target)

        # Entropy coefficient.
        self.alpha = 1.0
        self.lr_alpha = lr_alpha
        # We optimize log(alpha) because alpha should be always bigger than 0.
        self.log_alpha = th.zeros(1, device=self.device, requires_grad=True)
        # Target entropy is -|A|.
        self.target_entropy = -float(self.action_shape[0])

        # Config optimizer
        self.optim_actor = self.optim_cls(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = self.optim_cls(self.critic.parameters(), lr=self.lr_critic)
        self.optim_alpha = self.optim_cls([self.log_alpha], lr=self.lr_alpha)

        # Other algo params.
        self.start_steps = start_steps
        self.tau = tau

    def is_update(self, step):
        return step >= max(self.start_steps, self.batch_size)

    def step(self, env: GymEnv, state: th.Tensor, t: int, step: Optional[int] = None):
        """Intereact with environment and store the transition."""
        t += 1

        if step <= self.start_steps:
            # Random uniform sampling.
            action = env.action_space.sample()
            # TODO: (Yifan) may enable this in AIRL, test the case without it.
            # log_pi = self.actor.evaluate_log_pi(
            #     th.as_tensor(state, dtype=th.float, device=self.device),
            #     th.as_tensor(action, dtype=th.float, device=self.device)
            # )
        else:
            action, log_pi = self.explore(state)

        next_state, reward, done, info = env.step(action)
        mask = False if t == env._max_episode_steps else done  # TODO: Test this.
        self.buffer.append(state, action, reward, mask, next_state, log_pi)

        data = {
            "obs": asarray_shape2d(state),
            "acts": asarray_shape2d(action),
            "rews": asarray_shape2d(reward),
            "dones": asarray_shape2d(done),
            # "log_pis": asarray_shape2d(log_pi),
            "next_obs": asarray_shape2d(next_state),
        }

        # Store transition.
        # * ALLOW size larger than buffer capcity.
        self.buffer.store(data, truncate_ok=True)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        data = self.buffer.sample(self.batch_size)
        train_logs = self.update_sac(self, data)
        return train_logs

    def update_sac(self, data: TensorDict):
        states, actions, rewards, dones, next_states = (
            data["obs"],
            data["acts"],
            data["rews"],
            data["dones"],
            data["next_obs"],
        )

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states):
        self.optim_critic.zero_grad(set_to_none=self.opti_set_None)

        curr_qs1, curr_qs2 = self.critic(states, actions)

        with th.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = th.min(next_qs1, next_qs2) - self.alpha * log_pis

        target_qs = rewards + (1.0 - dones) * self.gamma * next_qs
        
        # TODO: (Yifan) modify this using one_gradient_step after test.
        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states: th.Tensor):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = self.alpha * log_pis.mean() - th.min(qs1, qs2).mean()

        # TODO: (Yifan) modify this using one_gradient_step after test.
        self.optim_actor.zero_grad(set_to_none=self.opti_set_None)
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        entropy = -log_pis.detach_().mean()
        loss_alpha = -self.log_alpha * (self.target_entropy - entropy)

        self.optim_alpha.zero_grad(set_to_none=self.opti_set_None)
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()

        with th.no_grad():
            self.alpha = self.log_alpha.exp().item()

    def update_target(self):
        soft_update(self.critic_target, self.critic, self.tau)

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # TODO: (Yifan) implement this.
        # Only save actor to reduce workloads
        # th.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
