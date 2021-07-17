from typing import Union, Optional, Dict, Any
from copy import deepcopy
from itertools import chain

import torch as th
from torch.cuda.amp import autocast

from ail.agents.rl_agent.base import OffPolicyAgent
from ail.common.type_alias import TensorDict, GymEnv, GymSpace
from ail.common.pytorch_util import asarray_shape2d, count_vars, disable_gradient, soft_update

from ail.network.policies import StateDependentPolicy
from ail.network.value import mlp_value


class SAC(OffPolicyAgent):
    
    """
    Soft Actor-Critic (SAC)
    """
    
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
        
        # Set target param equal to main param. or just use deep copy instead.
        soft_update(self.critic_target, self.critic, 1.0)
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        disable_gradient(self.critic_target)

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = chain(self.critic, self.critic_target)

        
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(
            count_vars(module)
            for module in [self.actor, self.critic, self.critic_target]
        )

        # Entropy regularization coefficient(alpha) explicitly controls explore-exploit tradeoff.
        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        self.alpha = 1.0  # * Default initial value of ent_coef when learned
        self.lr_alpha = lr_alpha
        
        # Enforces an entropy constraint by varying alpha over the course of training.
        # We optimize log(alpha) because alpha should be always bigger than 0.
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        self.log_alpha = th.zeros(1, device=self.device, requires_grad=True)
        
        # Target entropy is -|A|.
        self.target_entropy = -float(self.action_shape[0])
        # TODO: test is this same as ?
        # self.target_entropy = -np.prod(self.action_shape).astype(np.float32)

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
        """
        Intereact with environment and store the transition.
        
        A trick to improve exploration at the start of training (for a fixed number of steps)
        Agent takes actions which are sampled from a uniform random distribution over valid actions.
        After that, it returns to normal SAC exploration.
        """
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
        mask = False if t == env._max_episode_steps else done  # TODO: Test this. or make it optional.

        data = {
            "obs": asarray_shape2d(state),
            "acts": asarray_shape2d(action),
            "rews": asarray_shape2d(reward),
            "dones": asarray_shape2d(done), # * or mask
            # "log_pis": asarray_shape2d(log_pi), # * not store log_pi for pure SAC
            "next_obs": asarray_shape2d(next_state),
        }

        # Store transitions (s, a, r, s',d).
        # * ALLOW size larger than buffer capcity.
        self.buffer.store(data, truncate_ok=True)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        
        # Random uniform sampling a batch of transitions, B = {(s, a, r, s',d)}, from the buffer.
        data = self.buffer.sample(self.batch_size)
        train_logs = self.update_sac(self, data)
        return train_logs

    def update_sac(self, data: TensorDict):
        """
        Update the actor and critic and target network as well.
        :param data: a batch of randomly sampled transitions
        """
        states, actions, rewards, dones, next_states = (
            data["obs"],
            data["acts"],
            data["rews"],
            data["dones"],
            data["next_obs"],
        )
        
        self.update_critic(states, actions, rewards, dones, next_states)
        
        # TODO: Test this
        # # Freeze Q-networks so you don't waste computational effort 
        # # computing gradients for them during the policy learning step.
        # for p in self.q_params:
        #     p.requires_grad = False
        self.update_actor(states)
        
        # TODO: Test this
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        # for p in self.q_params:
        #     p.requires_grad = True
        
        # Update target networks by polyak averaging.
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states):
        """Update Q-functions by one step of gradient"""
        self.optim_critic.zero_grad(set_to_none=self.opti_set_None)

        curr_qs1, curr_qs2 = self.critic(states, actions)

        # Bellman backup for Q functions
        with th.no_grad():
            # Target actions come from *current* policy
            # whereas by contrast, r and s' should come from the replay buffer.
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            # clipped double-Q trick and takes the minimum Q-value between the two Q approximators.
            next_qs = th.min(next_qs1, next_qs2) - self.alpha * log_pis
        
        # Target is given by: r + gamma(1-d) * ( Q(s', a') - alpha log pi(a'|s') )
        target_qs = rewards + self.gamma * (1.0 - dones) * next_qs
        
        # TODO: (Yifan) modify this using one_gradient_step after test.
        #  Mean-squared Bellman error (MSBE) loss 
        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states: th.Tensor):
        """Update policy by one step of gradient"""
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
        """update the target network by polyak averaging."""
        # * here tau = (1 - polyak)
        soft_update(self.critic_target, self.critic, self.tau)

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # TODO: (Yifan) implement this.
        # Only save actor to reduce workloads
        # th.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
