from typing import Union, Optional, Dict, Any
from copy import deepcopy

import numpy as np
import torch as th
from torch.cuda.amp import autocast

from ail.agents.rl_agent.core import OffPolicyAgent
from ail.common.math import soft_update
from ail.common.pytorch_util import (
    asarray_shape2d,
    count_vars,
    disable_gradient,
    enable_gradient,
    obs_as_tensor
)
from ail.common.type_alias import TensorDict, GymEnv, GymSpace
from ail.network.policies import StateDependentPolicy
from ail.network.value import mlp_value


class SAC(OffPolicyAgent):

    """
    Soft Actor-Critic (SAC)
    Paper: https://arxiv.org/abs/1801.01290
    
    :param state_space: state space
    :param action_space: action space
    :param device: PyTorch device to which the values will be converted.
    :param seed: random seed.
    :param batch_size: size of the batch.
    :param buffer_size: size of the buffer.
    :param policy_kwargs: arguments to be passed to the policy on creation.
        e.g. : {
            pi: [64, 64],
            vf: [64, 64],
            activation: 'relu',
            lr_actor: 3e-4,
            lr_critic: 3e-4
            critic_type="twin",
            }
    :param lr_alpha: learning rate for the entropy coefficient.
    :param start_steps: how many steps of the model to collect transitions before learning starts.
    :param num_gradient_steps: How many gradient steps to do after each rollout
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param log_alpha_init: init value of log_alpha.
    :param gamma: discount factor.
    :param tau: soft update coefficient. ("Polyak update", between 0 and 1)
    :param max_grad_norm: Maximum norm for the gradient clipping.
    :param fp16: Whether to use float16 mixed precision training.
    :optim_kwargs: arguments to be passed to the optimizer.
        eg. : {
            "optim_cls": adam,
            "optim_set_to_none": True, # which set grad to None instead of zero.
            }
    :param buffer_kwargs: arguments to be passed to the buffer.
        eg. : {
            with_reward:True,
            extra_data:["log_pis"]
            }
    :param init_buffer: Whether to create the buffer during initialization.
    :param init_models: Whether to create the models during initialization.
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
        start_steps: int,
        num_gradient_steps: int = 1,
        target_update_interval: int = 1,
        log_alpha_init: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_grad_norm: Optional[float] = None,
        fp16: bool = False,
        optim_kwargs=None,
        buffer_kwargs: Optional[Dict[str, Any]] = None,
        init_buffer: bool = True,
        init_models: bool = True,
        **kwargs,
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

        # Set target param equalto main param. Using deep copy instead.
        self.critic_target = deepcopy(self.critic).to(self.device).eval()

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        disable_gradient(self.critic_target)

        # Entropy regularization coefficient (Inverse of the reward scale)
        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        # self.alpha = 1.0  # * Default initial value of ent_coef when learned
        self.lr_alpha = lr_alpha

        # Enforces an entropy constraint by varying alpha over the course of training.
        # We optimize log(alpha) because alpha should be always bigger than 0.
        self.log_alpha = th.log(
            th.ones(1, device=self.device) * log_alpha_init
        ).requires_grad_(True)

        # Target entropy is -|A|.
        self.target_entropy = -np.prod(self.action_shape).astype(np.float32)

        # Config optimizer
        self.optim_actor = self.optim_cls(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = self.optim_cls(self.critic.parameters(), lr=self.lr_critic)
        self.optim_alpha = self.optim_cls([self.log_alpha], lr=self.lr_alpha)

        # Other algo params.
        self.start_steps = start_steps
        self.num_gradient_steps = num_gradient_steps
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.one = th.ones(1, device=self.device)  # a constant for quick soft update

    def __repr__(self):
        return "SAC"    
    
    def info(self):
        """
        Count variables.
        protip: try to get a feel for how different size networks behave!
        """
        return {
            module: count_vars(module)
            for module in [self.actor, self.critic, self.critic_target]
        }

    def is_update(self, step: int):
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
            action, log_pi = self.explore(obs_as_tensor(state, self.device))

        next_state, reward, done, info = env.step(action)
        mask = False if t == env._max_episode_steps else done

        data = {
            "obs": asarray_shape2d(state),
            "acts": asarray_shape2d(action),
            "rews": asarray_shape2d(reward),
            "dones": asarray_shape2d(mask),  # ? or done?
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

    def update(self) -> Dict[str, Any]:
        """
        A general Roadmap
        for each gradient step do
            -Sample transition from replay buffer
            -Update the Q-function parameters
            -Update policy weights
            -Adjust temperature
            -Update target network weights every n gradient steps
        end for
        """
        for gradient_step in range(self.num_gradient_steps):
            self.learning_steps += 1    
            # Random uniform sampling a batch of transitions, B = {(s, a, r, s',d)}, from buffer.
            replay_data = self.buffer.sample(self.batch_size)
            train_logs = self.update_sac(replay_data, gradient_step)
        return train_logs

    def update_sac(self, data: TensorDict, gradient_step: int) -> TensorDict:
        """
        Update the actor and critic and target network as well.
        :param data: a batch of randomly sampled transitions
        :return train_logs: dict of training logs
        """
        states, actions, rewards, dones, next_states = (
            data["obs"],
            data["acts"],
            data["rews"],
            data["dones"],
            data["next_obs"],
        )

        # Sample new actions and log probs from the current policy
        actions_new, log_pis_new = self.actor.sample(states)

        # Update and obtain the entropy coefficient.
        self.update_alpha(log_pis_new)

        # First run one gradient descent step for Q1 and Q2
        self.update_critic(states, actions, rewards, dones, next_states)

        # Freeze Q-networks so we don't waste computational effort
        # computing gradients for them during the policy learning step.
        disable_gradient(self.critic)
        self.update_actor(states, actions_new, log_pis_new)
        enable_gradient(self.critic)

        # Update target networks by polyak averaging.
        if gradient_step % self.target_update_interval == 0:
            self.update_target()
        return {}  # TODO: add train logs

    def update_alpha(self, log_pis_new):
        """
        Optimize entropy coefficient (alpha)
        L(alpha) = E_{at ∼ pi_t} [−alpha * log pi(a_t |s_t ) − alpha * H].        
        ent_loss = E[-alpha * (log_pis) - alpha * target_ent]
                 = E [-alpha (log_pis + target_ent)]
                 = (-alpha * (log_pis + target_ent)).mean()
                 As log do preserve order
                 = (-log_alpha * (target_ent + log_ent).mean()
        As discussed in https://github.com/rail-berkeley/softlearning/issues/37
        """
        self.optim_alpha.zero_grad(set_to_none=self.optim_set_to_none)
        
        with autocast(enabled=self.fp16):
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            self.alpha = th.exp(self.log_alpha.detach())
            loss_alpha = -(
                self.log_alpha * (self.target_entropy + log_pis_new).detach()
            ).mean()
        self.one_gradient_step(loss_alpha, self.optim_alpha, self.log_alpha)

        # loss_alpha.backward()
        # self.optim_alpha.step()

    def update_critic(self, states, actions, rewards, dones, next_states):
        """Update Q-functions by one step of gradient"""
        self.optim_critic.zero_grad(set_to_none=self.optim_set_to_none)

        with autocast(enabled=self.fp16):
            # Get current Q-values estimation for each critic network
            # * using action from the replay buffer
            curr_qs1, curr_qs2 = self.critic(states, actions)

            # Bellman backup for Q functions
            with th.no_grad():
                # * Target actions come from *current* policy
                # * whereas by contrast, r and s' should come from the replay buffer.
                next_actions, next_log_pis = self.actor.sample(next_states)
                next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
                # clipped double-Q trick and takes the minimum Q-value between two Q approximators.
                next_qs = th.min(next_qs1, next_qs2) - self.alpha * next_log_pis

                # Target (TD error + entropy term):
                # r + gamma(1-d) * ( Q(s', a') - alpha log pi(a'|s') )
                target_qs = rewards + self.gamma * (1.0 - dones) * next_qs

            #  Mean-squared Bellman error (MSBE) loss
            loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
            loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()
            loss_critic = loss_critic1 + loss_critic2
        self.one_gradient_step(loss_critic, self.optim_critic, self.critic)

        # (loss_critic1 + loss_critic2).backward()
        # self.optim_critic.step()

    def update_actor(
        self, states: th.Tensor, actions_new: th.Tensor, log_pis_new: th.Tensor
    ):
        """
        Update policy by one step of gradient
        :param states: states from the replay buffer
        :param actions_new: sampled actions from the current policy
        :param log_pis_new: log_probs of the actions sampled from the current policy
        """

        self.optim_actor.zero_grad(set_to_none=self.optim_set_to_none)
        with autocast(enabled=self.fp16):
            qs1, qs2 = self.critic(states, actions_new)
            qs = th.min(qs1, qs2)
            loss_actor = (self.alpha * log_pis_new - qs).mean()
        self.one_gradient_step(loss_actor, self.optim_actor, self.actor)

        # loss_actor.backward()
        # self.optim_actor.step()

    def update_target(self):
        """update the target network by polyak averaging."""
        # * here tau = (1 - polyak)
        soft_update(self.critic_target, self.critic, self.tau, self.one)

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # TODO: (Yifan) implement this.
        # Only save actor to reduce workloads
        # th.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
