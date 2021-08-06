import math

import gym
import numpy as np

from ail.common.running_stats import RunningMeanStd
from ail.common.type_alias import GymStepReturn

# Borrow and modified
# From: https://github.com/DLR-RM/stable-baselines3/blob/3845bf9f3209173195f90752e341bbc45a44571b/stable_baselines3/common/vec_env/vec_normalize.py#L13
class VecNormalize(gym.Wrapper):
    """
    A moving average, normalizing wrapper for gym environment.

    :param venv: the gym environment to wrap
    :param norm_obs: Whether to normalize observation or not (default: True)
    :param norm_rew: Whether to normalize rewards or not (default: True)
    :param clip_obs: Max absolute value for observation
    :param clip_reward: Max value absolute for discounted reward
    :param gamma: discount factor
    :param epsilon: To avoid division by zero
    """

    __slots__ = ["ret", "ret_rms", "ob_rms", "gamma", "epsilon", "clip_obs", "clip_rew"]

    def __init__(
        self,
        env,
        norm_obs: bool = True,
        norm_rew: bool = True,
        clip_obs: float = float("inf"),
        clip_rew: float = float("inf"),
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        assert isinstance(
            env.observation_space, gym.spaces.Box
        ), "VecNormalize only support `gym.spaces.Box` observation spaces"
        params = {clip_obs, clip_rew, gamma, epsilon}
        for param in params:
            assert isinstance(param, float)

        super().__init__(env)
        self.obs_rms = (
            RunningMeanStd(shape=self.observation_space.shape) if norm_obs else None
        )
        self.ret_rms = RunningMeanStd(shape=(1,)) if norm_rew else None
        self.clip_obs = clip_obs
        self.clip_rew = clip_rew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action: np.ndarray) -> GymStepReturn:
        obs, rews, dones, infos = self.env.step(action)
        obs = self._normalize_obs(obs)

        infos["real_reward"] = rews
        self.ret = self.ret * self.gamma + rews
        rews = self._normalize_rews(rews)
        self.ret = self.ret * (1 - float(dones))
        return obs, rews, dones, infos

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms:
            self.obs_rms.update(obs)
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
            if not math.isinf(self.clip_obs):
                np.clip(obs, -self.clip_obs, self.clip_obs, out=obs)
        return obs

    def _normalize_rews(self, rews: np.ndarray) -> np.ndarray:
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = rews / np.sqrt(self.ret_rms.var + self.epsilon)
            if not math.isinf(self.clip_rew):
                np.clip(rews, -self.clip_rew, self.clip_rew, out=rews)
        return rews

    def reset(self) -> None:
        self.ret = np.zeros(())
        obs = self.env.reset()
        return self._normalize_obs(obs)

    @property
    def _max_episode_steps(self) -> int:
        return self.env._max_episode_steps
