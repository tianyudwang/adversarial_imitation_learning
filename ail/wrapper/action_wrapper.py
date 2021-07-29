import numpy as np
from gym import ActionWrapper
from gym.spaces.box import Box


# Borrow from (https://github.com/openai/gym/tree/ee5ee3a4a5b9d09219ff4c932a45c4a661778cd7/gym/wrappers)
class RescaleBoxAction(ActionWrapper):

    """
    Rescales the continuous action space of the environment to a range [a,b].
    Note: This is will not rescale the action back to original action space.
    Example::
        >>> RescaleAction(env, a, b).action_space == Box(a,b)
        True
    """

    def __init__(self, env, a, b):
        if not isinstance(env.action_space, Box):
            raise TypeError(f"expected Box action space, got {env.action_space}")
        assert np.less_equal(a, b).all(), (a, b)
        super().__init__(env)
        self.a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
        self.b = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
        self.action_space = Box(
            low=a, high=b, shape=env.action_space.shape, dtype=env.action_space.dtype
        )

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * ((action - self.a) / (self.b - self.a))
        action = np.clip(action, low, high)
        return action


# Borrow from (https://github.com/openai/gym/tree/ee5ee3a4a5b9d09219ff4c932a45c4a661778cd7/gym/wrappers)
class ClipBoxAction(ActionWrapper):
    """
    Clips Box actions to be within the high and low bounds of the action space.
    This is a standard transformation applied to environments with continuous action spaces
    to keep the action passed to the environment within the specified bounds.
    """

    def __init__(self, env):
        assert isinstance(env.action_space, Box)
        max_episode_steps = env.spec.max_episode_steps
        super().__init__(env)
        self._max_episode_steps = max_episode_steps

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


class NormalizeBoxAction(ActionWrapper):
    """Rescale the action space of the environment."""

    def __init__(self, env):
        if not isinstance(env.action_space, Box):
            raise ValueError(f"env {env} does not use spaces.Box.")
        super().__init__(env)
        try:
            self._max_episode_steps = env.max_episode_steps
        except AttributeError:
            self._max_episode_steps = env.spec.max_episode_steps

    def action(self, action):
        # rescale the action (MinMaxScaler)
        low, high = self.env.action_space.low, self.env.action_space.high
        scaled_action = low + (action + 1.0) * (high - low) / 2.0
        scaled_action = np.clip(scaled_action, low, high)
        return scaled_action

    def reverse_action(self, scaled_action):
        low, high = self.env.action_space.low, self.env.action_space.high
        action = (scaled_action - low) * 2.0 / (high - low) - 1.0
        return action
