import numpy as np
from gym import ActionWrapper
from gym.spaces.box import Box


# Borrow from (https://github.com/openai/gym/tree/ee5ee3a4a5b9d09219ff4c932a45c4a661778cd7/gym/wrappers)
class RescaleAction(ActionWrapper):
    """Rescales the continuous action space of the environment to a range [a,b]."""

    def __init__(self, env, a: float, b: float):
        assert isinstance(
            env.action_space, Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
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

    def reverse_action(self, action):
        raise NotImplementedError()


# Borrow from (https://github.com/openai/gym/tree/ee5ee3a4a5b9d09219ff4c932a45c4a661778cd7/gym/wrappers)
class ClipAction(ActionWrapper):
    """
    Clips Box actions to be within the high and low bounds of the action space.
    This is a standard transformation applied to environments with continuous action spaces
    to keep the action passed to the environment within the specified bounds.
    """

    def __init__(self, env):
        assert isinstance(env.action_space, Box)
        max_episode_steps = env.spec.max_episode_steps
        super(ClipAction, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        
    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)

    def reverse_action(self, action):
        raise NotImplementedError()


class NormalizeAction(RescaleAction):
    """
    Normalize continuous action space to be scaled to [-1, 1]
        (Assuming symmetric actions space)
    """

    def __init__(self, env):
        super(NormalizeAction, self).__init__(env, a=-1, b=1)
        self._max_episode_steps = env._max_episode_steps  # noqa

    def reverse_action(self, action):
        raise NotImplementedError()
