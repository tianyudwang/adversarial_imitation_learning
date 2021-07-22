import gym
import numpy as np
import torch as th

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from ail.buffer import RolloutBuffer, ReplayBuffer

env_id = "HalfCheetah-v2"
env = gym.make(env_id)

print(env.observation_space)
print(env.action_space)


state, done = env.reset(), False
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)


data = dict(np.load(f"../../scripts/transitions/{env_id}/size11000.npz"))
ic(data)
buffer = ReplayBuffer.from_data(data, device="cpu", with_reward=False)

trajectory = buffer.get()
for k, v in trajectory.items():
    assert v.shape[0] == 11000
    print(k, v.shape)
print("\n")
samples = buffer.sample(10)
for k, v in samples.items():
    assert v.shape[0] == 5000
    print(k, v.shape)
