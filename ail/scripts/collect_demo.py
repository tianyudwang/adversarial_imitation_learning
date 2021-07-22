import argparse
import os

import numpy as np
import torch as th
from tqdm import tqdm
from stable_baselines3 import A2C, SAC, PPO, HER

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from ail.agents import ALGO
from ail.buffer import ReplayBuffer
from ail.common.env_utils import maybe_make_env
from ail.common.utils import set_random_seed
from ail.common.pytorch_util import asarray_shape2d


def collect_demo(
    env, algo, buffer_size: int, device, seed=0, render=False, sb3_model=None
):
    env = maybe_make_env(env, tag="Expert", verbose=2)
    env.seed(seed)
    set_random_seed(seed)

    demo_buffer = ReplayBuffer(
        capacity=buffer_size,
        device=device,
        obs_shape=env.observation_space.shape,
        act_shape=env.action_space.shape,
        with_reward=False,
    )

    total_return = []
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if sb3_model is not None:
            action, _ = sb3_model.predict(state, deterministic=True)
        else:
            action = algo.exploit(state)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done

        data = {
            "obs": asarray_shape2d(state),
            "acts": asarray_shape2d(action),
            "dones": asarray_shape2d(mask),
            "next_obs": asarray_shape2d(next_state),
        }

        demo_buffer.store(transitions=data, truncate_ok=True)
        episode_return += reward

        if render:
            try:
                env.render()
            except KeyboardInterrupt:
                pass

        if done:
            num_episodes += 1
            total_return.append(episode_return)
            state = env.reset()
            t = 0
            episode_return = 0.0

        state = next_state

    print(
        f"Mean return of the expert is "
        f"{np.mean(total_return):.3f} +/- {np.std(total_return):.3f}"
    )
    return demo_buffer


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weight", type=str, default="")
    p.add_argument("--env_id", type=str, default="HalfCheetah-v2")
    p.add_argument("--buffer_size", type=int, default=1_000 * 11)
    p.add_argument("--algo", type=str, default="")
    p.add_argument("--render", "-r", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sb3_ALGO = {
        "sb3_ppo": PPO,
        "sb3_a2c": A2C,
        "sb3_sac": SAC,
        "sb3_her": HER,
    }

    if not args.weight:
        if args.algo.startswith("sb3"):
            args.weight = f"../rl-trained-agents/{args.env_id}"
        else:
            args.weight = f"../rl-trained-agents/{args.env_id}.pth"

    if args.algo.startswith("sb3"):
        print(args.weight)
        sb3_model = sb3_ALGO[args.algo].load(args.weight)
        algo = None
    else:
        sb3_model = None
        algo = ALGO[args.algo].load()  # TODO implement loading of custom models

    buffer = collect_demo(
        env=args.env_id,
        algo=algo,
        buffer_size=args.buffer_size,
        device=th.device("cuda" if args.cuda else "cpu"),
        render=args.render,
        seed=args.seed,
        sb3_model=sb3_model,
    )

    save_dir = os.path.join("transitions", args.env_id, f"size{args.buffer_size}")
    print(f"Saving to {save_dir}")
    buffer.save(save_dir)  # default save with .npz
