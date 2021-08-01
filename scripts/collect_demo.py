import argparse
import pathlib
from pprint import pprint

import yaml
import gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import SAC, PPO, HER

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


SB3_ALGO = {
    "sb3_ppo": PPO,
    "sb3_sac": SAC,
    "sb3_her": HER,
}


def collect_demo(
    env, algo, buffer_size: int, device, seed=0, render=False, sb3_model=None, save_dir=None
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
    toral_ep_len = []

    state = env.reset()
    t = 0
    episode_return = 0.0
    num_episodes = 0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if sb3_model is not None:
            action, _ = sb3_model.predict(
                th.as_tensor(state, dtype=th.float32), deterministic=True
            )
        elif algo is not None:
            action = algo.exploit(th.as_tensor(state, dtype=th.float32), scale=False)
        else:
            raise ValueError("Please provide either sb3_model or cumstom algo")

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
            toral_ep_len.append(t)
            state = env.reset()
            t = 0
            episode_return = 0.0

        state = next_state

    print(
        f"Mean return of the expert is "
        f"{np.mean(total_return):.3f} +/- {np.std(total_return):.3f}"
    )
    info = {
        "return_mean": np.mean(total_return).item(),
        "return_std": np.std(total_return).item(),
    }

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    ax[0].plot(total_return)
    ax[1].plot(toral_ep_len)
    ax[0].set_title("Return")
    ax[1].set_title("Episode Length")
    fig.supxlabel("Time Step")
    plt.tight_layout()
    fig.savefig(save_dir / "return.png")
    plt.show()
    return demo_buffer, info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weight", type=str, default="")
    p.add_argument(
        "--env_id",
        type=str,
        required=True,
        choices=["InvertedPendulum-v2", "HalfCheetah-v2", "Hopper-v3"],
        help="Envriment to interact with",
    )
    p.add_argument(
        "--algo",
        type=str, 
        choices=["ppo","sac","sb3_ppo","sb3_sac"]
    )
    p.add_argument("--hidden_size", "-hid", type=int, default=64)
    p.add_argument("--layers", "-l", type= int, default=2)
    p.add_argument("--buffer_size", type=int, default=1_000 * 11)
    p.add_argument("--render", "-r", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # run only on cpu for testing
    device = th.device("cpu")

    path = pathlib.Path(__file__).parent.resolve()
    print(path,'\n')
    
    demo_dir = path.parent / "ail" / "rl-trained-agents" / args.env_id

    if not args.weight:
        if args.algo.startswith("sb3"):
            args.weight = demo_dir / args.algo[4:] / f"{args.env_id}_sb3"
        else:
            args.weight = demo_dir / args.algo / f"{args.env_id}_actor.pth"

    print(args.weight, '\n')
    
    if args.algo.startswith("sb3"):
        sb3_model = SB3_ALGO[args.algo].load(args.weight)
        algo = None
    else:
        dummy_env = gym.make(args.env_id)
        sb3_model = None
        algo = ALGO[args.algo].load(
            path=args.weight,
            policy_kwargs={"pi": [args.hidden_size] * args.layers},
            env=dummy_env,
            device=device,
            seed=args.seed,
        )

    print(f"weight_dir: {args.weight}\n")
    
    save_dir = path / "transitions" / args.env_id
    buffer, ret_info = collect_demo(
        env=args.env_id,
        algo=algo,
        buffer_size=args.buffer_size,
        device=device,
        render=args.render,
        seed=args.seed,
        sb3_model=sb3_model,
        save_dir=save_dir,
    )
     
    data_path = save_dir / f"{args.algo}_size{args.buffer_size}"
    print(f"\nSaving to {data_path}")
    # * default save with .npz
    buffer.save(data_path)


    data = dict(np.load(f"{data_path}.npz"))
    obs = data["obs"]
    act = data["acts"]
    dones = data["dones"],
    next_obs = data["next_obs"]
    
    obs_norm = (-1 <= obs).all() and (obs <= 1).all()
    act_norm = (-1 <= act).all() and (act <= 1).all()
    next_obs_norm = (-1 <= next_obs).all() and (next_obs <= 1).all()
    
    
    info ={
        "action in [-1, 1]": act_norm.item(),
        "observation in [-1, 1]": obs_norm.item(),
        "next_observation in [-1, 1]": next_obs_norm.item()
    }
    pprint(info)
    info.update(ret_info)
    # Saving hyperparams to yaml file.
    with open(save_dir / "info.yaml", "w") as f:
        yaml.dump(info, f)