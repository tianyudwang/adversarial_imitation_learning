import os
import argparse
from datetime import datetime
import yaml

try:
    from icecream import install  # noqa
    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noq

from ail.agents.rl_agent.ppo import PPO
from ail.trainer.rl_trainer import RL_Trainer


def run(args):

    algo_kwargs = dict(
        device=args.device,
        seed=args.seed,
        batch_size=args.rollout_length,
        gamma=0.99,
        max_grad_norm=None,
        gae_lambda=0.97,
        coef_ent=0.01,
        lr_actor=1e-4,
        lr_critic=1e-4,
        units_actor=(64, 64),
        units_critic=(64, 64),
        epoch_ppo=50,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("runs", args.env_id, args.algo, f"seed{args.seed}-{time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "hyperparams.yaml"), "w") as f:
        yaml.dump(algo_kwargs, f)

    trainer = RL_Trainer(
        num_steps=args.num_steps,
        env=args.env_id,
        algo=PPO,
        algo_kwargs=algo_kwargs,
        env_kwargs=None,
        max_ep_len=None,
        seed=args.seed,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        save_freq=50_000,
        log_dir=log_dir,
        log_interval=10_000,
        verbose=args.verbose,
        use_wandb=args.use_wandb,
    )
    trainer.run_training_loop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_length", type=int, default=1_000)
    p.add_argument("--algo", type=str, default="ppo")
    p.add_argument("--num_steps", type=int, default=1 * 1e6)
    p.add_argument("--eval_interval", type=int, default=5 * 1e3)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--env_id", type=str, default="InvertedPendulum-v2")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", type=int, default=2)
    p.add_argument("--use_wandb", "-wb", action="store_true", default=False)
    args = p.parse_args()

    if not isinstance(args.num_steps, int):
        try:
            args.num_steps = int(args.num_steps)
        except ValueError:
            raise ValueError("Please provide integer for --num_steps")

    args.device = "cuda" if args.cuda else "cpu"

    run(args)
