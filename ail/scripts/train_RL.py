import os
import sys
import argparse
from datetime import datetime

import yaml
import torch as th

from ail.trainer import Trainer

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

try:
    from dotenv import load_dotenv, find_dotenv  # noqa

    load_dotenv(find_dotenv())

except ImportError:
    pass


def CLI():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_id",
        type=str,
        default="InvertedPendulum-v2",
        choices=["InvertedPendulum-v2", "HalfCheetah-v2", "Hopper-v3"],
        help="Envriment to train on",
    )
    p.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=[
            "ppo",
            "sac",
        ],
        help="RL algo to use",
    )
    p.add_argument("--num_steps", "-n", type=int, default=0.5 * 1e6)
    p.add_argument("--rollout_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=1 * 1e6)
    p.add_argument("--log_every_n_updates", "-lg", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=5 * 1e3)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--optim_cls", type=str, default="adam")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", type=int, default=2)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--profiling", "-prof", action="store_true")
    p.add_argument("--use_wandb", "-wb", action="store_true")

    args = p.parse_args()

    args.device = "cuda" if args.cuda else "cpu"

    # Enforce type int
    args.num_steps = int(args.num_steps)
    args.batch_size = int(args.batch_size)
    args.buffer_size = int(args.buffer_size)
    # How often (in terms of steps) to output training info.
    args.log_interval = args.batch_size * args.log_every_n_updates

    return args


def run(args):
    """Training Configuration"""
    algo_kwargs = dict(
        # common args
        device=args.device,
        fp16=args.fp16,
        seed=args.seed,
        gamma=0.99,
        max_grad_norm=None,
        optim_kwargs=dict(optim_cls=args.optim_cls, optim_set_to_none=True),
    )

    if args.algo.lower() == "ppo":
        # state_ space, action space inside trainer
        ppo_kwargs = dict(
            # buffer args
            batch_size=args.batch_size,  # PPO assums batch size == buffer_size
            buffer_size=args.buffer_size,  # only used in SAC,
            buffer_kwargs=dict(with_reward=True, extra_data=["log_pis"]),
            # PPO only args
            epoch_ppo=10,
            gae_lambda=0.97,
            clip_eps=0.2,
            coef_ent=0.00,
            # poliy args: net arch, activation, lr
            policy_kwargs=dict(
                pi=(64, 64),
                vf=(64, 64),
                activation="relu_inplace",
                critic_type="V",
                lr_actor=3e-4,
                lr_critic=3e-4,
            ),
        )
        algo_kwargs.update(ppo_kwargs)

    elif args.algo.lower() == "sac":
        sac_kwargs = dict(
            # buffer args
            batch_size=args.batch_size,  # PPO assums batch size == buffer_size
            buffer_size=args.buffer_size,  # only used in SAC,
            buffer_kwargs=dict(with_reward=True, extra_data=["log_pis"]),
            # SAC only args
            lr_alpha=3e-4,
            log_alpha_init=1.0,
            tau=0.02,  # 0.005
            start_steps=10_000,
            # * encourage to sync following two params to reduce overhead
            num_gradient_steps=1,  # ! slow O(n)
            target_update_interval=1,
            # poliy args: net arch, activation, lr
            policy_kwargs=dict(
                pi=(128, 128),
                qf=(128, 128),
                activation="relu_inplace",
                critic_type="twin",
                lr_actor=7.3 * 1e-4,
                lr_critic=7.3 * 1e-4,
            ),
        )
        algo_kwargs.update(sac_kwargs)

    else:
        raise ValueError()

    time = datetime.now().strftime("%Y%m%d-%H%M")
    exp_name = os.path.join(args.env_id, args.algo, f"seed{args.seed}-{time}")
    log_dir = os.path.join("runs", exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Mainly for wandb.watch function
    wandb_kwargs = dict(
        # strategy="hist",  # TODO: may implment this
        log_param=True,
        log_type="gradients",
        log_freq=100,
    )

    config = dict(
        num_steps=args.num_steps,
        env=args.env_id,
        algo=args.algo,
        algo_kwargs=algo_kwargs,
        env_kwargs=None,
        max_ep_len=args.rollout_length,
        seed=args.seed,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        save_freq=50_000,
        log_dir=log_dir,
        log_interval=args.log_interval,
        verbose=args.verbose,
        use_wandb=args.use_wandb,  # TODO: not implemented wandb intergration
        wandb_kwargs=wandb_kwargs,
    )

    # Saving hyperparams to yaml file
    with open(os.path.join(log_dir, "hyperparams.yaml"), "w") as f:
        yaml.dump(algo_kwargs, f)

    # Log with tensorboard and sync to wandb dashboard as well
    # https://docs.wandb.ai/guides/integrations/tensorboard
    if args.use_wandb:
        try:
            import wandb

            # Save API key for convenience or you have to login every time
            wandb.login()
            wandb.init(
                project="AIL",
                notes="tweak baseline",
                tags=["baseline"],
                config=config,  # Hyparams & meta data
            )
            wandb.run.name = exp_name
            # make sure to use the same config as passed to wandb
            config = wandb.config
        except ImportError:
            print("`wandb` Module Not Found")
            sys.exit(0)

    trainer = Trainer(**config)

    if args.profiling:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            trainer.run_training_loop()

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(filename="run_profiling.prof")
        stats.print_stats()
    else:
        trainer.run_training_loop()


if __name__ == "__main__":
    # ENVIRONMENT VARIABLE
    os.environ["WANDB_NOTEBOOK_NAME"] = "test"  # modify to assign a meaningful name

    args = CLI()

    if args.debug:
        import numpy as np

        np.seterr(all="raise")
        th.autograd.set_detect_anomaly(True)

    if args.cuda:
        # TODO: investigate this
        os.environ["OMP_NUM_THREADS"] = "1"
        # torch backends
        th.backends.cudnn.benchmark = True  # ? Does this useful for non-convolutions?
        
    run(args)
