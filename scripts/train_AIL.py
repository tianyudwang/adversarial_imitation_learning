import os
import pathlib
import sys
import argparse
from copy import deepcopy
from datetime import datetime

import yaml
import numpy as np
import torch as th

from ail.trainer import Trainer
from config.config import get_cfg_defaults


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
        help="Envriment to train on.",
    )
    p.add_argument(
        "--algo",
        type=str,
        default="airl",
        choices=[
            "airl",
            "gail",
        ],
        help="Adversarial imitation algo to use.",
    )
    p.add_argument(
        "--gen_algo",
        type=str,
        default="ppo",
        choices=[
            "ppo",
            "sac",
        ],
        help="RL algo to use as generator.",
    )
    p.add_argument(
        "--demo_path", "-demo", type=str, help="Path to demo"
    )  # required=True,

    # TODO: add more arguments to control discriminator
    # Discriminator features
    p.add_argument("--spectral_norm", "-sn", action="store_true")
    p.add_argument("--dropout", "-dp", action="store_true")  # TODO: Implement and test

    # Total steps and batch size
    p.add_argument("--num_steps", "-n", type=int, default=1 * 1e6)
    p.add_argument("--rollout_length", "-ep_len", type=int, default=None)
    p.add_argument("--gen_batch_size", "-gb", type=int, default=1_000)
    p.add_argument("--replay_batch_size", "-rbs", type=int, default=256)
    # p.add_argument("--buffer_size", type=int, default=1 * 1e6)

    # Logging and evaluation
    p.add_argument("--log_every_n_updates", "-lg", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=5 * 1e3)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--save_freq", type=int, default=50_000, 
                   help="Save model every `save_freq` steps.")

    # Cuda options
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fp16", action="store_true")

    # Random seed
    p.add_argument("--seed", type=int, default=0)

    # Utility
    p.add_argument("--verbose", type=int, default=2)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--use_wandb", "-wb", action="store_true")
    args = p.parse_args()

    args.device = "cuda" if args.cuda else "cpu"

    # Enforce type int
    args.num_steps = int(args.num_steps)
    args.batch_size = int(args.gen_batch_size)
    args.log_every_n_updates = int(args.log_every_n_updates)
    
    # How often (in terms of steps) to output training info.
    args.log_interval = args.gen_batch_size * args.log_every_n_updates
    return args


def run(args, cfg, path):
    """Training Configuration."""

    algo_kwargs = dict(
        # common args
        device=args.device,
        fp16=args.fp16,
        seed=args.seed,
        gamma=cfg.ALGO.gamma,
        max_grad_norm=100,#cfg.ALGO.max_grad_norm,
        optim_kwargs=dict(cfg.OPTIM),
    )
    
    gen_algo = args.gen_algo.lower()
    if  gen_algo == "ppo":
        # state space, action space inside trainer
        ppo_kwargs = dict(
            # buffer args
            batch_size=args.batch_size,  # PPO assums batch size == buffer_size
            buffer_kwargs=dict(with_reward=False, extra_data=["log_pis"]),
            # PPO only args
            epoch_ppo=cfg.PPO.epoch_ppo,
            gae_lambda=cfg.PPO.gae_lambda,
            clip_eps=cfg.PPO.clip_eps,
            coef_ent=cfg.PPO.coef_ent,
            
            # poliy args: net arch, activation, lr.
            policy_kwargs=dict(
                pi=cfg.PPO.pi,
                vf=cfg.PPO.vf,
                activation=cfg.PPO.activation,
                critic_type=cfg.PPO.critic_type,
                lr_actor=cfg.PPO.lr_actor,
                lr_critic=cfg.PPO.lr_critic,
            ),
        )
        gen_kwargs = {**algo_kwargs, **ppo_kwargs}
        sac_kwargs = None

    elif gen_algo == "sac":
        sac_kwargs = dict(
            # buffer args.
            batch_size=args.batch_size,  # PPO assums batch size == buffer_size
            buffer_size=cfg.SAC.buffer_size,  # only used in SAC,
            buffer_kwargs=dict(with_reward=False, extra_data=["log_pis"]),
            # SAC only args.
            start_steps=cfg.SAC.start_steps,
            lr_alpha=cfg.SAC.lr_alpha,
            log_alpha_init=cfg.SAC.log_alpha_init,
            tau=cfg.SAC.tau,  # 0.005
            # * Recommend to sync following two params to reduce overhead.
            num_gradient_steps=cfg.SAC.num_gradient_steps,  # ! slow O(n)
            target_update_interval=cfg.SAC.target_update_interval,
            
           # poliy args: net arch, activation, lr.
            policy_kwargs=dict(
                pi=cfg.SAC.pi,
                qf=cfg.SAC.qf,
                activation=cfg.SAC.activation,
                critic_type=cfg.SAC.critic_type,
                lr_actor=cfg.SAC.lr_actor,
                lr_critic=cfg.SAC.lr_critic,
            ),
        )
        gen_kwargs = {**algo_kwargs, **sac_kwargs}
        ppo_kwargs = None

    else:
        raise ValueError(f"RL ALgo (generator) {args.gen_algo} not Implemented.")

    
    # Demo data.
    if args.demo_path is None:
        # TODO: REMOVE THIS
        args.demo_path = (
            path / "transitions" / args.env_id / "size11000.npz"
        )
        
    transitions = dict(np.load(args.demo_path))

    algo_kwargs.update(
        dict(
            replay_batch_size=args.replay_batch_size,
            buffer_exp="replay",
            buffer_kwargs=dict(
                with_reward=False,
                transitions=transitions,  # * transitions must be a dict
            ),
            gen_algo=args.gen_algo,
            gen_kwargs=gen_kwargs,
            disc_cls="airl_sa",
            disc_kwargs=dict(
                hidden_units=cfg.DISC.hidden_units,
                hidden_activation=cfg.DISC.hidden_activation,
                gamma=cfg.ALGO.gamma,
                disc_kwargs={
                    "spectral_norm": cfg.DISC.spectral_norm,
                    "dropout": cfg.DISC.dropout,
                }
            ),
            epoch_disc=cfg.DISC.epoch_disc,
            lr_disc=cfg.DISC.lr_disc,
            subtract_logp = cfg.AIRL.subtract_logp,
            rew_type = cfg.AIRL.rew_type,
            rew_input_choice = cfg.AIRL.rew_input_choice,
        )
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    exp_name = os.path.join(args.env_id, args.algo, f"seed{args.seed}-{time}")
    log_dir = path.joinpath("runs", exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True) 


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
        use_wandb=args.use_wandb,
        wandb_kwargs=cfg.WANDB,
    )

    # Log with tensorboard and sync to wandb dashboard as well.
    # https://docs.wandb.ai/guides/integrations/tensorboard
    if args.use_wandb:
        try:
            import wandb

            # Not to store expert data in wandb.
            config_copy = deepcopy(config)
            config_copy["algo_kwargs"]["buffer_kwargs"].pop("transitions")

            # Save API key for convenience or you have to login every time.
            wandb.login()
            wandb.init(
                project="AIL",
                notes="tweak baseline",
                tags=["baseline"],
                config=config_copy,  # Hyparams & meta data.
            )
            wandb.run.name = exp_name
        except ImportError:
            print("`wandb` Module Not Found")
            sys.exit(0)

    # Create Trainer.
    trainer = Trainer(**config)

    # It's a dict of data too large to store.
    algo_kwargs["buffer_kwargs"].pop("transitions")
    # algo kwargs
    print("-" * 10, f"{args.algo}", "-" * 10)
    ic(algo_kwargs)

    # Saving hyperparams to yaml file.
    with open(os.path.join(log_dir, "hyperparams.yaml"), "w") as f:
        yaml.dump(algo_kwargs, f)

    del algo_kwargs, gen_kwargs, ppo_kwargs, sac_kwargs

    trainer.run_training_loop()


if __name__ == "__main__":
    # ENVIRONMENT VARIABLE
    os.environ["WANDB_NOTEBOOK_NAME"] = "test"  # Modify to assign a meaningful name.

    args = CLI()
    
    # Path
    path = pathlib.Path(__file__).parent.resolve()
    print(f"File_dir: {path}")    
    
    cfg_path = path / "config" /'ail_configs.yaml'
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    if args.debug:
        np.seterr(all="raise")
        th.autograd.set_detect_anomaly(True)

    if args.cuda:
        # os.environ["OMP_NUM_THREADS"] = "1"
        # torch backends
        th.backends.cudnn.benchmark = cfg.CUDA.cudnn  # ? Does this useful for non-convolutions?

    run(args, cfg, path)
