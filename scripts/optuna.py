import argparse
import os


import optuna
from optuna import Trial

from ail.trainer import Trainer
from ail.common.utils import make_unique_timestamp


def CLI():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_id",
        type=str,
        choices=["InvertedPendulum-v2", "HalfCheetah-v2", "Hopper-v3"],
        help="Envriment to train on",
    )
    p.add_argument(
        "--algo",
        type=str,
        choices=[
            "ppo",
            "sac",
        ],
        help="RL algo to use",
    )
    p.add_argument("--num_steps", "-n", type=int, default=0.5 * 1e6)
    p.add_argument("--rollout_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    # p.add_argument("--buffer_size", type=int, default=1 * 1e6)
    p.add_argument("--log_every_n_updates", "-lg", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=5 * 1e3)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--save_freq", type=int, default=50_000)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", type=int, default=2)

    args = p.parse_args()

    args.device = "cuda" if args.cuda else "cpu"

    # Enforce type int
    args.num_steps = int(args.num_steps)
    args.batch_size = int(args.batch_size)
    args.log_every_n_updates = int(args.log_every_n_updates)

    # How often (in terms of steps) to output training info.
    args.log_interval = args.batch_size * args.log_every_n_updates

    return args


def configure(args, trial):
    """Training Configuration"""
    algo_kwargs = dict(
        # common args
        device=args.device,
        fp16=args.fp16,
        seed=args.seed,
        gamma=0.99,
        max_grad_norm=trial.suggest_categorical(
            "max_grad_norm", [None, 0.5, 1.0, 5.0, 10]
        ),
        optim_kwargs={
            "optim_cls": trial.suggest_categorical("optim_cls", ["Adam", "AdamW"]),
            "optim_set_to_none": False,
        },
    )

    net_arch = {
        "small": [64, 64],
        "medium": [128, 128],
        "big": [256, 256],
    }

    rl_algo = args.algo.lower()

    if rl_algo == "ppo":
        # state_ space, action space inside trainer
        ppo_kwargs = dict(
            # buffer args
            batch_size=args.batch_size,  # PPO assums batch size == buffer_size
            buffer_kwargs=dict(with_reward=True, extra_data=["log_pis"]),
            # PPO only args
            epoch_ppo=trial.suggest_categorical("epoch_ppo", [5, 10, 15, 20]),
            gae_lambda=0.97,
            clip_eps=0.2,
            coef_ent=0.01,
            # poliy args: net arch, activation, lr
            policy_kwargs=dict(
                pi=net_arch[
                    trial.suggest_categorical("pi", ["small", "medium", "big"])
                ],
                vf=net_arch[
                    trial.suggest_categorical("vf", ["small", "medium", "big"])
                ],
                activation="relu",
                critic_type="relu",
                lr_actor=trial.suggest_loguniform("lr_actor", 1e-4, 3e-3),
                lr_critic=trial.suggest_loguniform("lr_critic", 1e-4, 3e-3),
                orthogonal_init=trial.suggest_categorical(
                    "orthogonal_init", [True, False]
                ),
            ),
        )
        algo_kwargs.update(ppo_kwargs)

    # elif rl_algo == "sac":
    #     sac_kwargs = dict(
    #         # buffer args
    #         batch_size=args.batch_size,
    #         buffer_size=cfg.SAC.buffer_size,
    #         buffer_kwargs=dict(
    #             with_reward=cfg.SAC.with_reward,
    #             extra_data=cfg.SAC.extra_data),
    #         # SAC only args
    #         start_steps=cfg.SAC.start_steps,
    #         lr_alpha=cfg.SAC.lr_alpha,
    #         log_alpha_init=cfg.SAC.log_alpha_init,
    #         tau=cfg.SAC.tau,
    #         # * Recommend to sync following two params to reduce overhead
    #         num_gradient_steps=cfg.SAC.num_gradient_steps,  # ! slow O(n)
    #         target_update_interval=cfg.SAC.target_update_interval,

    #         # poliy args: net arch, activation, lr
    #         policy_kwargs=dict(
    #             pi=cfg.SAC.pi,
    #             qf=cfg.SAC.qf,
    #             activation=cfg.SAC.activation,
    #             critic_type=cfg.SAC.critic_type,
    #             lr_actor=cfg.SAC.lr_actor,
    #             lr_critic=cfg.SAC.lr_critic,
    #         ),
    #     )
    #     algo_kwargs.update(sac_kwargs)
    #     ppo_kwargs = None

    config = dict(
        total_timesteps=args.num_steps,
        env=args.env_id,
        algo=args.algo,
        algo_kwargs=algo_kwargs,
        env_kwargs={"env_wrapper": ["clip_act"]},
        max_ep_len=args.rollout_length,
        seed=args.seed,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        save_freq=args.save_freq,
        log_dir="",
        log_interval=args.log_interval,
        verbose=args.verbose,
        use_wandb=False,
        wandb_kwargs={},
    )

    # Create Trainer
    # trainer = Trainer(**config)

    return config


def objective(trial: Trial):
    pass


if __name__ == "__main__":
    pass
