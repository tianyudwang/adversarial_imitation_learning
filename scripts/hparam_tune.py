from typing import Any, Dict
import argparse
import sys
import logging
from pprint import pprint

import pandas as pd
import torch as th

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

import optuna
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances
)
from ail.trainer import Trainer


NetArch = {
    "tiny": [32, 32],
    "small": [64, 64],
    "medium": [128, 128],
    "large": [256, 256],
    "huge": [512, 512],
}


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
    p.add_argument("--log_every_n_updates", "-lg", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=5 * 1e3)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--save_freq", type=int, default=50_000)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", type=int, default=1)

    # Optuna args
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument(
        "--sampler", type=str, choices=["tpe", "random", "skopt"], default="tpe"
    )
    p.add_argument(
        "--pruner", type=str, choices=["halving", "median", "none"], default="median"
    )
    p.add_argument("--n_startup_trials", type=int, default=5)
    p.add_argument("--n_evaluations", type=int, default=2)
    args = p.parse_args()

    args.device = "cuda" if args.cuda else "cpu"

    # Enforce type int
    args.num_steps = int(args.num_steps)
    args.log_every_n_updates = int(args.log_every_n_updates)
    return args


def sample_ppo_params(trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    buffer_kwargs = dict(
        with_reward=True, extra_data=["log_pis"]
    )  # no need to change this
    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512, 1024, 2048]
    )
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    epoch_ppo = trial.suggest_categorical("epoch_ppo", [1, 5, 10, 15, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.92, 0.95, 0.98, 0.99])
    clip_eps = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.25, 0.3, 0.4])
    coef_ent = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)

    # poliy args: net arch, activation, lr
    policy_kwargs = dict(
        pi=NetArch[trial.suggest_categorical("pi", ["small", "medium", "large"])],
        vf=NetArch[trial.suggest_categorical("vf", ["small", "medium", "large"])],
        activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
        critic_type="V",
        lr_actor=trial.suggest_loguniform("lr_actor", 1e-4, 5e-3),
        lr_critic=trial.suggest_loguniform("lr_critic", 1e-4, 5e-3),
        orthogonal_init=trial.suggest_categorical("orthogonal_init", [True, False]),
    )
    optim_kwargs = {
        "optim_cls": trial.suggest_categorical("optim_cls", ["Adam", "AdamW"]),
        "optim_set_to_none": True,
    }

    ppo_hparams = {
        "buffer_kwargs": buffer_kwargs,
        "batch_size": batch_size,
        "gamma": gamma,
        "max_grad_norm": max_grad_norm,
        "epoch_ppo": epoch_ppo,
        "gae_lambda": gae_lambda,
        "clip_eps": clip_eps,
        "coef_ent": coef_ent,
        "policy_kwargs": policy_kwargs,
        "optim_kwargs": optim_kwargs,
    }
    return ppo_hparams


def create_sampler(args) -> BaseSampler:
    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
    if args.sampler == "random":
        sampler = RandomSampler(seed=args.seed)
    elif args.sampler == "tpe":
        sampler = TPESampler(n_startup_trials=args.n_startup_trials, seed=args.seed)
    elif args.sampler == "skopt":
        # cf https://scikit-optimize.github.io/#skopt.Optimizer
        # GP: gaussian process
        # Gradient boosted regression: GBRT
        sampler = SkoptSampler(
            skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"}
        )
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")
    return sampler


def create_pruner(args) -> BasePruner:
    if args.pruner == "halving":
        pruner = SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=4, min_early_stopping_rate=0
        )
    elif args.pruner == "median":
        # Do not prune before 1/3 of the max budget is used
        pruner = MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_evaluations // 3,
            
        )
    elif args.pruner == "none":
        # Do not prune
        pruner = MedianPruner(
            n_startup_trials=args.n_trials, n_warmup_steps=args.n_evaluations
        )
    else:
        raise ValueError(f"Unknown pruner: {args.pruner}")
    return pruner


def objective(trial):
    """Training Configuration"""
    algo_kwargs = dict(
        # common args
        device=args.device,
        fp16=args.fp16,
        seed=args.seed,
    )
    if args.algo.lower() == "ppo":
        ppo_kwargs = sample_ppo_params(trial).copy()
        algo_kwargs.update(ppo_kwargs)

    elif args.algo.lower() == "sac":
        pass

    config = dict(
        total_timesteps=args.num_steps,
        env=args.env_id,
        algo=args.algo,
        algo_kwargs=algo_kwargs,
        env_kwargs={"env_wrapper": ["clip_act"]},
        test_env_kwargs={"env_wrapper": ["clip_act"]},
        max_ep_len=args.rollout_length,
        seed=args.seed,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        save_freq=args.save_freq,
        log_dir="",
        log_interval=5_000,
        verbose=args.verbose,
        use_wandb=False,
        wandb_kwargs={},
        use_optuna=True,
        trial=trial,
    )
    trainer = Trainer(**config)

    try:
        trainer.run_training_loop()
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        raise optuna.exceptions.TrialPruned()

    finally:
        # Free memory
        trainer.env.close()
        trainer.env_test.close()

    return trainer.get_records()


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    global args
    args = CLI()

    # Set pytorch num threads to 1 for faster training
    th.set_num_threads(1)

    sampler = create_sampler(args)
    pruner = create_pruner(args)

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name="ppo",
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )
    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    except KeyboardInterrupt:
        pass

    print("\n")

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("\n\n")

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv("./ppo_trials.csv", index=False)
    pprint(df)
    
    plot_optimization_history(study)
    plot_parallel_coordinate(study)
    plot_contour(study)
    plot_param_importances(study)

    if True:
        pass
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
        # algo_kwargs.update(sac_kwargs)
        # ppo_kwargs = None
