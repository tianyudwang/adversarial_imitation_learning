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


def configure(args, cfg, path):
    """Training Configuration"""
    algo_kwargs = dict(
        # common args
        device=args.device,
        fp16=args.fp16,
        seed=args.seed,
        gamma=cfg.ALGO.gamma,
        max_grad_norm=cfg.ALGO.max_grad_norm,
        optim_kwargs=dict(cfg.OPTIM)
    )
    
    rl_algo = args.algo.lower()
    
    if rl_algo == "ppo":
        # state_ space, action space inside trainer
        ppo_kwargs = dict(
            # buffer args
            batch_size=args.batch_size,  # PPO assums batch size == buffer_size
            buffer_kwargs=dict(
                with_reward=cfg.PPO.with_reward,
                extra_data=cfg.PPO.extra_data),
            # PPO only args
            epoch_ppo=cfg.PPO.epoch_ppo,
            gae_lambda=cfg.PPO.gae_lambda,
            clip_eps=cfg.PPO.clip_eps,
            coef_ent=cfg.PPO.coef_ent,
            # poliy args: net arch, activation, lr
            policy_kwargs=dict(
                pi=cfg.PPO.pi,
                vf=cfg.PPO.vf,
                activation=cfg.PPO.activation,
                critic_type=cfg.PPO.critic_type,
                lr_actor=cfg.PPO.lr_actor,
                lr_critic=cfg.PPO.lr_critic,
                orthogonal_init=cfg.PPO.orthogonal_init,
            ),
        )
        algo_kwargs.update(ppo_kwargs)
        sac_kwargs = None

    elif rl_algo == "sac":
        sac_kwargs = dict(
            # buffer args
            batch_size=args.batch_size,  
            buffer_size=cfg.SAC.buffer_size,  
            buffer_kwargs=dict(
                with_reward=cfg.SAC.with_reward, 
                extra_data=cfg.SAC.extra_data),
            # SAC only args
            start_steps=cfg.SAC.start_steps,
            lr_alpha=cfg.SAC.lr_alpha,
            log_alpha_init=cfg.SAC.log_alpha_init,
            tau=cfg.SAC.tau,  
            # * Recommend to sync following two params to reduce overhead
            num_gradient_steps=cfg.SAC.num_gradient_steps,  # ! slow O(n)
            target_update_interval=cfg.SAC.target_update_interval,
            
            # poliy args: net arch, activation, lr
            policy_kwargs=dict(
                pi=cfg.SAC.pi,
                qf=cfg.SAC.qf,
                activation=cfg.SAC.activation,
                critic_type=cfg.SAC.critic_type,
                lr_actor=cfg.SAC.lr_actor,
                lr_critic=cfg.SAC.lr_critic,
            ),
        )
        algo_kwargs.update(sac_kwargs)
        ppo_kwargs = None

    else:
        raise ValueError(f"RL ALgo {args.algo} not Implemented.")

    config = dict(
        total_timesteps=args.num_steps,
        env=args.env_id,
        algo=args.algo,
        algo_kwargs=algo_kwargs,
        env_kwargs={"env_wrapper": cfg.ENV.wrapper},
        max_ep_len=args.rollout_length,
        seed=args.seed,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        save_freq=args.save_freq,
        log_dir='',
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



if __name__ == '__main__':
    pass