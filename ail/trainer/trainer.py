from typing import Union, Optional, Dict, Any
from pprint import pprint
from time import time
import os
import re

from torch.utils.tensorboard import SummaryWriter

from ail.agents import ALGO
from ail.agents.rl_agent.rl_core import OnPolicyAgent, OffPolicyAgent
from ail.common.type_alias import GymEnv
from ail.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    RL_Trainer with tensorboard integration.

    :param num_steps: number of steps to train
    :param env: The environment must satisfy the OpenAI Gym API.
    :param algo: The RL algorithm to train with.
    :param algo_kwargs: kwargs to pass to the algorithm.
    :param env_kwargs: Any kwargs appropriate for the gym env object
        including custom wrapper.
    :param max_ep_len: Total length of a trajectory
        By default, equals to env's own time limit.
    :param eval_interval: How often to evaluate current policy
        By default, we enforce to create a copy of training env for evaluation.
    :param save_freq: How often to save the current policy.
    :param log_dir: path to log directory
    :param log_interval: How often to output training info.
    :param seed: random seed.
    :param verbose: The verbosity level: 0 no output, 1 info, 2 debug.
    :param use_wandb: Wether to use wandb for metrics visualization.
    :param wandb_kwargs: kwargs to pass to wandb.init.
    """

    def __init__(
        self,
        num_steps: int,
        env: Union[GymEnv, str],
        algo: Union[OnPolicyAgent, OffPolicyAgent],
        algo_kwargs: Dict[str, Any],
        env_kwargs: Optional[Dict[str, Any]] = None,
        max_ep_len=None,
        seed: int = 42,
        eval_interval: int = 5_000,
        num_eval_episodes: int = 10,
        save_freq: int = 50_000,
        log_dir: str = "runs",
        log_interval: int = 10_000,
        verbose: int = 2,
        use_wandb: bool = False,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            num_steps,
            env,
            env_kwargs,
            max_ep_len,
            eval_interval,
            num_eval_episodes,
            save_freq,
            log_dir,
            log_interval,
            seed,
            verbose,
            use_wandb,
            **kwargs,
        )

        if isinstance(algo, str):
            self.algo = ALGO[algo.lower()](
                self.env.observation_space,
                self.env.action_space,
                **algo_kwargs,
            )
        else:
            self.algo = algo(
                self.env.observation_space,
                self.env.action_space,
                **algo_kwargs,
            )

        # Number of variables and net arch.
        if self.verbose > 1:
            var_counts = self.algo.info()
            print("-"*70)
            pprint(var_counts)
            print("-"*70, "\n")
            

        # Sync same device with algo.
        self.device = self.algo.device

        if self.use_wandb:
            import wandb

            if wandb_kwargs is None:
                wandb_kwargs = {}

            if wandb_kwargs.get("log_param", True):
                # wandb magic to track gradients.
                wandb.watch(
                    self.algo,
                    log=wandb_kwargs.get("log_type", "gradients"),
                    log_freq=wandb_kwargs.get("log_freq", 1000),
                )
            # Sync with tensorboard.
            wandb.tensorboard.patch(root_logdir=self.summary_dir, pytorch=True)

        # Log setting.
        self.writer = SummaryWriter(log_dir=self.summary_dir)

        DEVICE = "".join(re.findall("[a-zA-Z]+", str(self.device)))
        self.n_steps_pbar.set_description(f"{self.algo} ({DEVICE})")

    def run_training_loop(self):
        """
        Interactive with environment and train agent for num_steps.
        """
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        obs = self.env.reset()

        # log = []
        for step in self.n_steps_pbar:
            # Pass to the algorithm to update state and episode timestep.
            # * return of algo.step() is next_obs
            obs, t = self.algo.step(self.env, obs, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                if self.is_train_logging(step):
                    train_logs = self.algo.update(log_this_batch=True)
                    train_logs = self.convert_logs(train_logs)

                    # Print changes from training updates.
                    self.train_logging(train_logs, step)

                    # Logging changes to tensorboard.
                    self.info_to_tb(train_logs, step)
                    # TODO: Set a better log strategy to reduce overhead. Current downsampling.
                    # TODO: implement two more logging strategies: Summarization / histogram.
                    # log.append(train_logs)
                    # if len(log) == 5:
                    #     summary = defaultdict(list)
                    #     for info in log:
                    #         for k, v in info.items():
                    #             summary[k].append(v)
                    #     summary = self.convert_logs(summary)
                    #     ic(summary)
                    #     self.info_to_tb(summary, step)
                    #     log.clear()
                else:
                    self.algo.update(log_this_batch=False)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)

            # Saving the model.
            if self.is_saving_model(step):
                self.save_models(os.path.join(self.model_dir, f"step{step}"))

        self.finish_logging()
