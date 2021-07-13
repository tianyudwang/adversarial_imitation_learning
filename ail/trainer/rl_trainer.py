from typing import Union
from time import time

from torch.utils.tensorboard import SummaryWriter
from ail.trainer.base_trainer import BaseTrainer
from ail.common.utils import countdown
from ail.common.type_alias import GymEnv


class RL_Trainer(BaseTrainer):
    def __init__(
        self,
        num_steps: int,
        env: Union[GymEnv, str],
        algo,
        algo_kwargs: dict,
        env_kwargs=None,
        max_ep_len=None,
        seed: int = 42,
        eval_interval: int = 5_000,
        num_eval_episodes: int = 10,
        save_freq: int = 10_000,
        log_dir: str = "runs",
        log_interval: int = 5_000,
        verbose: int = 2,
        use_wandb: bool = False,
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

        self.algo = algo(
            self.env.observation_space,
            self.env.action_space,
            **algo_kwargs,
        )

        # Sync same device with algo.
        self.device = self.algo.device

        # Log setting.
        self.writer = SummaryWriter(log_dir=self.summary_dir)

    def run_training_loop(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        obs = self.env.reset()

        for step in self.n_steps_pbar:
            # Pass to the algorithm to update state and episode timestep.
            # return of algo.step() is next_obs
            obs, t = self.algo.step(self.env, self.obs_as_tensor(obs), t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update()

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)

        # Wait for the logging to be finished.
        countdown(10)
