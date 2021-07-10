import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Any, Union

import numpy as np
import torch as th
from tqdm import tqdm

from ail.common.env_utils import maybe_make_env
from ail.common.running_stats import RunningStats
from ail.common.pytorch_util import to_torch, to_numpy
from ail.common.type_alias import GymEnv
from ail.common.utils import set_random_seed, duration, get_statistics
from ail.console.color_console import COLORS


class Trainer(ABC):
    def __init__(
        self,
        env: Union[GymEnv, str],
        num_steps: int,
        log_dir: str = "",
        env_kwargs=None,
        max_ep_len=None,
        seed: int = 42,
        eval_interval: int = 5_000,
        num_eval_episodes: int = 10,
        save_freq: int = 10,
        log_interval: int = 5_000,
        verbose: int = 2,
        use_wandb=False,
        **kwargs,
    ):

        if env_kwargs is None:
            env_kwargs = {
                "env_wrapper": [],
                "tag": "training",
                "color": "dim_blue",
            }

        # Env to collect samples.
        self.env = maybe_make_env(env, verbose=verbose, **env_kwargs)
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = maybe_make_env(
            env,
            env_wrapper=None,
            verbose=verbose,
            tag="test",
            color="dim_blue",
        )
        self.env_test.seed(2 ** 31 - seed)

        self.max_ep_len: int = (
            max_ep_len
            if max_ep_len is not None and isinstance(max_ep_len, int)
            else self.env._max_episode_steps  # noqa
        )

        # Set RNG seed.
        set_random_seed(seed)

        # Tensorboard/wandb log setting.
        self.log_dir, self.summary_dir, self.model_dir = (
            log_dir,
            os.path.join(log_dir, "summary"),
            os.path.join(log_dir, "model"),
        )
        self.use_wandb = use_wandb
        self.writer = None

        # Log and Saving.
        self.save_freq = save_freq
        self.eval_interval = eval_interval

        self.num_eval_episodes = num_eval_episodes
        self.stochastic_eval_episodes = num_eval_episodes // 2
        self.log_interval = log_interval

        # Other parameters.
        self.num_steps = num_steps
        self.n_steps_pbar = tqdm(range(1, num_steps + 1))
        self.best_rew_mean, self.best_ret = -float("inf"), -float("inf")
        self.rs = RunningStats()
        self.verbose = verbose
        self.algo = None
        self.batch_size = None
        self.device = None
        self.start_time = None


    # -----------------------
    # Training/ evaluation
    # -----------------------


    @abstractmethod
    def run_training_loop(self):
        raise NotImplementedError()

    @th.no_grad()
    def evaluate(self, step) -> None:
        """Evaluate current policy"""
        self.algo.eval()
        train_returns, train_ep_lens = [], []
        valid_returns, valid_ep_lens = [], []

        for t in range(self.num_eval_episodes):

            state, done = self.env_test.reset(), False
            ep_len, ep_ret = 0, 0.0
            deterministic = False if t < self.stochastic_eval_episodes else True

            while not done:
                state = self.obs_as_tensor(state)
                action = to_numpy(self.algo.get_action(state, deterministic))

                state, reward, done, info = self.env_test.step(action)
                ep_len += 1
                ep_ret += reward
            if deterministic:
                valid_ep_lens.append(ep_len)
                valid_returns.append(ep_ret)
            else:
                train_ep_lens.append(ep_len)
                train_returns.append(ep_ret)

        # Logging evaluation.
        self.eval_logging(
            step,
            train_returns,
            train_ep_lens,
            valid_returns,
            valid_ep_lens,
        )
        # Turn back to train mode.
        self.algo.train()


    # -----------------------
    # Conditions
    # -----------------------


    def is_train_logging(self, step) -> bool:
        return (
            step % self.log_interval == 0 or step == self.num_steps
        ) and self.verbose == 1

    def is_eval_logging(self, step) -> bool:
        cond = (
            (step % self.eval_interval == 0) or (step == self.num_steps),
            self.verbose == 1,
        )
        return all(cond)

    def is_saving_model(self, step) -> bool:
        cond = (
            step > 0,
            step % self.save_freq == 0,
            step / self.num_steps > 0.3
        )
        return all(cond) or step == self.num_steps - 1

    def train_logging(self, train_logs, step) -> None:
        """Log training info (no saving yet)"""
        time_logs = OrderedDict()
        time_logs["total_timestep"] = step
        time_logs["time_elapsed "] = duration(self.start_time)

        print("-" * 41)
        self.output_block(train_logs, tag="Train", color="back_dim_green")
        self.output_block(time_logs, tag="Time", color="back_dim_cyan")
        print("-" * 41 + "\n")

    def eval_logging(
        self, step, train_returns, train_ep_lens, eval_returns, eval_ep_lens
    ) -> None:
        """Log evaluation info"""
        train_logs, eval_logs, time_logs = OrderedDict(), OrderedDict(), OrderedDict()

        # Time
        time_logs["total_timestep"] = step
        time_logs["time_elapsed "] = duration(self.start_time)

        # Train
        train_logs["ep_len_mean"] = np.mean(train_ep_lens)
        (
            train_logs["ep_return_mean"],
            train_logs["std_return"],
            train_logs["max_return"],
            train_logs["min_return"]
        ) = get_statistics(train_returns)

        # Eval
        eval_logs["ep_len_mean"] = np.mean(eval_ep_lens)
        (
            eval_logs["ep_return_mean"],
            eval_logs["std_return"],
            eval_logs["max_return"],
            eval_logs["min_return"]
        ) = get_statistics(eval_returns)

        print("-" * 41)
        self.output_block(train_logs, tag="Train", color="back_dim_green")
        self.output_block(eval_logs, tag="Evaluate", color="back_dim_red")
        self.output_block(time_logs, tag="Time", color="back_dim_yellow")
        print("-" * 41 + "\n")

        self.metric_to_tb(step, train_logs, eval_logs)

    @staticmethod
    def output_block(logs: Dict[str, Any], tag: str, color="invisible") -> None:
        """print a block of logs with color and format"""
        print("".join([COLORS[color], f"| {tag + '/': <10}{'|': >29}"]))
        for k, v in logs.items():
            a = f"|  {k: <15}\t{'| '}"
            if isinstance(v, float):
                b = f"{v: <12.3e}\t|" if abs(v) < 1e-4 else f"{v: <12.3f}\t|"
            else:
                b = f"{v: <12}\t|"
            print("".join([a, b]))

    # -----------------------
    # Logging/Saving methods
    # -----------------------

    def metric_to_tb(self, step, train_logs, eval_logs):
        # Train logs
        self.writer.add_scalar(
            "return/train/ep_len", train_logs.get("ep_len_mean"), step
        )
        self.writer.add_scalar(
            "return/train/ep_rew_mean", train_logs.get("ep_return_mean"), step
        )
        self.writer.add_scalar(
            "return/train/ep_rew_std", train_logs.get("std_return"), step
        )

        # Test log
        self.writer.add_scalar(
            "return/test/ep_len", eval_logs.get("ep_len_mean"), step
        )
        self.writer.add_scalar(
            "return/test/ep_rew_mean", eval_logs.get("ep_return_mean"), step
        )
        self.writer.add_scalar(
            "return/test/ep_rew_std", eval_logs.get("std_return"), step
        )

    def info_to_tb(self, train_logs, epoch) -> None:
        """Logging to tensorboard or wandb (if sync)"""
        assert train_logs is not None, "train log can not be `None`"
        self.writer.add_scalar("loss/actor", train_logs.get("actor_loss"), epoch)
        self.writer.add_scalar("loss/critic", train_logs.get("critic_loss"), epoch)
        self.writer.add_scalar("info/actor/approx_kl", train_logs.get("approx_kl"), epoch)
        self.writer.add_scalar("info/actor/entropy", train_logs.get("entropy"), epoch)
        self.writer.add_scalar(
            "info/actor/clip_fraction", train_logs["clip_fraction"], epoch
        )

    def save_models(self, save_dir: str, verbose=False):
        # use algo.sav_mdoels directly for now
        self.algo.save_models(save_dir, verbose)

    def update_progress(self, ep_ret) -> None:

        if ep_ret > self.best_ret:
            self.best_ret = ep_ret

        self.n_steps_pbar.set_description(
            f"{self.algo.__class__.__name__} ({self.device}): "
            f"| Best Mean Ret: {self.best_ret:.2f} "
            f"| Running Stats: {self.rs.mean():.2f} +/- "
            f"{self.rs.standard_deviation():.2f}"
        )

    # -----------------------
    # Helper functions.
    # -----------------------

    def obs_as_tensor(self, obs, copy=False) -> Union[Dict[str, th.Tensor], th.Tensor]:
        """
        Moves the observation to the given device.
        :param obs:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return: PyTorch tensor of the observation on a desired device.
        """
        if isinstance(obs, np.ndarray):
            return to_torch(obs, self.device, copy)
        elif isinstance(obs, th.Tensor):
            return obs.to(self.device)
        elif isinstance(obs, dict):
            return {
                key: to_torch(_obs, self.device, copy) for (key, _obs) in obs.items()
            }
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    def to_torch(self, array: np.ndarray, copy: bool = True,) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array, dtype=th.float32).to(self.device)
        elif isinstance(array, np.ndarray):
            return self.from_numpy(array)
        else:
            return th.as_tensor(array, dtype=th.float32).to(self.device)

    def from_numpy(self, array: np.ndarray) -> th.Tensor:
        """Convert numpy array to torch tensor  and send to device('cuda:0' or 'cpu')"""
        return th.from_numpy(array).float().to(self.device)

    @staticmethod
    def to_numpy(tensor: th.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array and send to CPU"""
        return tensor.detach().cpu().numpy()
