import os
from time import time
from datetime import timedelta
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Any, Union, Optional

import numpy as np
import torch as th
from tqdm import tqdm

from ail.common.env_utils import maybe_make_env
from ail.common.running_stats import RunningStats
from ail.common.pytorch_util import to_torch
from ail.common.type_alias import GymEnv
from ail.common.utils import set_random_seed, get_stats, countdown
from ail.console.color_console import COLORS, Console


class BaseTrainer(ABC):
    def __init__(
        self,
        num_steps: int,
        env: Union[GymEnv, str],
        env_kwargs: Dict[str, Any],
        max_ep_len: Optional[int],
        eval_interval: int,
        num_eval_episodes: int,
        save_freq: int,
        log_dir: str,
        log_interval: int,
        seed: int,
        verbose: int,
        use_wandb: bool,
        **kwargs,
    ):

        if env_kwargs is None:
            env_kwargs = {
                "env_wrapper": [],  # ? should we apply ActionNormlize wrapper by default?
                "tag": "training",
                "color": "dim_blue",
            }

        # Env to collect samples.
        self.env = maybe_make_env(env, verbose=verbose, **env_kwargs)
        self.seed = seed
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

        # Set max_ep_len or use default
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

        for d in [self.log_dir, self.summary_dir, self.model_dir]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

        self.use_wandb = use_wandb
        self.writer = None

        # Log and Saving.
        self.save_freq = save_freq
        self.eval_interval = eval_interval

        self.num_eval_episodes = num_eval_episodes
        self.stochastic_eval_episodes = num_eval_episodes // 2
        self.log_interval = log_interval

        # Progress param
        self.n_steps_pbar = tqdm(range(1, num_steps + 1))
        self.best_ret = -float("inf")
        self.rs = RunningStats()
        self.train_count = 0

        # Other parameters.
        self.num_steps = num_steps
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
    def evaluate(self, step: int) -> None:
        # set algo to evaluation mode
        self.algo.actor.eval()
        self.rs.clear()
        train_returns, train_ep_lens = [], []
        valid_returns, valid_ep_lens = [], []
        # visualize result from half explore and half exploit
        stochastic_eval_episodes = self.num_eval_episodes // 2
        for t in range(self.num_eval_episodes):
            obs = self.env_test.reset()
            ep_ret = 0.0
            ep_len = 0
            done = False

            while not done:
                obs = self.obs_as_tensor(obs)
                if t < stochastic_eval_episodes:
                    act, _ = self.algo.explore(obs)
                else:
                    act = self.algo.exploit(obs)
                obs, reward, done, _ = self.env_test.step(act)
                ep_ret += reward
                ep_len += 1

            self.rs.push(ep_ret)
            deterministic = False if t < stochastic_eval_episodes else True
            if deterministic:
                valid_ep_lens.append(ep_len)
                valid_returns.append(ep_ret)
            else:
                train_ep_lens.append(ep_len)
                train_returns.append(ep_ret)

        # Logging evaluation.
        if self.is_eval_logging(step):
            self.eval_logging(
                step,
                train_returns,
                train_ep_lens,
                valid_returns,
                valid_ep_lens,
            )
        # Turn back to train mode.
        self.algo.train()

        if self.rs.mean() > self.best_ret:
            self.best_ret = self.rs.mean()
        Console.info(
            f"Num steps: {step}\t"
            f"| Best Ret: {self.best_ret:.1f}\t"
            f"| Return: {self.rs.mean():.1f}"
        )

    # -----------------------
    # Logging conditions
    # -----------------------

    def is_train_logging(self, step: int) -> bool:
        return all(
            [
                step % self.log_interval == 0,
                step > self.log_interval,
                self.verbose >= 1,
            ]
        )

    def is_eval_logging(self, step: int) -> bool:
        return all(
            [
                step % self.eval_interval == 0,
                step > self.eval_interval,
                self.verbose >= 1,
            ]
        )

    def is_saving_model(self, step: int) -> bool:
        cond = (step > 0, step % self.save_freq == 0, step / self.num_steps > 0.3)
        return all(cond) or step == self.num_steps - 1

    def train_logging(self, train_logs: Dict[str, Any], step: int) -> None:
        """Log training info (no saving yet)"""
        if self.is_train_logging(step):
            time_logs = OrderedDict()
            time_logs["total_timestep"] = step
            time_logs["time_elapsed "] = self.duration(self.start_time)

            print("-" * 41)
            self.output_block(train_logs, tag="Train", color="invisible")
            self.output_block(time_logs, tag="Time", color="invisible")
            print("\n")

    def eval_logging(
        self,
        step: int,
        train_returns: np.ndarray,
        train_ep_lens: np.ndarray,
        eval_returns: np.ndarray,
        eval_ep_lens: np.ndarray,
    ) -> None:
        """Log evaluation info"""
        train_logs, eval_logs, time_logs = OrderedDict(), OrderedDict(), OrderedDict()

        # Time
        time_logs["total_timestep"] = step
        time_logs["time_elapsed "] = self.duration(self.start_time)

        # Train
        train_logs["ep_len_mean"] = np.mean(train_ep_lens)
        (
            train_logs["ep_return_mean"],
            train_logs["std_return"],
            train_logs["max_return"],
            train_logs["min_return"],
        ) = get_stats(train_returns)

        # Eval
        eval_logs["ep_len_mean"] = np.mean(eval_ep_lens)
        (
            eval_logs["ep_return_mean"],
            eval_logs["std_return"],
            eval_logs["max_return"],
            eval_logs["min_return"],
        ) = get_stats(eval_returns)

        print("-" * 41)
        self.output_block(train_logs, tag="Train", color="invisible")
        self.output_block(eval_logs, tag="Evaluate", color="invisible")
        self.output_block(time_logs, tag="Time", color="invisible")
        print("\n")

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
        print("-" * 41)

    # -----------------------
    # Logging/Saving methods
    # -----------------------

    def metric_to_tb(
        self, step: int, train_logs: Dict[str, Any], eval_logs: Dict[str, Any]
    ) -> None:
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
        self.writer.add_scalar("return/test/ep_len", eval_logs.get("ep_len_mean"), step)
        self.writer.add_scalar(
            "return/test/ep_rew_mean", eval_logs.get("ep_return_mean"), step
        )
        self.writer.add_scalar(
            "return/test/ep_rew_std", eval_logs.get("std_return"), step
        )

    def info_to_tb(self, train_logs: Dict[str, Any], epoch: int) -> None:
        """Logging to tensorboard or wandb (if sync)"""
        assert train_logs is not None, "train log can not be `None`"
        if len(train_logs) > 0:
            self.writer.add_scalar("loss/actor", train_logs.get("actor_loss"), epoch)
            self.writer.add_scalar("loss/critic", train_logs.get("critic_loss"), epoch)
            self.writer.add_scalar(
                "info/actor/approx_kl", train_logs.get("approx_kl"), epoch
            )
            self.writer.add_scalar(
                "info/actor/entropy", train_logs.get("entropy"), epoch
            )
            self.writer.add_scalar(
                "info/actor/clip_fraction", train_logs["clip_fraction"], epoch
            )

    def save_models(self, save_dir: str, verbose: bool = False, **kwargs) -> None:
        # use algo.sav_mdoels directly for now
        # self.algo.save_models(save_dir, verbose)
        pass

    # -----------------------
    # Helper functions.
    # -----------------------

    def obs_as_tensor(
        self, obs: Union[dict, np.ndarray, th.Tensor], copy: bool = False
    ) -> Union[Dict[str, th.Tensor], th.Tensor]:
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

    def to_torch(
        self,
        array: np.ndarray,
        copy: bool = True,
    ) -> th.Tensor:
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

    @staticmethod
    def duration(start_time: float) -> str:
        return str(timedelta(seconds=int(time() - start_time)))

    def finish_logging(self) -> None:
        # Wait to ensure that all pending events have been written to disk.
        self.writer.flush()
        Console.info(
            "Wait to ensure that all pending events have been written to disk."
        )
        countdown(10)
        self.writer.close()
