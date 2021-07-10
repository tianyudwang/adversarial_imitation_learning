from math import pi, log
from itertools import accumulate

import numpy as np
import torch as th
import torch.nn.functional as F
from scipy.signal import lfilter

LOG_2 = log(2)
LOG2PI = log(2 * pi)


def pure_discount_cumsum(x, discount) -> list:
    """
    Discount cumsum implemented in pure python
    (For an input of size N,
    it requires O(N) operations and takes O(N) time steps to complete.)
    :param x: vector [x0, x1, x2]
    :param discount: float
    :return: list
    """
    # only works when x has shape (n,)
    acc = list(accumulate(x[::-1], lambda a, b: a * discount + b))
    return acc[::-1]


def discount_cumsum(x, discount) -> np.ndarray:
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    Note this is a faster when vector is large (e.g: len(x) >= 1e3)
    :param x: vector [x0, x1, x2]
    :param discount: float
    :return:[x0 + discount * x1 + discount^2 * x2,   x1 + discount * x2, ... , xn]
    """
    # This function works better than pure python version when size of x is large
    # works in both [n,] (fast) and [n, 1](slow)
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def normalize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)


def unnormalize(x, mean, std):
    return x * std + mean


def reparameterize(means: th.Tensor, log_stds: th.Tensor):
    noises = th.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = th.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)


def atanh(x: th.Tensor):
    # pytorch's atanh does not clamp the value learning to Nan/inf
    return 0.5 * (th.log(1 + x + 1e-6) - th.log(1 - x + 1e-6))


def evaluate_lop_pi(means: th.Tensor, log_stds: th.Tensor, actions: th.Tensor):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True
    ) - 0.5 * LOG2PI * log_stds.size(-1)

    correction = squash_logprob_correction(actions).sum(dim=-1, keepdim=True)

    return gaussian_log_probs - correction


def squash_logprob_correction(actions: th.Tensor) -> th.Tensor:
    """
    Squash correction (from original SAC implementation)
    log(1 - tanh(x)^2)
    this code is more numerically stable.
    (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py#L195)
    Derivation:
    = log(sech(x)^2)
    = 2 * log(sech(x))
    = 2 * log(2e^-x / (e^-2x + 1))
    = 2 * (log(2) - x - log(e^-2x + 1))
    = 2 * (log(2) - x - softplus(-2x))
    :param actions:
    """
    x = atanh(actions)
    return 2 * (LOG_2 - x - F.softplus(-2 * x))
