from typing import Tuple
from math import sqrt

import numpy as np


class RunningStats:
    """
    Welford’s method
    Based on (https://www.johndcook.com/standard_deviation.html).
    This algorithm is much less prone to loss of precision due to catastrophic cancellation,
    but might not be as efficient because of the division operation inside the loop.
    """

    __slots__ = ["n", "old_m", "new_m", "old_s", "new_s"]

    def __init__(self):
        self.n: int = 0
        self.old_m: float = 0
        self.new_m: float = 0
        self.old_s: float = 0
        self.new_s: float = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:(n={self.n}, mean={self.mean()}, std={self.standard_deviation()})"

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return sqrt(self.variance())


class RunningMeanStd:
    """
    Parallel algorithm
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Chan's method for estimating the mean is numerically unstable
    when n_A \approx n_B and both are large,
    because the numerical error in \delta ={\bar {x}}_{B}-{\bar {x}}_{A}}
    is not scaled down in the way that it is in the n_B = 1 case.
    :param epsilon: helps with arithmetic issues
    :param shape: the shape of the data stream's output
    """

    __slots__ = ["mean", "var", "count"]

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
