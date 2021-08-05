from typing import Tuple

import numpy as np


# Taken from: https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/running_stat.py#L4
class RunningStats:
    """
    Welford’s method: keeps track of first and second moments (mean and variance)
    of a streaming time series.
    Based on (https://www.johndcook.com/standard_deviation.html).
    This algorithm is much less prone to loss of precision due to catastrophic cancellation,
    but might not be as efficient because of the division operation inside the loop.

    The algorithm is as follows:
        Initialize M1 = x1 and S1 = 0.
        For subsequent x‘s, use the recurrence formulas
        Mk = Mk-1+ (xk – Mk-1)/k
        Sk = Sk-1 + (xk – Mk-1)*(xk – Mk).
        For 2 ≤ k ≤ n, the kth estimate of the variance is s2 = Sk/(k – 1).
    """

    __slots__ = ["_n", "_M", "_S"]

    def __init__(self, shape: Tuple[int, ...]):
        self._n: int = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    def clear(self):
        self._n = 0
        self._M = np.zeros_like(self._M)
        self._S = np.zeros_like(self._M)     
    
    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> np.ndarray:
        return self._M

    @property
    def var(self) -> np.ndarray:
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    @property
    def shape(self) -> np.ndarray:
        return self._M.shape
