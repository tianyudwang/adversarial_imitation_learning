from typing import Tuple, Optional

import numpy as np
import torch as th


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
        M_k = M_k-1+ (x_k – M_k-1)/k
        S_k = S_k-1 + (x_k – M_k-1)*(x_k – M_k).
        For 2 <= k <= n, the kth estimate of the variance is s**2 = S_k/(k – 1).
    """

    __slots__ = ["_n", "_M", "_S"]

    def __init__(self, shape: Tuple[int, ...]):
        self._n: int = 0
        self._M = np.zeros(shape, dtype=np.float64)
        self._S = np.zeros(shape, dtype=np.float64)

    def push(self, x) -> None:
        if isinstance(x, th.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        assert x.shape == self._M.shape
        
        self._n += 1

        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    def clear(self) -> None:
        self._n = 0
        self._M = np.zeros_like(self._M, dtype=np.float64)
        self._S = np.zeros_like(self._M, dtype=np.float64)
    
    def __repr__(self) -> str:
        return f"RunningStats(shape={self._M.shape}, mean={self.mean}, std={self.std})"    
    
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


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    :param shape: shape of input 
    :param center: If True, center the output by subtract running mean
    :param scale: If True, scale the output by dividing by running std
    :param clip: If not None, clip the output to be in [-clip,clip]
    :param eps: very small value to avoid divide by zero error
    """

    __slots__ = ["_shape", "rs", "center", "scale", "clip", "eps"]    
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        center: bool = True,
        scale: bool = True,
        clip: Optional[float] = None,
        eps: float = 1e-8
    ):
        assert isinstance(shape, tuple)
        self._shape = shape
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStats(shape)
        self.eps = eps
    
    def __call__(self, x, update=True):
        if update: 
            self.rs.push(x)

        if self.center:
            x = x - self.rs.mean
        if self.scale:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
    def __repr__(self) -> str:
        return f"ZFilterer(shape={self.shape}, center={self.center}, scale={self.scale}, clip={self.clip})"
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


class StateWithTime:
    '''
    Keeps track of the time t in an environment, and 
    adds t/T as a dimension to the state, where T is the 
    time horizon, given at initialization.
    '''
    def __init__(self, prev_filter, horizon):
        self.counter = 0
        self.horizon = horizon
        self.prev_filter = prev_filter

    def __call__(self, x, reset=False, count=True, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.counter += 1 if count else 0
        self.counter = 0 if reset else self.counter
        return np.array(list(x) + [self.counter/self.horizon])

    def reset(self):
        self.prev_filter.reset()
