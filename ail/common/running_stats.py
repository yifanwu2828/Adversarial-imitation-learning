from typing import Tuple

import numpy as np
import torch as th


class RunningMeanStd:
    """
    Calulates the running mean and std of a data stream
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    :param epsilon: helps with arithmetic issues
    :param shape: the shape of the data stream's output

    The algorithm is as follows:
        n = n_a + n_b
        delta = avg_b - avg_a
        M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / n
        var_ab = M2 / (n - 1)
    """

    __slots__ = ["mean", "var", "count", "shape", "epsilon"]

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):

        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon
        self.shape = shape
        self.epsilon = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        new_count = tot_count

        self.mean, self.var, self.count = new_mean, new_var, new_count

    def std(self):
        return np.sqrt(self.var)

    def __repr__(self) -> str:
        return f"RunningMeanStd(shape={self.shape}, mean={self.mean}, std={self.std})"

    def push(self, x):
        x = np.asarray(x).reshape(self.shape)
        self.update(x)

    def clear(self) -> None:
        self.mean = np.zeros(self.shape, np.float64)
        self.var = np.ones(self.shape, np.float64)
        self.count = self.epsilon


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
        from icecream import ic

        ic(x.shape, self._M.shape)
        assert x.shape == self._M.shape

        self._n += 1

        if self._n == 1:
            self._M[...] = x
            oldM = self._M.copy()
            self._S[...] = self._S + (x - oldM) * (x - self._M)
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    def clear(self) -> None:
        self._n = 0
        self._M = np.zeros_like(self._M, dtype=np.float64)
        self._S = np.zeros_like(self._M, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.push(x)

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
