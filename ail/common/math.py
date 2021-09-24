from math import pi, log
from itertools import accumulate
from typing import Union, List

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from scipy.signal import lfilter

from ail.common.utils import zip_strict


LOG2PI = log(2 * pi)


def pure_discount_cumsum(x: Union[list, np.ndarray], discount: float) -> List[float]:
    """
    Discount cumsum implemented in pure python.
    (For an input of size N,
    it requires O(N) operations and takes O(N) time steps to complete.)
    :param x: vector [x0, x1, x2]
    :param discount: float
    :return: list
    """
    # only works when x has shape (n,)
    acc = list(accumulate(x[::-1], lambda a, b: a * discount + b))
    return acc[::-1]


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    Note this is a faster when vector is large. (e.g: len(x) >= 1e3)
    :param x: vector [x0, x1, x2]
    :param discount: float
    :return:[x0 + discount * x1 + discount^2 * x2,   x1 + discount * x2, ... , xn]
    """
    # This function works better than pure python version when size of x is large
    # works in both [n,] (fast) and [n, 1](slow)
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def normalize(
    x: Union[np.ndarray, th.Tensor],
    mean: Union[np.ndarray, th.Tensor],
    std: Union[np.ndarray, th.Tensor],
    eps: float = 1e-8,
) -> Union[np.ndarray, th.Tensor]:
    """Normalize or standardize."""
    return (x - mean) / (std + eps)


def unnormalize(
    x: Union[np.ndarray, th.Tensor],
    mean: Union[np.ndarray, th.Tensor],
    std: Union[np.ndarray, th.Tensor],
) -> Union[np.ndarray, th.Tensor]:
    """Unnormalize or Unstandardize."""
    return x * std + mean


@th.jit.script
def fused_normalize(x: th.Tensor, mean: th.Tensor, std: th.Tensor, eps: float = 1e-8):
    """Normalize or standardize."""
    return (x - mean) / (std + eps)


@th.jit.script
def fused_unnormalize(x: th.Tensor, mean: th.Tensor, std: th.Tensor):
    """Unnormalize or Unstandardize."""
    return x * std + mean


def reparameterize(means: th.Tensor, log_stds: th.Tensor):
    """Reparameterize Trick."""
    noises = th.randn_like(means)
    us = fused_unnormalize(noises, means, log_stds.exp())
    actions = th.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)


def gaussian_logprobs(x: th.Tensor, log_stds: th.Tensor):
    """Calculate log probabilities for Gaussian Distribution"""
    return (-0.5 * x.pow(2) - log_stds).sum(
        dim=-1, keepdim=True
    ) - 0.5 * LOG2PI * log_stds.size(-1)


@th.jit.script
def atanh(x: th.Tensor):
    """Numerical stable atanh."""
    # pytorch's atanh does not clamp the value learning to Nan/inf
    return 0.5 * (th.log(1 + x + 1e-6) - th.log(1 - x + 1e-6))


def evaluate_lop_pi(
    means: th.Tensor, log_stds: th.Tensor, actions: th.Tensor
) -> th.Tensor:
    noises = fused_normalize(atanh(actions), means, log_stds.exp())
    return calculate_log_pi(log_stds, noises, actions)


def calculate_log_pi(
    log_stds: th.Tensor, noises: th.Tensor, actions: th.Tensor
) -> th.Tensor:
    """Calculate log probability of squash Gaussian."""
    correction = squash_logprob_correction(actions).sum(dim=-1, keepdim=True)
    return gaussian_logprobs(noises, log_stds) - correction


@th.jit.script
def squash_logprob_correction(actions: th.Tensor) -> th.Tensor:
    """
    Squash correction. (from original SAC implementation)
    log(1 - tanh(x)^2)
    # TODO: mark 1e-6
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
    return 2 * (log(2) - x - F.softplus(-2 * x))


def soft_update(
    target: nn.Module,
    source: nn.Module,
    tau: float,
    one: th.Tensor,
    safe_zip=False,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``
    target parameters are slowly updated towards the main parameters.
    :param target: Target network
    :param source: Source network
    :param tau: the soft update coefficient controls the interpolation:
        ``tau=1`` corresponds to copying the parameters to the target ones
        whereas nothing happens when ``tau=0``.
    :param one: dummy variable to equals to th.ones(1, device=device)
        Since it's a constant should pre-define it on proper device.
    :param safe_zip: if true, will raise error
        if source and target have different length of parameters.
    See https://github.com/DLR-RM/stable-baselines3/issues/93
    """
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        if safe_zip:
            # ! This is slow.
            for t, s in zip_strict(target.parameters(), source.parameters()):
                t.data.mul_(1.0 - tau)
                t.data.addcmul_(s.data, one, value=tau)
        else:
            # * Fast but safty not gurantee.
            # * should check if source and target have the same parameters outside.
            for t, s in zip(target.parameters(), source.parameters()):
                t.data.mul_(1.0 - tau)
                t.data.addcmul_(s.data, one, value=tau)
