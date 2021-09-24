from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import Adam, AdamW


# Gym type
GymEnv = gym.Env
GymWrapper = gym.Wrapper
GymSpace = gym.spaces.Space
GymDict = gym.spaces.Dict
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]

# tensor type
TensorDict = Dict[Union[str, int], th.Tensor]
Activation = Union[str, nn.Module]

# ----------------------------------------------------------------
# string to object naming conventions
class StrToActivation(Enum):
    """Torch activation function."""

    relu = nn.ReLU()
    relu_inplace = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    leaky_relu = nn.LeakyReLU()
    sigmoid = nn.Sigmoid()
    selu = nn.SELU()
    softplus = nn.Softplus()
    identity = nn.Identity()


class OPT(Enum):
    """Torch optimizers."""

    adam = Adam
    adamw = AdamW
    adam_w = AdamW


# ----------------------------------------------------------------
# Buffer shape and dtype
@dataclass(frozen=True, eq=False)
class Extra_shape:
    """Shape of extra data store in buffer."""

    advs: Tuple[int, ...] = (1,)
    rets: Tuple[int, ...] = (1,)
    vals: Tuple[int, ...] = (1,)
    log_pis: Tuple[int, ...] = (1,)
    remaining_steps: Tuple[int, ...] = (1,)


@dataclass(frozen=True, eq=False)
class Extra_dtypes:
    """Dtypes of extra data store in buffer."""

    advs: np.dtype = np.float32
    rets: np.dtype = np.float32
    vals: np.dtype = np.float32
    log_pis: np.dtype = np.float32
    remaining_steps: np.dtype = np.float32


EXTRA_SHAPES = Extra_shape()
EXTRA_DTYPES = Extra_dtypes()


class AlgoTags(Enum):
    PPO = auto()
    SAC = auto()
    AIRL = auto()
    GAIL = auto()


class DoneMask(Enum):
    ABSORBING = -1.0
    DONE = 0.0
    NOT_DONE = 1.0
