from typing import Union, Optional, Dict, Any, Tuple
import warnings

from gym.spaces import Box
import numpy as np
import torch as th
from torch import nn
from torch.cuda.amp import GradScaler


from ail.common.type_alias import OPT, GymSpace
from ail.common.utils import set_random_seed
from ail.common.pytorch_util import init_gpu
from ail.common.env_utils import get_obs_shape, get_flat_obs_dim, get_act_dim


class BaseAgent(nn.Module):
    """
    Base class for all agents.
    :param state_space: state space.
    :param action_space: action space.
    :param device: PyTorch device to which the values will be converted.
    :param fp16: Whether to use float16 mixed precision training.
    :param seed: random seed.
    :optim_kwargs: arguments to be passed to the optimizer.
        eg. : {
            "optim_cls": adam,
            "optim_set_to_none": True, # which set grad to None instead of zero.
            }
    """

    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        optim_kwargs: Optional[Dict[str, Any]],
    ):
        super().__init__()

        # RNG.
        if not isinstance(seed, int):
            raise ValueError("seed must be integer.")
        self.seed = seed
        set_random_seed(self.seed)

        # env spaces.
        self._state_space = state_space
        self._action_space = action_space

        # Shapes of space useful for buffer.
        self._state_shape = get_obs_shape(state_space)
        if isinstance(action_space, Box):
            self._action_shape = action_space.shape
        else:
            raise NotImplementedError()

        # Space dimension and action dimension.
        self._obs_dim = get_flat_obs_dim(state_space)
        self._act_dim = get_act_dim(action_space)

        # Action limits.
        self._act_low = action_space.low
        self._act_high = action_space.high

        assert (self._act_low < self._act_high).all()
        self._act_half_range = (self._act_high - self._act_low) / 2.0

        self._symmetric_action_space = all(
            [
                isinstance(self._action_space, Box),
                (abs(self._act_low) == abs(self._act_high)).all(),
            ]
        )

        self._normalized_action_space = all(
            [
                isinstance(self._action_space, Box),
                (self._act_low == -1).all(),
                (self._act_high == 1).all(),
                self._symmetric_action_space,
            ]
        )

        if not self._symmetric_action_space:
            warnings.warn(f"Not symmetric action_space.")

        # Device management.
        self.device = init_gpu(use_gpu=(device == "cuda"), verbose=True)

        # Use automatic mixed precision training in GPU
        self.fp16 = all([fp16, th.cuda.is_available(), device == "cuda"])
        self.scaler = GradScaler() if self.fp16 else None

        # Optimizer kwargs.
        optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        optim_cls = optim_kwargs.get("optim_cls", "adam")
        self.optim_set_to_none = optim_kwargs.get("optim_set_to_none", False)

        if isinstance(optim_cls, str):
            self.optim_cls = OPT[optim_cls.lower()].value
        elif isinstance(optim_cls, th.optim.Optimizer):
            self.optim_cls = optim_cls
        else:
            raise ValueError("optim_cls must be a string or an torch. optim.Optimizer.")

    @property
    def state_space(self) -> GymSpace:
        return self._state_space

    @property
    def action_space(self) -> GymSpace:
        return self._action_space

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return self._state_shape

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return self._action_shape

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        return self._act_dim

    @property
    def act_low(self) -> np.ndarray:
        return self._act_low

    @property
    def act_high(self) -> np.ndarray:
        return self._act_high

    @property
    def act_half_range(self) -> np.ndarray:
        return self._act_half_range

    @property
    def symmetric_action_space(self) -> bool:
        return self._symmetric_action_space

    @property
    def normalized_action_space(self) -> bool:
        return self._normalized_action_space
