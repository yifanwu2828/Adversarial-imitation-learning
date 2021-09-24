from abc import ABC, abstractmethod
from typing import Sequence, Union, Optional, Tuple

import torch as th
from torch import nn

from ail.common.pytorch_util import build_mlp


class BaseValue(nn.Module, ABC):
    """
    Basic class of a general Value or Q function
    :param state_dim: State dimensions of the environment
    :param action_dim: Action dimensions of the environment
    """

    def __init__(self, state_dim: int, action_dim: Optional[int] = None):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None

    @abstractmethod
    def forward(self, *args, **kwargs) -> th.Tensor:
        """
        Output of Value Network
        """
        raise NotImplementedError()

    @abstractmethod
    def get_value(self, *args, **kwargs) -> th.Tensor:
        """
        Squeeze Output of Value Network
        """
        raise NotImplementedError()


class StateFunction(BaseValue):
    """
    Basic implementation of a general state function (value function)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_units: Sequence[int],
        activation: Union[str, nn.Module],
        output_activation: Union[str, nn.Module] = nn.Identity(),
        **kwargs,
    ):
        super().__init__(obs_dim)
        self.net = build_mlp(
            [obs_dim] + list(hidden_units) + [1],
            activation,
            output_activation,
            kwargs.get("use_spectral_norm", False),
            kwargs.get("dropout_input", False),
            kwargs.get("dropout_hidden", False),
            kwargs.get("dropout_input_rate", 0.1),
            kwargs.get("dropout_hidden_rate", 0.1),
        )

    def forward(self, state: th.Tensor) -> th.Tensor:
        """self.net()"""
        return self.net(state)

    def get_value(self, state: th.Tensor) -> th.Tensor:
        return self.forward(state).squeeze(-1)


class StateActionFunction(BaseValue):
    """
    Basic implementation of a general state-action function (Q function)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_units: Sequence[int],
        activation: Union[str, nn.Module],
        output_activation: Union[str, nn.Module] = nn.Identity(),
        **kwargs,
    ):
        super().__init__(obs_dim, act_dim)
        self.net = build_mlp(
            [obs_dim + act_dim] + list(hidden_units) + [1],
            activation,
            output_activation,
            kwargs.get("use_spectral_norm", False),
            kwargs.get("dropout_input", False),
            kwargs.get("dropout_hidden", False),
            kwargs.get("dropout_input_rate", 0.1),
            kwargs.get("dropout_hidden_rate", 0.1),
        )

    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        return self.net(th.cat([state, action], dim=-1))

    def get_value(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        """The output has shape (n,)"""
        return self.net(th.cat([state, action], dim=-1)).squeeze(-1)


class TwinnedStateActionFunction(BaseValue):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_units: Sequence[int],
        activation: Union[str, nn.Module],
        output_activation: Union[str, nn.Module] = nn.Identity(),
        **kwargs,
    ):
        super().__init__(obs_dim, act_dim)

        self.net1 = build_mlp(
            [obs_dim + act_dim] + list(hidden_units) + [1],
            activation,
            output_activation,
        )
        self.net2 = build_mlp(
            [obs_dim + act_dim] + list(hidden_units) + [1],
            activation,
            output_activation,
        )

    def forward(
        self, states: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        xs = th.cat([states, actions], dim=-1)
        return self.net1(xs), self.net2(xs)

    def q1(self, states: th.Tensor, actions: th.Tensor) -> th.Tensor:
        return self.net1(th.cat([states, actions], dim=-1))

    def get_value(self, states: th.Tensor, actions: th.Tensor) -> th.Tensor:
        return self.net1(th.cat([states, actions], dim=-1)).squeeze(-1)


def mlp_value(
    state_dim: int,
    action_dim: int,
    value_layers: Sequence[int],
    activation: Union[nn.Module, str],
    val_type: str,
    output_activation: Union[str, nn.Module] = nn.Identity(),
    **kwargs,  # * use_spectral_norm should specified in kwargs
) -> nn.Module:
    val_type = val_type.lower()
    if val_type in ["v", "vs"]:
        return StateFunction(
            state_dim, value_layers, activation, output_activation, **kwargs
        )
    elif val_type in ["q", "qsa"]:
        return StateActionFunction(
            state_dim, action_dim, value_layers, activation, output_activation, **kwargs
        )
    elif val_type in ["twin", "twinstate"]:
        return TwinnedStateActionFunction(
            state_dim, action_dim, value_layers, activation, output_activation, **kwargs
        )
    else:
        raise ValueError(f"val_type: {val_type} not support.")
