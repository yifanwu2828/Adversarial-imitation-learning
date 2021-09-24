from typing import Tuple, Sequence, Union

import torch as th
from torch import nn

from ail.common.pytorch_util import build_mlp
from ail.common.math import reparameterize, evaluate_lop_pi
from ail.common.type_alias import Activation


class StateIndependentPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_units: Sequence[int],
        hidden_activation: Activation,
        output_activation: Union[str, nn.Module] = nn.Identity(),
    ):
        super().__init__()

        self.net = build_mlp(
            [obs_dim] + list(hidden_units) + [act_dim],
            hidden_activation,
            output_activation,
        )
        # TODO: allow log_std init
        self.log_stds = nn.Parameter(th.zeros(1, act_dim))

    def forward(self, states: th.Tensor):
        return th.tanh(self.net(states))

    def sample(self, states: th.Tensor):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states: th.Tensor, actions: th.Tensor):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_units: Sequence[int],
        hidden_activation: Activation,
        output_activation: Union[str, nn.Module] = nn.Identity(),
    ):
        super().__init__()

        self.net = build_mlp(
            [obs_dim] + list(hidden_units) + [2 * act_dim],
            hidden_activation,
            output_activation,
        )

    def forward(self, states: th.Tensor) -> th.Tensor:
        return th.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp(-20, 2))

    def evaluate_log_pi(self, states: th.Tensor, actions: th.Tensor) -> th.Tensor:
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return evaluate_lop_pi(means, log_stds, actions)
