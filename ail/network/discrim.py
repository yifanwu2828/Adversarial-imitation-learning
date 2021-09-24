from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence
from enum import Enum, auto

import torch as th
from torch import nn
import torch.nn.functional as F

from ail.network.value import StateFunction, StateActionFunction
from ail.common.type_alias import Activation


class ArchType(Enum):
    """Arch types of Discriminator"""

    s = auto()
    sa = auto()
    ss = auto()
    sas = auto()


class RewardType(Enum):
    airl = "airl"
    AIRL = "airl"
    gail = "gail"
    GAIL = "gail"
    inverse_gail = "inverse_gail"
    INVERSE_GAIL = "inverse_gail"


class ChoiceType(Enum):
    logit = "logit"
    logsigmoid = "logsigmoid"
    log_sigmoid = "log_sigmoid"
    softplus = "softplus"
    soft_plus = "soft_plus"


class DiscrimNet(nn.Module, ABC):
    """
    Abstract base class for discriminator, used in AIRL and GAIL.

    D = sigmoid(f)
    D(s, a) = sigmoid(f(s, a))
    D(s, a) = exp{f(s,a)} / (exp{f(s,a) + \pi(a|s)}
    where f is a discriminator logit (a learnable function represented as MLP)

    Choice of reward function:
    1. r(s, a) = − ln(1 − D) = softplus(h) \in [0, inf)
        (used in the original GAIL paper),

    2. r(s, a) = ln D − ln(1 − D) = h \in (-inf, inf)
        (introduced in AIRL).

    3. r(s, a) = ln D = −softplus(−h) \in (-inf, 0]
        (a natural choice we have not encountered in literature)

    # ! 4 Not Implemented.
    4. r(s, a) = −h exp(h) (introduced in FAIRL)
    View the diference: https://www.desmos.com/calculator/egzxzpi4b7

    * The original GAIL paper uses the inverse convention in which
        D denotes the probability as being classified as non-expert.
    * The FAIRL reward performed much worse than all others in the initial wide experiment
        therefore was not included in our main experiment.

    ---------------------------------------------------------------------------
    BIAS IN REWARD FUNCTIONS:
    (1) Strictly positive reward worked well for environments that require a survival bonus.
        (It encourages longer episodes)

    (2) Able to assign both positive and negative rewards for each time step.
        -Positive: this leading to sub-optimal policies (and training instability)
        in environments with a survival bonus.
        -Negative: this assigns rewards with a negative bias(in the beginning of training).
            It is common for learned agents to finish an episode
            earlier. (to avoid additional negative penalty)
            instead of trying to imitate the expert.

    (3) Strictly negative reward often used for tasks with a per step penalty.
        However, this variant assigns only negative rewards
        and cannot learn a survival bonus.
        (It encourages shorter episodes)

    **Also notes that the choice of a specific reward function might already
    provide strong prior knowledge that helps the RL algorithm
    to move towards recovering the expertpolicy,
    irrespective of the quality of the learned reward.

    More discussion: https://arxiv.org/pdf/1809.02925.pdf section 4.1, 4.1.1, 5.2

    ---------------------------------------------------------------------------
    The objective of the discriminator is to
    minimize cross-entropy loss
    between expert demonstrations and generated samples:

    L = \sum[ -E_{D} log(D) - E_{\pi} log(1 - D)]

    Write the negative loss to turn the minimization problem into maximization:
    -L = \sum[ -E_{D} log(D) + E_{\pi} log(1 - D)]

    """

    def __init__(
        self,
        disc_type: ArchType,
        state_dim: int,
        action_dim: Optional[int] = None,
        hidden_units: Sequence[int] = (128, 128),
        hidden_activation: Activation = nn.ReLU(inplace=True),
        init_model=True,
        **disc_kwargs,
    ):
        super().__init__()
        if disc_kwargs is None:
            disc_kwargs = {}

        # Net input
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

        # Regularization should be define in dsic_kwargs
        """
        Example:
            disc_kwargs: {
                'dropout_hidden': True,
                'dropout_hidden_rate': 0.1,
                'dropout_input': True,
                'dropout_input_rate': 0.1,
                'use_spectral_norm': False}
        """
        # Init Discriminator
        if init_model:
            self._init_model(disc_type, **disc_kwargs)

    def _init_model(self, disc_type: ArchType, **disc_kwargs) -> None:
        if disc_type == ArchType.s:
            self.hidden_units_r = disc_kwargs.get("hidden_units_r", self.hidden_units)
            self.hidden_units_v = disc_kwargs.get("hidden_units_v", self.hidden_units)
            self.g = StateFunction(
                self.state_dim,
                self.hidden_units_r,
                self.hidden_activation,
                **disc_kwargs,
            )
            self.h = StateFunction(
                self.state_dim,
                self.hidden_units_v,
                self.hidden_activation,
                **disc_kwargs,
            )

        elif disc_type == ArchType.sa:
            self.f = StateActionFunction(
                self.state_dim,
                self.action_dim,
                self.hidden_units,
                self.hidden_activation,
                **disc_kwargs,
            )

        elif disc_type == ArchType.ss:
            raise NotImplementedError(f"disc_type: {disc_type} not implemented.")

        elif disc_type == ArchType.sas:
            raise NotImplementedError(f"disc_type: {disc_type} not implemented.")

        else:
            raise NotImplementedError(
                f"Type {self.disc_type} is not supported or arch not provide in dist_kwargs."
            )

    @abstractmethod
    def forward(self, *args, **kwargs) -> th.Tensor:
        """Output logits of discriminator."""
        raise NotImplementedError()

    @abstractmethod
    def calculate_rewards(self, *args, **kwargs) -> th.Tensor:
        """Calculate learning rewards based on choice of reward formulation."""
        raise NotImplementedError()

    def reward_fn(self, rew_type: str, choice: str) -> Callable[[th.Tensor], th.Tensor]:
        """
        The learning rewards formulation.
        (GAIL):r(s, a) = − ln(1 − D) = softplus(h)
        (AIRL): r(s, a) = ln D − ln(1 − D) = h

        Paper:"What Matters for Adversarial Imitation Learning?" Appendix C.2.
        See: https://arxiv.org/abs/2106.00672

        :param rew_type: airl or gail
        :param choice: logsigmoid, sofplus, logit
        Note logit only available in airl and returns itself without any transformation.

        LHS equation and RHS equation are mathmatically identical why implement both?
        Because Pytorch's logsigmoid and softplus behaves differently in the same reward function.
        Might due to the threshold value in softplus.
        Refer to https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
        """
        rew_types = {"gail", "airl"}
        choices = {"logsigmoid", "softplus", "logit"}

        rew_type = RewardType[rew_type.lower()].value
        choice = ChoiceType[choice.lower()].value

        if rew_type == "gail":
            # * (1)  − ln(1 − D) = softplus(h)
            if choice == "logsigmoid":
                return self.gail_logsigmoid
            elif choice == "softplus":
                return self.gail_softplus
            elif choice == "logit":
                raise ValueError(f"Choice logit not supported for Gail.")
            else:
                raise ValueError(
                    f"Choice {choices} not supported with rew_type gail. "
                    f"Valid choices are {choices}."
                )

        elif rew_type == "airl":
            # * (2)  ln D − ln(1 − D) = h = −softplus(-h) + softplus(h)
            if choice == "logsigmoid":
                return self.airl_logsigmoid
            elif choice == "softplus":
                return self.airl_softplus
            elif choice == "logit":
                return self.airl_logit
            else:
                raise ValueError(
                    f"Choice {choices} not supported. Valid choices are {choices}."
                )

        elif rew_type == "inverse_gail":
            if choice == "logsigmoid":
                return self.airl_logsigmoid
            elif choice == "softplus":
                return self.airl_softplus

        else:
            raise ValueError(
                f"Reward type {rew_type} not supported. "
                f"Valid rew_types: {rew_types}"
            )

    @staticmethod
    def gail_logsigmoid(x: th.Tensor) -> th.Tensor:
        """
        (GAIL):r(s, a) = − ln(1 − D)
        :param x: logits
        """
        return -F.logsigmoid(-x)

    @staticmethod
    def gail_softplus(x: th.Tensor) -> th.Tensor:
        """
        (GAIL):r(s, a) = softplus(h)
        :param x: logits
        """
        return F.softplus(x)

    @staticmethod
    def airl_logsigmoid(x: th.Tensor) -> th.Tensor:
        """
        (AIRL): r(s, a) = ln D − ln(1 − D)
        :param x: logits
        """
        return F.logsigmoid(x) - F.logsigmoid(-x)

    @staticmethod
    def airl_softplus(x: th.Tensor) -> th.Tensor:
        """
        (AIRL): r(s, a) = -softplus(-x) + softplus(x)
        :param x: logits
        """
        return -F.softplus(-x) + F.softplus(x)

    @staticmethod
    def airl_logit(x: th.Tensor) -> th.Tensor:
        """
        (AIRL): r(s, a) = ln D − ln(1 − D) = h
        where h is the logits. Output of f net/function.
        :param x: logits
        """
        return x

    @staticmethod
    def inverse_gail_logsigmoid(x: th.Tensor) -> th.Tensor:
        return F.logsigmoid(x)

    @staticmethod
    def inverse_gail_sofplus(x: th.Tensor) -> th.Tensor:
        return -F.softplus(-x)


class DiscrimTag(Enum):
    GAIL_DISCRIM = auto()
    AIRL_STATE_ONLY_DISCRIM = auto()
    AIRL_STATE_ACTION_DISCRIM = auto()


class GAILDiscrim(DiscrimNet):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Sequence[int],
        hidden_activation: Activation,
        **disc_kwargs,
    ):
        if disc_kwargs is None:
            disc_kwargs = {}

        super().__init__(
            ArchType.sa,
            state_dim,
            action_dim,
            hidden_units,
            hidden_activation,
            **disc_kwargs,
        )
        self._tag = DiscrimTag.GAIL_DISCRIM
        self.inverse = disc_kwargs.get("inverse", False)

    @property
    def tag(self):
        return self._tag

    def forward(self, obs: th.Tensor, acts: th.Tensor, **kwargs):
        """
        Output logits of discriminator.
        Naming `f` to keep consistent with base DiscrimNet.
        """
        return self.f(obs, acts)

    def calculate_rewards(
        self, obs: th.Tensor, acts: th.Tensor, choice: str = "logsigmoid", **kwargs
    ):
        """
        (GAIL) is to maximize E_{\pi} [-log(1 - D)].
        r(s, a) = − ln(1 − D) = softplus(h)
        """
        with th.no_grad():
            if self.inverse:
                reward_fn = self.reward_fn("inverse_gail", choice)
            else:
                reward_fn = self.reward_fn("gail", choice)
            logits = self.forward(obs, acts, **kwargs)
            rews = reward_fn(logits)
        return rews


class AIRLStateDiscrim(DiscrimNet):
    """
    Discriminator used in AIRL with disentangled reward.
    f_{θ,φ} (s, a, s') = g_θ (s, a) + \gamma h_φ (s') − h_φ (s)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Sequence[int],
        hidden_activation: Activation,
        gamma: float,
        **disc_kwargs,
    ):
        if disc_kwargs is None:
            disc_kwargs = {}
        super().__init__(
            ArchType.s,
            state_dim,
            None,
            hidden_units,
            hidden_activation,
            **disc_kwargs,
        )
        self.gamma = gamma
        self._tag = DiscrimTag.AIRL_STATE_ONLY_DISCRIM

    @property
    def tag(self):
        return self._tag

    def f(
        self, obs: th.Tensor, dones: th.FloatTensor, next_obs: th.Tensor
    ) -> th.Tensor:
        """
        f(s, a, s' ) = g_θ (s) + \gamma h_φ (s') − h_φ (s)

        f: recover to the advantage
        g: state-only reward function approximator
        h: shaping term
        """
        r_s = self.g(obs)
        v_s = self.h(obs)
        next_vs = self.h(next_obs)
        # * Reshape (1-done) to (n,1) to prevent boardcasting mismatch in case done is (n,).
        # Change to match back to convention of dones where done = 1 not_done = 0
        dones = 1.0 - th.where(dones <= 0, 0, 1)
        return r_s + self.gamma * (1 - dones).view(-1, 1) * next_vs - v_s

    def forward(
        self,
        obs: th.Tensor,
        dones: th.Tensor,
        next_obs: th.Tensor,
        log_pis: Optional[th.Tensor] = None,
        subtract_logp: bool = True,
        **kwargs,
    ) -> th.Tensor:
        """
        Policy Objective.
        \hat{r}_t = log[D_θ(s,a)] - log[1-D_θ(s,a)]
        = log[exp{f_θ} /(exp{f_θ} + \pi)] - log[\pi / (exp{f_θ} + \pi)]
        = f_θ (s,a) - log \pi (a|s)
        """
        if log_pis is not None and subtract_logp:
            # reshape log_pi to prevent size mismatch
            return self.f(obs, dones, next_obs) - log_pis.view(-1, 1)
        elif log_pis is None and subtract_logp:
            raise ValueError("log_pis is None! Can not subtract None.")
        else:
            return self.f(obs, dones, next_obs)

    def calculate_rewards(
        self,
        obs: th.Tensor,
        dones: th.Tensor,
        next_obs: th.Tensor,
        log_pis: Optional[th.Tensor] = None,
        subtract_logp: bool = True,
        choice: str = "logit",
        **kwargs,
    ) -> th.Tensor:
        """
        Calculate GAN reward.
        """
        kwargs = {
            "dones": dones,
            "next_obs": next_obs,
            "log_pis": log_pis,
            "subtract_logp": subtract_logp,
        }
        with th.no_grad():
            reward_fn = self.reward_fn(rew_type="airl", choice=choice)
            logits = self.forward(obs, **kwargs)
            rews = reward_fn(logits)
        return rews


class AIRLStateActionDiscrim(DiscrimNet):
    """
    Discriminator used in AIRL with entangled reward.
    As in the trajectory-centric case,
        f* (s, a) = log π* (a|s) = A*(s, a)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Sequence[int],
        hidden_activation: Activation,
        **disc_kwargs,
    ):
        if disc_kwargs is None:
            disc_kwargs = {}

        super().__init__(
            ArchType.sa,
            state_dim,
            action_dim,
            hidden_units,
            hidden_activation,
            **disc_kwargs,
        )
        self._tag = DiscrimTag.AIRL_STATE_ACTION_DISCRIM

    @property
    def tag(self):
        return self._tag

    def forward(
        self,
        obs: th.Tensor,
        acts: th.Tensor,
        log_pis: Optional[th.Tensor] = None,
        subtract_logp: bool = True,
        **kwargs,
    ) -> th.Tensor:
        if log_pis is not None and subtract_logp:
            # Reshape log_pi to prevent size mismatch.
            return self.f(obs, acts) - log_pis.view(-1, 1)
        elif log_pis is None and subtract_logp:
            raise ValueError("log_pis is None! Cannot subtract None.")
        else:
            return self.f(obs, acts)

    def calculate_rewards(
        self,
        obs: th.Tensor,
        acts: th.Tensor,
        log_pis: Optional[th.Tensor] = None,
        subtract_logp: bool = True,
        choice: str = "logit",
        **kwargs,
    ) -> th.Tensor:
        kwargs = {
            "acts": acts,
            "log_pis": log_pis,
            "subtract_logp": subtract_logp,
        }
        # TODO: apply reward bound
        with th.no_grad():
            reward_fn = self.reward_fn(rew_type="airl", choice=choice)
            logits = self.forward(obs, **kwargs)
            rews = reward_fn(logits)
        return rews


class DiscrimType(Enum):
    gail = GAILDiscrim
    airl_so = AIRLStateDiscrim
    airl_sa = AIRLStateActionDiscrim
