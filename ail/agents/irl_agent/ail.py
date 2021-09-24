from typing import Union, Optional, Dict, Any

import torch as th

from ail.agents.irl_agent.adversarial import Adversarial
from ail.buffer import ReplayBuffer
from ail.common.type_alias import AlgoTags, GymSpace
from ail.network.discrim import DiscrimNet, DiscrimTag, DiscrimType


class AIRL(Adversarial):
    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        max_grad_norm: Optional[float],
        epoch_disc: int,
        replay_batch_size: int,
        buffer_exp: Union[ReplayBuffer, str],
        buffer_kwargs: Dict[str, Any],
        gen_algo,
        gen_kwargs: Dict[str, Any],
        disc_cls: Union[DiscrimNet, str],
        disc_kwargs: Dict[str, Any],
        lr_disc: float,
        disc_ent_coef: float = 0.0,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        subtract_logp: bool = False,
        rew_input_choice: str = "logit",
        rew_clip: bool = False,
        max_rew_magnitude: float = 10.0,
        min_rew_magnitude: Optional[float] = None,
        use_absorbing_state: bool = False,
        infinite_horizon: bool = False,
        **kwargs,
    ):

        if disc_kwargs is None:
            disc_kwargs = {}

        # Discriminator
        if isinstance(disc_cls, str):
            disc_cls = disc_cls.lower()
            if disc_cls not in ["airl_so", "airl_sa"]:
                raise ValueError(
                    f"No string string conversion of AIRL discriminator class {disc_cls}."
                )
            self.name = disc_cls.upper()
            disc_cls = DiscrimType[disc_cls].value

        else:
            if isinstance(disc_cls, DiscrimNet) and disc_cls.tag in list(DiscrimTag):
                if (
                    disc_cls.tag == DiscrimTag.AIRL_STATE_ONLY_DISCRIM
                    or disc_cls.tag == DiscrimTag.AIRL_STATE_ACTION_DISCRIM
                ):
                    disc_cls = DiscrimNet
                elif disc_cls.tag == DiscrimTag.GAIL_DISCRIM:
                    raise ValueError("Using GAIL DiscrimNet for AIRL is not Allowed.")
            else:
                raise ValueError(f"Unknown discriminator class: {disc_cls}.")
            self.name = None

        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            max_grad_norm,
            epoch_disc,
            replay_batch_size,
            buffer_exp,
            buffer_kwargs,
            gen_algo,
            gen_kwargs,
            disc_cls,
            disc_kwargs,
            lr_disc,
            disc_ent_coef,
            optim_kwargs,
            subtract_logp,
            "airl",
            rew_input_choice,
            rew_clip,
            max_rew_magnitude,
            min_rew_magnitude,
            use_absorbing_state,
            infinite_horizon,
            **kwargs,
        )
        self.tag = AlgoTags.AIRL

    def __repr__(self):
        if self.name is None:
            return self.__class__.__name__
        else:
            return f"{self.name}"


class GAIL(Adversarial):
    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        max_grad_norm: Optional[float],
        epoch_disc: int,
        replay_batch_size: int,
        buffer_exp: Union[ReplayBuffer, str],
        buffer_kwargs: Dict[str, Any],
        gen_algo,
        gen_kwargs: Dict[str, Any],
        disc_kwargs: Dict[str, Any],
        lr_disc: float,
        disc_ent_coef: float = 0.0,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        rew_input_choice: str = "logsigmoid",
        rew_clip: bool = False,
        max_rew_magnitude: float = 10.0,
        min_rew_magnitude: Optional[float] = None,
        use_absorbing_state: bool = False,
        infinite_horizon: bool = False,
        **kwargs,
    ):

        if disc_kwargs is None:
            disc_kwargs = {}

        # Discriminator
        disc_cls = DiscrimType["gail"].value

        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            max_grad_norm,
            epoch_disc,
            replay_batch_size,
            buffer_exp,
            buffer_kwargs,
            gen_algo,
            gen_kwargs,
            disc_cls,
            disc_kwargs,
            lr_disc,
            disc_ent_coef,
            optim_kwargs,
            False,
            "gail",
            rew_input_choice,
            rew_clip,
            max_rew_magnitude,
            min_rew_magnitude,
            use_absorbing_state,
            infinite_horizon,
        )
        self.tag = AlgoTags.GAIL
