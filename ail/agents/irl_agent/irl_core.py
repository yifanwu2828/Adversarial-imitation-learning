from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Dict, Any
import os

import numpy as np
import torch as th
from torch import nn
from torch.nn.utils import clip_grad_norm_


from ail.agents.base import BaseAgent
from ail.agents.rl_agent import RL_ALGO
from ail.agents.rl_agent.rl_core import OnPolicyAgent, OffPolicyAgent
from ail.buffer import ReplayBuffer, BufferType
from ail.common.type_alias import GymEnv, GymSpace
from ail.common.pytorch_util import count_vars


class BaseIRLAgent(BaseAgent, ABC):
    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        max_grad_norm: Optional[float],
        replay_batch_size: int,
        buffer_exp: Union[ReplayBuffer, str],
        buffer_kwargs: Optional[Dict[str, Any]],
        gen_algo: Union[OnPolicyAgent, OffPolicyAgent, str],
        gen_kwargs: Dict[str, Any],
        optim_kwargs: Optional[Dict[str, Any]],
    ):
        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            optim_kwargs,
        )

        if buffer_kwargs is None:
            buffer_kwargs = {}

        # Expert's buffer.
        self.replay_batch_size = replay_batch_size
        if isinstance(buffer_exp, ReplayBuffer):
            # Assume expert's buffer is already initialized.
            self.buffer_exp = buffer_exp
        elif isinstance(buffer_exp, str):
            # Initialize the buffer from a dictionary of data.
            if len(buffer_kwargs) == 0:
                raise ValueError("Need specifies buffer_kwargs for replay buffer.")
            self.buffer_exp = BufferType[buffer_exp].value.from_data(
                device=self.device, **buffer_kwargs
            )
        else:
            raise ValueError(f"Unsupported buffer type: {buffer_exp}")

        # Generator
        gen_cls = RL_ALGO[gen_algo] if isinstance(gen_algo, str) else gen_algo
        self.gen = gen_cls(
            self.state_space,
            self.action_space,
            **gen_kwargs,
        )

        # Other parameters.
        self.max_grad_norm = max_grad_norm
        self.clipping = max_grad_norm is not None

        # Create Alias
        self.actor = self.gen.actor

    def info(self) -> Dict[nn.Module, int]:
        """
        Count variables.
        (protip): try to get a feel for how different size networks behave!
        """
        models = (self.gen.actor, self.gen.critic, self.disc)
        return {module: count_vars(module) for module in models}

    def step(
        self, env: GymEnv, state: th.Tensor, t: th.Tensor, step: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Same as generator/policy step.
        Intereact with environment and store the transition.
        return: next_state, episode length
        """
        return self.gen.step(env, state, t, step)

    def explore(self, state: th.Tensor, scale: bool) -> Tuple[np.ndarray, th.Tensor]:
        """Same as generator/policy explore."""
        return self.gen.explore(state, scale)

    def exploit(self, state, scale: bool) -> np.ndarray:
        """Same as generator/policy exploit"""
        return self.gen.exploit(state, scale)

    def is_update(self, step: int) -> bool:
        return self.gen.is_update(step)

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main loop
         1. Interact with the environment using the current generator/ policy.
            and store the experience in a replay buffer (implementing in step()).
         2. Update discriminator.
         3. Update generator.
        """
        raise NotImplementedError()

    @abstractmethod
    def update_generator(self) -> Dict[str, Any]:
        """Train generator (RL policy)"""
        raise NotImplementedError()

    @abstractmethod
    def update_discriminator(self) -> Dict[str, Any]:
        """Train discriminator"""
        raise NotImplementedError()

    def save_models(self, save_dir: str) -> None:
        """Save both actor and discrimNet"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Generator
        th.save(self.actor.state_dict(), os.path.join(save_dir, "gen_actor.pth"))

        # Discriminator
        th.save(self.disc.state_dict(), os.path.join(save_dir, "discrim.pth"))

    def one_gradient_step(
        self,
        loss: th.Tensor,
        opt: th.optim.Optimizer,
        net: nn.Module,
    ) -> None:
        """Take one gradient step with grad clipping.(AMP support)"""
        if self.fp16:
            # AMP.
            self.scaler.scale(loss).backward()
            if self.clipping:
                self.scaler.unscale_(opt)
                clip_grad_norm_(net.parameters(), max_norm=self.max_grad_norm)
            self.scaler.step(opt)
            self.scaler.update()
        else:
            # Unscale_update.
            loss.backward()
            if self.clipping:
                clip_grad_norm_(net.parameters(), max_norm=self.max_grad_norm)
            opt.step()
