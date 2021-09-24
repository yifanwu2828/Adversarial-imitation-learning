from typing import Union, Optional, Tuple, Dict, Any
from collections import defaultdict
from copy import deepcopy
import os

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ail.agents.rl_agent.rl_core import OffPolicyAgent
from ail.common.math import soft_update
from ail.common.pytorch_util import (
    asarray_shape2d,
    count_vars,
    disable_gradient,
    enable_gradient,
    obs_as_tensor,
    to_torch,
)
from ail.common.type_alias import AlgoTags, DoneMask, GymEnv, GymSpace, TensorDict
from ail.network.policies import StateDependentPolicy
from ail.network.value import mlp_value
from ail.color_console import Console


class SAC(OffPolicyAgent):

    """
    Soft Actor-Critic (SAC)
    Paper: https://arxiv.org/abs/1801.01290
    :param state_space: state space
    :param action_space: action space
    :param device: PyTorch device to which the values will be converted.
    :param seed: random seed.
    :param batch_size: size of the batch.
    :param buffer_size: size of the buffer.
    :param policy_kwargs: arguments to be passed to the policy on creation.
        e.g. : {
            pi: [64, 64],
            vf: [64, 64],
            activation: 'relu',
            lr_actor: 3e-4,
            lr_critic: 3e-4
            critic_type="twin",
            }
    :param lr_alpha: learning rate for the entropy coefficient.
    :param start_steps: how many steps of the model to collect transitions before learning starts.
    :param num_gradient_steps: How many gradient steps to do after each rollout
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param log_alpha_init: Init value of log_alpha.
    :param gamma: discount factor.
    :param tau: soft update coefficient. ("Polyak update", between 0 and 1)
    :param max_grad_norm: Maximum norm for the gradient clipping.
    :param fp16: Whether to use float16 mixed precision training.
    :optim_kwargs: arguments to be passed to the optimizer.
        eg. : {
            "optim_cls": adam,
            "optim_set_to_none": True, # which set grad to None instead of zero.
            }
    :param buffer_kwargs: arguments to be passed to the buffer.
        eg. : {
            with_reward:True,
            extra_data:["log_pis"]
            }
    :param init_buffer: Whether to create the buffer during initialization.
    :param init_models: Whether to create the models during initialization.
    """

    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        seed: int,
        policy_kwargs: Dict[str, Any],
        batch_size: int = 256,
        buffer_size: int = 10 ** 6,
        lr_alpha: float = 3e-4,
        start_steps: int = 10_000,
        num_gradient_steps: int = 1,
        target_update_interval: int = 1,
        log_alpha_init: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_grad_norm: Optional[float] = None,
        fp16: bool = False,
        optim_kwargs=None,
        buffer_kwargs: Optional[Dict[str, Any]] = None,
        init_buffer: bool = True,
        init_models: bool = True,
        expert_mode: bool = False,
        use_as_generator: bool = False,
        use_absorbing_state: bool = False,
        **kwargs,
    ):
        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            gamma,
            max_grad_norm,
            batch_size,
            buffer_size,
            policy_kwargs,
            optim_kwargs,
            buffer_kwargs,
            init_buffer,
            init_models,
            expert_mode,
        )
        if expert_mode:
            self.actor = StateDependentPolicy(
                self.obs_dim, self.act_dim, self.units_actor, "relu_inplace"
            ).to(self.device)
        else:
            self._setup_models()
        # Entropy regularization coefficient (Inverse of the reward scale)
        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        self.lr_alpha = lr_alpha

        # Enforces an entropy constraint by varying alpha over the course of training.
        # We optimize log(alpha) because alpha should be always bigger than 0.
        self.log_alpha = th.log(
            th.ones(1, device=self.device) * log_alpha_init
        ).requires_grad_(True)

        # Target entropy is -|A|.
        self.target_entropy = -np.prod(self.action_shape).astype(np.float32)
        self.optim_alpha = self.optim_cls([self.log_alpha], lr=self.lr_alpha)

        # Other algo params.
        self.start_steps = start_steps
        self.num_gradient_steps = num_gradient_steps
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.one = th.ones(1, device=self.device)  # a constant for quick soft update

        self.tag = AlgoTags.SAC
        self.use_as_generator = use_as_generator
        self.use_absorbing_state = use_absorbing_state

        if self.use_as_generator:
            Console.info("Using as generator.")
        if self.use_absorbing_state:
            Console.info("wrapped absorbing state.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def _setup_models(self) -> None:
        # TODO: (Yifan) Build the model inside off policy class latter.

        # Actor.
        self.actor = StateDependentPolicy(
            self.obs_dim, self.act_dim, self.units_actor, self.hidden_activation
        ).to(self.device)

        # Critic.
        self.critic = mlp_value(
            self.obs_dim,
            self.act_dim,
            self.units_critic,
            self.hidden_activation,
            self.critic_type,
        ).to(self.device)

        # Set target param equal to main param. Using deep copy instead.
        self.critic_target = deepcopy(self.critic).to(self.device).eval()

        # Sanity check for zip() in soft_update.
        assert count_vars(self.critic_target) == count_vars(self.critic)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        disable_gradient(self.critic_target)

        # Config optimizer
        self.optim_actor = self.optim_cls(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = self.optim_cls(self.critic.parameters(), lr=self.lr_critic)

    def is_update(self, step: int) -> bool:
        """Whether or not to update the agent"""
        return step >= max(self.start_steps, self.batch_size)

    def step(
        self,
        env: GymEnv,
        state: th.Tensor,
        episode_timesteps: th.Tensor,
        global_timesteps: Optional[int] = None,
        max_ep_len: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Intereact with environment and store the transition.
        A trick to improve exploration at the start of training (for a fixed number of steps)
        Agent takes actions which are sampled from a uniform random distribution over valid actions.
        After that, it returns to normal SAC exploration.

        :param env: gym environment
        :param state: orginal state return by the environment
        :param episode_timesteps: number of timesteps this episode
        :param total_timesteps: total number of timesteps to run in outer loop
        :return: next_state, episode length
        """
        if max_ep_len is None:
            max_ep_len = env._max_episode_steps

        # Random exploration step
        if global_timesteps <= self.start_steps:
            # Random uniform sampling from env's action space.
            scaled_action = env.action_space.sample()
            # * Need to apply tanh transoform to the action to make it in range [-1, 1]
            action = (
                np.tanh(scaled_action, dtype=np.float32)
                if not self.normalized_action_space
                else scaled_action
            )

            # If we want to record log_pi during random exploration
            # We need to applied tanh transform
            # and calculate the squashed log prob correction as well.
            log_pi = (
                self.actor.evaluate_log_pi(
                    to_torch(state, self.device), to_torch(action, self.device)
                )
                if self.use_as_generator
                else None
            )

        # Returns to normal SAC exploration.
        else:
            action, log_pi = self.explore(
                obs_as_tensor(state, self.device), scale=False
            )
            scaled_action = (
                self.scale_action(action)
                if not self.normalized_action_space
                else action
            )

        # Ensures log_pi is not NaN.
        if log_pi is not None:
            assert not th.isnan(log_pi)

        # Interact with environment (Info might be useful for some special env).
        next_state, reward, done, info = env.step(scaled_action)

        # Done mask removes the time limit constrain of some env to keep makorvian.
        # Agent keeps alive should not be assigned to done by env's time limit.
        # See: https://github.com/sfujim/TD3/blob/master/main.py#L127
        # * Here we use an inverse convention in which DONE = 0 and NOT_DONE = 1.
        if (episode_timesteps + 1 == env._max_episode_steps) or not done:
            done_mask = DoneMask.NOT_DONE.value
        else:
            done_mask = DoneMask.DONE.value

        episode_timesteps += 1

        remaining_steps = 0
        if self.use_absorbing_state:
            if done and episode_timesteps < env._max_episode_steps:
                next_state = env.absorbing_state
                remaining_steps = env._max_episode_steps - episode_timesteps
                self.buffer.abs_counter += remaining_steps

        data = {
            "obs": asarray_shape2d(state),
            "acts": asarray_shape2d(action),
            "rews": asarray_shape2d(reward),
            "dones": asarray_shape2d(done_mask),
            "next_obs": asarray_shape2d(next_state),
        }
        if remaining_steps > 0:
            data["remaining_steps"] = asarray_shape2d(remaining_steps)

        # * No need to store log_pi for pure SAC, may be useful when use SAC as generator
        if self.use_as_generator:
            assert (
                "log_pis" in self.buffer.stored_keys()
            ), "Need initialize log_pis in buffer"
            data["log_pis"] = asarray_shape2d(log_pi)

        # Store transitions (s, a, r, s',d).
        # * ALLOW size larger than buffer capcity and truncate.
        self.buffer.store(data, truncate_ok=True)

        # Reset env if encounter done signal (not done mask!)
        if done:
            """
            Absorbing state corresponds to a state
            which a policy gets in after reaching a goal state and stays there forever.
            For most RL problems, we can just assign 0 to all reward after the goal.
            But for GAIL, we need to have an actual absorbing state.
            """
            if self.use_absorbing_state and (
                episode_timesteps < env._max_episode_steps
            ):
                # A fake action for the absorbing state (all zeros).
                absorbing_data = {
                    "obs": asarray_shape2d(env.absorbing_state),
                    "acts": asarray_shape2d(env.zero_action),
                    "rews": asarray_shape2d(0.0),
                    "dones": asarray_shape2d(DoneMask.ABSORBING.value),
                    "log_pis": asarray_shape2d(0.0),  # TODO: what to do with log_pi?
                    "next_obs": asarray_shape2d(env.absorbing_state),
                }
                self.buffer.store(absorbing_data, truncate_ok=True)
            episode_timesteps = 0
            next_state = env.reset()
        return next_state, episode_timesteps

    def update(self, log_this_batch: bool = False) -> Dict[str, Any]:
        """
        A general Roadmap
        for each gradient step do
            -Sample transition from replay buffer
            -Update the Q-function parameters
            -Update policy weights
            -Adjust temperature
            -Update target network weights every n gradient steps
        end for
        """
        train_logs = defaultdict(list)
        for gradient_step in range(self.num_gradient_steps):
            self.learning_steps += 1
            # Random uniform sampling a batch of transitions, B = {(s, a, r, s',d)}, from buffer.
            replay_data = self.buffer.sample(self.batch_size)
            info = self.update_algo(replay_data, gradient_step)
            if log_this_batch:
                for k in info.keys():
                    if info[k] is not None:
                        train_logs[k].append(info[k])
                    else:
                        continue
        return train_logs

    def update_algo(self, data: TensorDict, gradient_step: int) -> Dict[str, Any]:
        """
        Update the actor and critic and target network as well.
        :param data: a batch of randomly sampled transitions
        :return train_logs: dict of training logs
        """
        states, actions, rewards, dones, next_states = (
            data["obs"],
            data["acts"],
            data["rews"],
            data["dones"],
            data["next_obs"],
        )

        # Sample new actions and log probs from the current policy
        actions_new, log_pis_new = self.actor.sample(states)

        # Update and obtain the entropy coefficient.
        loss_alpha = self._update_alpha(log_pis_new)

        # First take one gradient descent step for Q1 and Q2
        loss_critic = self._update_critic(states, actions, rewards, dones, next_states)

        # Freeze Q-networks so we don't waste computational effort
        # computing gradients for them during the policy learning step.
        disable_gradient(self.critic)
        loss_actor = self._update_actor(states, actions_new, log_pis_new, dones)
        enable_gradient(self.critic)

        # Update target networks by polyak averaging.
        if gradient_step % self.target_update_interval == 0:
            self._update_target()
        # Return log changes(key used for logging name).
        return {
            "actor_loss": loss_actor,
            "critic_loss": loss_critic,
            "entropy_loss": loss_alpha,
            "entropy_coef": self.alpha,
            "pi_lr": self.lr_actor,
            "vf_lr": self.lr_critic,
            "ent_lr": self.lr_alpha,
        }

    def _update_alpha(self, log_pis_new: th.Tensor) -> th.Tensor:
        """
        Optimize entropy coefficient (alpha)
        L(alpha) = E_{at ∼ pi_t} [−alpha * log pi(a_t |s_t ) − alpha * H].
        ent_loss = E[-alpha * (log_pis) - alpha * target_ent]
                 = E [-alpha (log_pis + target_ent)]
                 = (-alpha * (log_pis + target_ent)).mean()
                 As log do preserve orders
                 = (-log_alpha * (target_ent + log_ent).mean()
        As discussed in https://github.com/rail-berkeley/softlearning/issues/37
        """
        self.optim_alpha.zero_grad(set_to_none=self.optim_set_to_none)
        # Important: detach the variable from the graph
        # So we don't change it with other losses
        # See https://github.com/rail-berkeley/softlearning/issues/60
        self.alpha = th.exp(self.log_alpha.detach())
        loss_alpha = -(
            self.log_alpha * (self.target_entropy + log_pis_new).detach()
        ).mean()
        loss_alpha.backward()
        self.optim_alpha.step()
        return loss_alpha.detach()

    def _update_critic(
        self,
        states: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        dones: th.Tensor,
        next_states: th.Tensor,
    ) -> th.Tensor:
        """Update Q-functions by one gradient step"""
        self.optim_critic.zero_grad(set_to_none=self.optim_set_to_none)

        with autocast(enabled=self.fp16):
            if self.use_absorbing_state:
                a_mask = th.maximum(th.zeros_like(dones), dones)
                # * Using action from the replay buffer
                curr_qs1, curr_qs2 = self.critic(states, actions)

                # Bellman backup for Q functions
                with th.no_grad():
                    # * Target actions come from *current* policy
                    next_actions, next_log_pis = self.actor.sample(next_states)
                    assert th.isnan(next_log_pis).sum() == 0
                    next_qs1, next_qs2 = self.critic_target(
                        next_states, next_actions * a_mask
                    )
                    # TODO: what to do with log_pis?
                    next_qs = th.min(next_qs1, next_qs2) - self.alpha * next_log_pis
                    # Target (TD error + entropy term):
                    # * Here we use an inverse convention in which DONE = 0 and NOT_DONE = 1.
                    target_qs = rewards + self.gamma * dones * next_qs
            else:
                # Get current Q-values estimation for each critic network
                # * Using action from the replay buffer
                curr_qs1, curr_qs2 = self.critic(states, actions)

                # Bellman backup for Q functions
                with th.no_grad():
                    # * Target actions come from *current* policy
                    # * Whereas by contrast, r and s' should come from the replay buffer.
                    next_actions, next_log_pis = self.actor.sample(next_states)
                    next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
                    # Double-Q trick and takes the minimum Q-value between two Q approximators.
                    next_qs = th.min(next_qs1, next_qs2) - self.alpha * next_log_pis

                    # Target (TD error + entropy term):
                    # r + gamma(1-d) * ( Q(s', a') - alpha log pi(a'|s') )
                    # * Here we use an inverse convention in which DONE = 0 and NOT_DONE = 1.
                    target_qs = rewards + self.gamma * dones * next_qs

            #  Mean-squared Bellman error (MSBE) loss
            loss_critic = F.mse_loss(curr_qs1, target_qs) + F.mse_loss(
                curr_qs2, target_qs
            )
        self.one_gradient_step(loss_critic, self.optim_critic, self.critic)
        return loss_critic.detach()

    def _update_actor(
        self,
        states: th.Tensor,
        actions_new: th.Tensor,
        log_pis_new: th.Tensor,
        dones: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Update policy by one step of gradient
        :param states: states from the replay buffer
        :param actions_new: sampled actions from the current policy
        :param log_pis_new: log_probs of the actions sampled from the current policy
        """

        self.optim_actor.zero_grad(set_to_none=self.optim_set_to_none)
        with autocast(enabled=self.fp16):
            qs1, qs2 = self.critic(states, actions_new)
            qs = th.min(qs1, qs2)
            if self.use_absorbing_state:
                # Don't update the actor for absorbing states.
                # And skip update if all states are absorbing.
                a_mask = 1.0 - th.maximum(th.zeros_like(dones), -dones)
                if a_mask.sum() < 1e-8:
                    return
                else:
                    loss_actor = (
                        self.alpha * log_pis_new - qs * a_mask
                    ).sum() / a_mask.sum()
            else:
                loss_actor = (self.alpha * log_pis_new - qs).mean()
        self.one_gradient_step(loss_actor, self.optim_actor, self.actor)
        return loss_actor.detach()

    def _update_target(self) -> None:
        """update the target network by polyak averaging."""
        # * here tau = (1 - polyak)
        soft_update(self.critic_target, self.critic, self.tau, self.one)

    def save_models(self, save_dir: str) -> None:
        """
        Save the model. (Only save actor to reduce workloads)
        """
        super().save_models(save_dir)
        th.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))

    @classmethod
    def load(
        cls,
        path: str,
        policy_kwargs: Dict[str, Any],
        env: Union[GymEnv, str, None] = None,
        state_space: Optional[GymSpace] = None,
        action_space: Optional[GymSpace] = None,
        device: Union[th.device, str] = "cpu",
        seed: int = 42,
        **kwargs,
    ) -> "SAC":
        """
        Load the model from a saved model directory.
        we only load actor.
        """
        super().load(env, state_space, action_space)
        if env is not None:
            if isinstance(env, str):
                import gym

                env = gym.make(env)
            state_space, action_space = env.observation_space, env.action_space

        sac_expert = cls(
            state_space,
            action_space,
            device,
            seed,
            policy_kwargs,
            init_buffer=False,
            init_models=False,
            expert_mode=True,
            **kwargs,
        )
        state_dict = th.load(path)
        sac_expert.actor.load_state_dict(state_dict)
        disable_gradient(sac_expert.actor)
        sac_expert.actor.eval()
        return sac_expert
