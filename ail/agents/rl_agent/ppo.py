from typing import Union, Optional, Tuple, Dict, Any
import os

import numpy as np
import torch as th
from torch.cuda.amp import autocast

from ail.agents.rl_agent.rl_core import OnPolicyAgent
from ail.common.math import normalize
from ail.common.pytorch_util import asarray_shape2d, obs_as_tensor, disable_gradient
from ail.common.type_alias import AlgoTags, DoneMask, GymEnv, GymSpace, TensorDict


class PPO(OnPolicyAgent):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    Paper: https://arxiv.org/abs/1707.06347

    :param state_space: state space.
    :param action_space: action space.
    :param device: PyTorch device to which the values will be converted.
    :param seed: random seed.
    :param batch_size: size of the batch (we assume batch_size == buffer_size).
    :param policy_kwargs: arguments to be passed to the policy on creation.
        e.g. : {
            pi: [64, 64],
            vf: [64, 64],
            activation: 'relu',
            lr_actor: 3e-4,
            lr_critic: 3e-4,
            critic_type="V",
            orthogonal_init: True,
            }
    :param epoch_ppo: Number of epoch when optimizing the surrogate loss.
    :param gamma: Discount factor.
    :param clip_eps: PPO clipping parameter.
    :param coef_ent: Entropy coefficient for the loss calculation.
    :param max_grad_norm: Maximum norm for the gradient clipping.
    :param fp16: Whether to use float16 mixed precision training.
    :optim_kwargs: arguments to be passed to the optimizer.
        eg. : {
            "optim_cls": adam,
            "optim_set_to_none": True, # which set grad to None instead of zero.
            }
    :param buffer_kwargs: Arguments to be passed to the buffer.
        eg. : {
            with_reward: True,
            extra_data: ["log_pis"]
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
        batch_size: int = 2_000,
        epoch_ppo: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_eps: float = 0.2,
        coef_ent: float = 0.01,
        max_grad_norm: Optional[float] = None,
        fp16: bool = False,
        optim_kwargs: Optional[dict] = None,
        buffer_kwargs: Optional[Dict[str, Any]] = None,
        init_buffer: bool = True,
        init_models: bool = True,
        expert_mode: bool = False,
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
            batch_size,  # * (Yifan) here assumes batch_size == buffer_size
            policy_kwargs,
            optim_kwargs,
            buffer_kwargs,
            init_buffer,
            init_models,
            expert_mode,
        )

        # learning rate scheduler.
        # TODO: add learning rate scheduler.
        # ? (Yifan) Is there one suitable for RL?

        """alpha_t = alpha_0 (1 - t/T)"""
        # schedule = lambda epoch: 1 - epoch/(self.param.evaluation['total_timesteps'] // self.batch_size)
        # self.scheduler_actor = optim.lr_scheduler.LambdaLR(self.optim_actor, schedule)
        # self.scheduler_critic = optim.lr_scheduler.LambdaLR(self.optim_critic, schedule)

        # Other algo params.
        self.learning_steps_ppo = 0
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.coef_ent = coef_ent

        self.tag = AlgoTags.PPO

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def is_update(self, step: int) -> bool:
        """Whether or not to update the agent"""
        return step % self.batch_size == 0

    def step(
        self,
        env: GymEnv,
        state: th.Tensor,
        episode_timesteps: th.Tensor,
        global_timesteps: Optional[int] = None,
        add_absorbing_state: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Intereact with environment and store the transition.

        :param env: gym environment
        :param state: orginal state return by the environment
        :param episode_timesteps: number of timesteps this episode
        :param total_timesteps: total number of timesteps to run in outer loop
        :return: next_state, episode length
        """
        episode_timesteps += 1

        # Sample actions from action distribution.
        # which is then wrapped by tanh transform to keep it in range [-1, 1].
        action, log_pi = self.explore(obs_as_tensor(state, self.device), scale=False)

        # Resacle actions to match original action space.
        scale_action = (
            self.scale_action(action) if not self.normalized_action_space else action
        )

        # Interact with environment (Info might be useful for some special env).
        next_state, reward, done, info = env.step(scale_action)

        # Done mask removes the time limit constrain of some env to keep makorvian.
        # Agent keeps alive should not be done by env's time limit.
        # See: https://github.com/sfujim/TD3/blob/master/main.py#L127
        # * Here we use an inverse convention in which DONE = 0 and NOT_DONE = 1
        # * to match absorbing state implementation in DAC paper.
        done_mask: float
        if (episode_timesteps == env._max_episode_steps) or not done:
            done_mask = DoneMask.NOT_DONE.value
        else:
            done_mask = DoneMask.DONE.value

        absorbing_cond = all(
            [add_absorbing_state, done, episode_timesteps < env._max_episode_steps]
        )
        if absorbing_cond:
            next_state = env.get_absorbing_state()

        data = {
            "obs": asarray_shape2d(state),
            "acts": asarray_shape2d(action),
            "rews": asarray_shape2d(reward),
            "dones": asarray_shape2d(done_mask),
            "log_pis": asarray_shape2d(log_pi),
            "next_obs": asarray_shape2d(next_state),
        }

        # Store transition.
        # * NOT ALLOW size larger than buffer capcity.
        self.buffer.store(data, truncate_ok=False)

        # Reset env if encounter done signal (not done mask!)
        if done:
            episode_timesteps = 0
            next_state = env.reset()
            # Add a absorbing state to buffer when done.
            if add_absorbing_state and (episode_timesteps < env._max_episode_steps):
                # A fake action for the absorbing state.
                zero_action = np.zeros(env.action_space.shape)
                absorbing_state = env.get_absorbing_state()
                absorbing_data = {
                    "obs": asarray_shape2d(absorbing_state),
                    "acts": asarray_shape2d(zero_action),
                    "rews": asarray_shape2d(0.0),
                    "dones": asarray_shape2d(DoneMask.ABSORBING.value),
                    "log_pis": asarray_shape2d(log_pi),  # TODO: what to do with log_pi?
                    "next_obs": asarray_shape2d(absorbing_state),
                }
                self.buffer.store(absorbing_data, truncate_ok=False)
        return next_state, episode_timesteps

    def update(self, log_this_batch: bool = False) -> Dict[str, Any]:
        """
        A general road map for updating the model.
        Obtain the training batch and perform update.
        :return train_logs: dict of training logs
        """
        self.learning_steps += 1
        rollout_data = self.buffer.get()

        # Clear buffer after getting entire buffer.
        self.buffer.reset()
        train_logs = self.update_algo(rollout_data, log_this_batch)
        return train_logs

    def update_algo(
        self, data: TensorDict, log_this_batch: bool = False
    ) -> Dict[str, Any]:
        """
        Update the actor and critic.
        :param data: a batch of randomly sampled transitions
        :return train_logs: dict of training logs
        """
        states, actions, rewards, dones, next_states, log_pis = (
            data["obs"],
            data["acts"],
            data["rews"],
            data["dones"],
            data["next_obs"],
            data["log_pis"],
        )
        with th.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            rewards, (1.0 - dones), values, next_values, self.gamma, self.gae_lambda
        )

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            loss_critic = self._update_critic(states, targets)
            loss_actor, pi_info = self._update_actor(states, actions, log_pis, gaes)

        if log_this_batch:
            # Return log changes(key used for logging name).
            return {
                "actor_loss": loss_actor,
                "critic_loss": loss_critic,
                "approx_kl": pi_info["kl"],
                "entropy": pi_info["ent"],
                "clip_fraction": pi_info["cf"],
                "pi_lr": self.lr_actor,
                "vf_lr": self.lr_critic,
                "learn_steps_ppo": self.learning_steps_ppo,
            }
        else:
            return {}

    def _update_critic(self, states: th.Tensor, targets: th.Tensor) -> th.Tensor:
        """
        Update critic. (value function approximation)
        :param states:
        :param targets: should be gae + v_pred
        return: critic loss
        """
        self.optim_critic.zero_grad(set_to_none=self.optim_set_to_none)
        with autocast(enabled=self.fp16):
            loss_critic = (self.critic(states) - targets).pow(2).mean()
        self.one_gradient_step(loss_critic, self.optim_critic, self.critic)
        return loss_critic.detach()

    def _update_actor(
        self,
        states: th.Tensor,
        actions: th.Tensor,
        log_pis_old: th.Tensor,
        gaes: th.Tensor,
    ) -> Tuple[th.Tensor, Dict[str, Any]]:
        """
        Update actor. (function for computing PPO policy loss)
        :param states:
        :param actions:
        :param log_pis_old:
        :param gaes: general advantage estimation
        : return: actor loss, policy_info
        """
        log_pis = self.actor.evaluate_log_pi(states, actions)

        # * (Yifan) Since we bounded the mean action with tanh(),
        # * there is no analytical form of entropy
        # Approximate entropy.
        approx_ent = -log_pis.mean()

        # ratio between old and new policy, should be one at the first iteration
        log_ratios = log_pis - log_pis_old
        ratios = (log_ratios).exp()

        # clipped surrogate loss
        loss_actor1 = ratios * gaes
        loss_actor2 = th.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes
        loss_actor = -th.min(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad(set_to_none=self.optim_set_to_none)
        with autocast(enabled=self.fp16):
            loss_actor_ent = loss_actor - self.coef_ent * approx_ent
        self.one_gradient_step(loss_actor_ent, self.optim_actor, self.actor)

        """
        Calculate approximate form of reverse KL Divergence for early stopping.
        See issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        and Schulman blog: https://joschu.net/blog/kl-approx.html
        KL(q||p): (r-1) - log(r), where r = p(x)/q(x)
        """
        # ! (Yifan) Deprecated :
        # ! Naive version: approx_kl = (log_pi_old - log_pi).mean().item()
        # ! This is an unbiased estimator, but it has large variance.
        # ! Since it can take on negative values.
        # ! as opposed to the actual KL Divergence measure
        # Useful extra info
        with th.no_grad():
            approx_kl = ((ratios - 1) - log_ratios).mean()
            clipped = ratios.gt(1 + self.clip_eps) | ratios.lt(1 - self.clip_eps)
            clip_frac = th.as_tensor(clipped, dtype=th.float32).mean()
            pi_info = {"kl": approx_kl, "ent": approx_ent.detach(), "cf": clip_frac}
        return loss_actor.detach(), pi_info

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
    ) -> "PPO":
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

        ppo_expert = cls(
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
        ppo_expert.actor.load_state_dict(state_dict)
        disable_gradient(ppo_expert.actor)
        ppo_expert.actor.eval()
        return ppo_expert


def calculate_gae(
    rewards: th.Tensor,
    dones: th.Tensor,
    values: th.Tensor,
    next_values: th.Tensor,
    gamma: float,
    lambd: float,
    normal: bool = True,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Compute the lambda-return (TD(lambda) estimate) and GAE(lambda) advantage.

    Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
    to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
    where R is the discounted reward with value bootstrap,
    set `lambd=1.0`.
    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo1/pposgd_simple.py#L66
    """
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1.0 - dones) - values
    # Initialize gae.
    gaes = th.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1.0 - dones[t]) * gaes[t + 1]

    targets = values + gaes

    if normal:
        return targets, normalize(gaes, gaes.mean(), gaes.std())
    else:
        return targets, gaes
