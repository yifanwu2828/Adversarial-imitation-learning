import math
from copy import deepcopy
from typing import Dict, Union

import gym
import numpy as np


from ail.common.running_stats import RunningMeanStd
from ail.common.type_alias import GymEnv, GymStepReturn


class VecNormalize(gym.Wrapper):
    """
    A moving average, normalizing wrapper for gym environment.

    :param venv: the vectorized environment to wrap
    :param training: Whether to update or not the moving average
    :param norm_obs: Whether to normalize observation or not (default: True)
    :param norm_reward: Whether to normalize rewards or not (default: True)
    :param clip_obs: Max absolute value for observation
    :param clip_reward: Max value absolute for discounted reward
    :param gamma: discount factor
    :param epsilon: To avoid division by zero
    """

    __slots__ = [
        "ret_rms",
        "ret",
        "gamma",
        "epsilon",
        "training",
        "norm_obs",
        "norm_reward",
        "clip_obs",
        "clip_reward",
    ]

    def __init__(
        self,
        env: GymEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        assert isinstance(
            env.observation_space, (gym.spaces.Box, gym.spaces.Dict)
        ), "VecNormalize only support `gym.spaces.Box` and `gym.spaces.Dict` observation spaces"

        params = {clip_obs, clip_reward, gamma, epsilon}
        for param in params:
            assert isinstance(param, float)

        super().__init__(env)

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_keys = set(self.observation_space.spaces.keys())
            self.obs_spaces = self.observation_space.spaces
            self.obs_rms = {
                key: RunningMeanStd(shape=space.shape)
                for key, space in self.obs_spaces.items()
            }
        else:
            self.obs_keys, self.obs_spaces = None, None
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)

        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_rew = clip_reward

        # Returns: discounted rewards
        # * Currently only support one env
        self.num_envs = 1
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = np.array([])
        self.old_reward = np.array([])

    def step(self, action: np.ndarray) -> GymStepReturn:
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)
        where ``dones`` is a boolean vector indicating whether each element is new.
        """
        obs, rewards, dones, infos = self.env.step(action)
        self.old_obs = obs
        self.old_reward = rewards

        if self.training:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.obs_rms.update(obs)

        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        # # Normalize the terminal observations
        # for idx, done in enumerate(dones):
        #     if not done:
        #         continue
        #     if "terminal_observation" in infos[idx]:
        #         infos[idx]["terminal_observation"] = self.normalize_obs(infos[idx]["terminal_observation"])
        return obs, rewards, dones, infos

    def _update_reward(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.ret = self.ret * self.gamma + reward
        self.ret_rms.update(self.ret)

    def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: normalized observation
        """
        norm_obs = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon)
        if not math.isinf(self.clip_obs):
            np.clip(norm_obs, -self.clip_obs, self.clip_obs, out=norm_obs)
        return norm_obs

    def _unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to unnormalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: unnormalized observation
        """
        return (obs * np.sqrt(obs_rms.var + self.epsilon)) + obs_rms.mean

    def normalize_obs(
        self, obs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # Avoid modifying by reference the original object
        obs_ = deepcopy(obs)
        if self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    obs_[key] = self._normalize_obs(obs[key], self.obs_rms[key]).astype(
                        np.float32
                    )
            else:
                obs_ = self._normalize_obs(obs, self.obs_rms).astype(np.float32)
        return obs_

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        `"Incorrect"` Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.

        Incorrect in the sense that we
        1. update return
        2. divide reward by std(return) *without* subtracting and adding back mean
        See: https://openreview.net/attachment?id=r1etN1rtPB&name=original_pdf
        """
        if self.norm_reward:

            if math.isinf(self.clip_rew):
                norm_rew = reward / np.sqrt(self.ret_rms.var + self.epsilon)
            else:
                norm_rew = np.clip(
                    reward / np.sqrt(self.ret_rms.var + self.epsilon),
                    -self.clip_rew,
                    self.clip_rew,
                )
        else:
            norm_rew = reward
        return norm_rew

    def unnormalize_obs(
        self, obs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        # Avoid modifying by reference the original object
        obs_ = deepcopy(obs)
        if self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    obs_[key] = self._unnormalize_obs(obs[key], self.obs_rms[key])
            else:
                obs_ = self._unnormalize_obs(obs, self.obs_rms)
        return obs_

    def unnormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.norm_reward:
            return reward * np.sqrt(self.ret_rms.var + self.epsilon)
        return reward

    def get_original_obs(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns an unnormalized version of the observations from the most recent
        step or reset.
        """
        return deepcopy(self.old_obs)

    def get_original_reward(self) -> np.ndarray:
        """
        Returns an unnormalized version of the rewards from the most recent step.
        """
        return self.old_reward.copy()

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments
        :return: first observation of the episode
        """
        obs = self.env.reset()
        self.old_obs = obs
        self.ret = np.zeros(self.num_envs)
        if self.training:
            self._update_reward(self.ret)
        return self.normalize_obs(obs)

    @property
    def _max_episode_steps(self) -> int:
        return self.env._max_episode_steps
