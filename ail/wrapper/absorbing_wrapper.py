import gym
import numpy as np


# Take from https://github.com/google-research/google-research/blob/21545f4ae92de3d512a9e52fa668a1988ea802c0/dac/lfd_envs.py#L28
class AbsorbingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            shape=(obs_space.shape[0] + 1,),
            low=obs_space.low[0],
            high=obs_space.high[0],
        )
        self._absorbing_state = self.get_absorbing_state()
        self._zero_action = self.get_fake_action()

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self.get_non_absorbing_state(observation)

    def get_non_absorbing_state(self, obs: np.ndarray) -> np.ndarray:
        """
        Converts an original state of the environment into a non-absorbing state.

        :param obs: a numpy array that corresponds to a state of unwrapped environment.
        :returns: a numpy array corresponding to a non-absorbing state obtained from input.
        """
        return np.concatenate([obs, [0]], -1)

    def get_absorbing_state(self) -> np.ndarray:
        """
        Returns an absorbing state that corresponds to the environment.
        :returns: a numpy array that corresponds to an absorbing state.
        """
        obs = np.zeros(self.observation_space.shape)
        obs[-1] = 1
        return obs

    def get_fake_action(self) -> np.ndarray:
        return np.zeros(self.env.action_space.shape)

    @property
    def zero_action(self) -> np.ndarray:
        return self._zero_action

    @property
    def absorbing_state(self) -> np.ndarray:
        return self._absorbing_state

    @property
    def _max_episode_steps(self) -> int:
        return self.env._max_episode_steps  # pylint: disable=protected-access
