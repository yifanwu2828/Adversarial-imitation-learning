import gym

import numpy as np
from ail.common.type_alias import GymEnv, GymObs, GymStepReturn

class TimeAwareObsWrapper(gym.Wrapper):
    """
    Add remaining time to observation. Normalize it to [-1, 1]
    See https://arxiv.org/pdf/1712.00378.pdf
    :param env: gym environment.
    :param max_steps: Maximum number of stpes in an episode.
    """
    def __init__(self, env: GymEnv, max_steps: int = 1_000):
        obs_space = env.observation_space
        assert len(obs_space.shape) == 1, "Only 1D observation space is supported."
        
        low, high = obs_space.low, obs_space.high
        # ? In original paper it should normalize observation space to [-1, 1]
        # ? But it make no sense if the time is negative.
        low, high = np.concatenate([low, [0]]), np.concatenate([high, [1]])
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        super().__init__(env)
        
        # obtain max episode length from environment
        try:
            self._max_steps = env._max_episode_steps # pylint: disable=protected-access
            print(f"max episode length from the environment: {self._max_steps}")
            
        except AttributeError:
            self._max_steps = None
            
        # if this is not specified, we'll use a user-specified default value
        # Should make this consistent with the timelimit wrapper
        if self._max_steps is None:
            self._max_steps = max_steps
            print(f"Can not infer max episode length from the environment, use-defined values: {self._max_steps}")
            
        
        self._current_step = 0
    
    @property
    def _max_episode_steps(self) -> int:
        return self._max_steps
    
    def reset(self) -> GymObs:
        self._current_step = 0
        return self._get_obs(self.env.reset())
    
    def step(self, action: int) -> GymStepReturn:
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs: GymObs) -> GymObs:
        """
        Appedn remaining time to observation and normalize it it to [-1, 1]
        :param obs: observation.
        :return: observation with remaining time.
        """
        # remaining_time = (self._max_steps - self._current_step) 
        # normalized_remaining_time = remaining_time / self._max_steps
        normalized_remaining_time = 1.0 - (self._current_step / self._max_steps)
        
        return np.append(obs, normalized_remaining_time)