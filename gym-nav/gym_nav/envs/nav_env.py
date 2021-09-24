from functools import lru_cache
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from matplotlib import pyplot as plt


class NavEnv(gym.Env):
    """
    2D continuous box environment for navigation
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, size=1, seed=1337):

        # max episode length
        self.max_steps: int = 100
        self._max_episode_steps: int = self.max_steps

        # resolution
        self.resolution: int = 64 * 2

        """
        A simple dynamics model
        position: x = x + x' * dt,
        velocity: x' = x' + u * dt
        """
        self.pos_dim: int = 2
        self.vel_dim: int = 2

        self.pos: np.ndarray
        self.vel: np.ndarray
        self.dt: float = 1e-1

        # Keep track of the position in the current episode
        self.pos_list: list

        # Step count since episode start
        self.step_count: int

        # obs and act dim
        self.obs_dim: int = self.pos_dim + self.vel_dim
        self.action_dim: int = 2

        # the box size is (2*size, 2*size)
        self.size: int = size

        # Observations are the position and velocity
        self.observation_space = spaces.Box(
            low=-self.size,
            high=self.size,
            shape=(self.pos_dim + self.vel_dim,),
            dtype="float32",
        )

        # Actions are 2D velocity
        self.action_space = spaces.Box(
            low=-self.size, high=self.size, shape=(self.action_dim,), dtype="float32"
        )
        self.act_range = (-1, 1)

        # Reward map
        self.reward_map = None
        self.reward_min = None
        self.reward_max = None
        self.reward_range = (-20, 0)

        # Window to use for human rendering mode
        self.window = None

        # Mission of the task
        self.mission: str = "Navigate to the origin"

        # Initialize seed
        self.seed(seed=seed)

        # Initialize reward map
        self.init_reward()

        # Initialize the state
        self.reset()

    ####################################
    ####################################

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    ####################################
    ####################################

    def step(self, action):
        """env.step"""
        self.step_count += 1

        # clip action
        a_min, a_max = self.act_range
        action = np.clip(action, a_min, a_max)
        self.update_states(action)

        # Observations are the position and velocity
        obs = np.concatenate((self.pos, self.vel))
        reward = self.eval_reward(self.pos)
        self.pos_list.append(self.pos.copy())

        done = False
        threshold = 3e-2

        # end the rollout if (rollout can end due to done, or due to max_path_length)
        done_cond = self.terminate_condition(self.pos, threshold)
        terminate_cond = self.step_count >= self.max_steps
        if done_cond or terminate_cond:
            done = True

        return obs, reward, done, {}

    def preprocess_obs(self, agent_pos):
        """
        Add the agent position to the reward map
        """
        obs = self.reward_map.copy()
        agent_idx = (
            (agent_pos + self.size) / (2 * self.size) * (self.resolution - 1)
        ).astype(int)

        # display 2 recent pos
        if len(self.pos_list) >= 3:
            agent_idx1 = self.get_idx(self.pos_list[-2])
            agent_idx2 = self.get_idx(self.pos_list[-3])
            obs[agent_idx1[0], agent_idx1[1]] = np.array([200, 0, 0], dtype=np.uint8)
            obs[agent_idx2[0], agent_idx2[1]] = np.array([100, 0, 0], dtype=np.uint8)

        # Set the agent to be red
        obs[agent_idx[0], agent_idx[1]] = np.array([255, 0, 0], dtype=np.uint8)
        return obs

    def update_states(self, action):
        """
        Update the position and velocity with a simple dynamics model
        x = x + x' * dt, x' = x' + u * dt
        Force the position and velocity within bounds
        """
        self.pos += self.vel * self.dt
        self.vel += action * self.dt
        self.pos[self.pos > self.size] = self.size
        self.pos[self.pos < -self.size] = -self.size
        self.vel[self.vel > self.size] = self.size
        self.vel[self.vel < -self.size] = -self.size

    def reset(self):
        """
        Randomly spawn a starting location
        """
        self.pos = np.random.uniform(low=-self.size, high=self.size, size=self.pos_dim)

        self.vel = np.zeros(self.vel_dim)

        # Keep track of the position in the current episode
        self.pos_list = [self.pos.copy()]

        # Step count since episode start
        self.step_count: int = 0

        obs = np.concatenate((self.pos, self.vel))

        return obs

    def render(self, mode="human"):
        """
        Visualize the trajectory on the reward map
        """

        if mode == "human" and not self.window:
            import gym_nav.window

            self.window = gym_nav.window.Window("gym_nav")
            self.window.show(block=False)

        img = self.preprocess_obs(self.pos)

        if mode == "human":
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        elif mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            self.window.close()
        return

    @lru_cache(maxsize=5)
    def init_reward(self):
        """
        The reward is a mixture of Gaussians functions
        Highest at the origin and lowest at the four corners
        """
        mu_factor = 1 / 3
        std_factor = 1 / 8
        A_factor = -1
        self.mixtures = {
            "mu": [
                np.zeros(self.pos_dim),
                np.array([self.size * mu_factor, self.size * mu_factor]),
                np.array([self.size * mu_factor, -self.size * mu_factor]),
                np.array([-self.size * mu_factor, self.size * mu_factor]),
                np.array([-self.size * mu_factor, -self.size * mu_factor]),
            ],
            "std": [
                self.size * 1.2,
                self.size * std_factor,
                self.size * std_factor,
                self.size * std_factor,
                self.size * std_factor,
            ],
            "A": [2.5, A_factor, A_factor, A_factor, A_factor],
        }

        # Increase self.resolution for higher resolution
        num = self.resolution
        x = np.linspace(-self.size, self.size, num=num)
        y = np.linspace(-self.size, self.size, num=num)
        # TODO: vectorized this computation
        X, Y = np.meshgrid(x, y)
        Z = self.eval_loop(X, Y, num)

        # rescale values to [0, 255]
        self.reward_min, self.reward_max = np.min(Z), np.max(Z)
        Z = ((Z - self.reward_min) / (self.reward_max - self.reward_min) * 255).astype(
            np.uint8
        )
        self.Z = Z

        # convert grayscale to rgb by stacking three channels
        self.reward_map = np.stack((Z, Z, Z), axis=-1)

    def eval_gaussian(self, x):
        """
        Evaluate the value of mixture of gaussian functions at location x
        """
        ans = 0
        for mu, std, A in zip(
            self.mixtures["mu"], self.mixtures["std"], self.mixtures["A"]
        ):
            ans += A * np.exp(-(x - mu).T @ (x - mu) / (2 * std ** 2))
        return ans

    def eval_reward(self, x):
        """
        Evaluate the reward at the current position x
        """
        reward = self.eval_gaussian(x)

        # shape reward to [-1, 1] to assist learning
        # reward = (reward - self.reward_min) / (self.reward_max - self.reward_min) * 2 - 1
        reward_std = (reward - self.reward_min) / (self.reward_max - self.reward_min)
        min_val, max_val = self.reward_range
        scale_reward = reward_std * (max_val - min_val) + min_val
        return scale_reward

    @staticmethod
    def terminate_condition(pos, threshold):
        x, y = pos
        return abs(x) <= threshold and abs(y) <= threshold

    def get_idx(self, pos):
        idx = ((pos + self.size) / (2 * self.size) * (self.resolution - 1)).astype(int)
        return idx

    def eval_loop(self, X, Y, num):
        Z = np.empty((num, num))
        for i in range(num):
            for j in range(num):
                Z[i, j] = self.eval_gaussian(np.array([X[i, j], Y[i, j]]))
        return Z
