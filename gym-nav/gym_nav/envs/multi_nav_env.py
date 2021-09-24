from typing import List

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from matplotlib import pyplot as plt


class MultiNavEnv(gym.Env):
    """
    2D continuous box environment for navigation
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, size=1, seed=1337, multiplayer=2):

        # the box size is (2*size, 2*size)
        self.size = size
        self.max_steps = 100
        self.action_dim = 2
        self.pos_dim = 2
        self.vel_dim = 2
        self.obs_dim = 64 * 2
        self.dt = 1e-1

        self.act_range = (-1, 1)
        # Actions are 2D velocity
        self.action_space = spaces.Box(
            low=-self.size, high=self.size, shape=(self.action_dim,), dtype="float32"
        )

        # Observations are the position and velocity
        self.observation_space = spaces.Box(
            low=-self.size,
            high=self.size,
            shape=(self.pos_dim + self.vel_dim,),
            dtype="float32",
        )

        # Two player

        # pos
        self.demo_pos: np.ndarray
        self.agent_pos: np.ndarray
        # vel
        self.demo_vel = np.zeros(self.vel_dim)
        self.agent_vel = np.zeros(self.vel_dim)
        # steps
        self.demo_step_count: int
        self.agent_step_count: int

        self.demo_trajs: List[np.ndarray] = []
        self.agent_trajs: List[np.ndarray] = []

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

    def step(self, demo_action, agent_action):

        self.demo_step_count += 1
        self.agent_step_count += 1

        a_min, a_max = self.act_range
        demo_action = np.clip(demo_action, a_min, a_max)
        agent_action = np.clip(agent_action, a_min, a_max)

        self.update_states(demo_action, agent_action)

        # Observations are the position and velocity
        demo_obs = np.concatenate((self.demo_pos, self.demo_vel))
        demo_reward = self.eval_reward(self.demo_pos)
        self.demo_pos_list.append(self.demo_pos.copy())
        # agent
        agent_obs = np.concatenate((self.agent_pos, self.agent_vel))
        agent_reward = self.eval_reward(self.agent_pos)
        self.agent_pos_list.append(self.agent_pos.copy())

        demo_done, agent_done = False, False

        self.threshold = 3e-2
        demo_success = self.terminate_condition(self.demo_pos, self.threshold)
        agent_success = self.terminate_condition(self.agent_pos, self.threshold)
        demo_terminate_cond = self.demo_step_count >= self.max_steps
        agent_terminate_cond = self.agent_step_count >= self.max_steps
        if demo_success or demo_terminate_cond:
            demo_done = True

        if agent_success or agent_terminate_cond:
            agent_done = True

        return (
            (demo_obs, agent_obs),
            (demo_reward, agent_reward),
            (demo_done, agent_done),
            {},
        )

    def preprocess_obs(self, demo_pos, agent_pos):
        """
        Add the agent position to the reward map
        """
        obs = self.reward_map.copy()
        demo_idx0 = self.get_idx(demo_pos)
        agent_idx0 = self.get_idx(agent_pos)

        # traj
        if len(self.demo_pos_list) >= 3:
            demo_idx1 = self.get_idx(self.demo_pos_list[-2])
            demo_idx2 = self.get_idx(self.demo_pos_list[-3])
            obs[demo_idx1[0], demo_idx1[1]] = np.array([200, 0, 0])
            obs[demo_idx2[0], demo_idx2[1]] = np.array([100, 0, 0])
        if len(self.agent_pos_list) >= 3:
            agent_idx1 = self.get_idx(self.agent_pos_list[-2])
            agent_idx2 = self.get_idx(self.agent_pos_list[-3])
            obs[agent_idx1[0], agent_idx1[1]] = np.array([0, 200, 0])
            obs[agent_idx2[0], agent_idx2[1]] = np.array([0, 100, 0])

        # Set the demo to be red, agent to be green
        obs[demo_idx0[0], demo_idx0[1]] = np.array([255, 0, 0])
        obs[agent_idx0[0], agent_idx0[1]] = np.array([0, 255, 0])
        # if demo_idx[0] == agent_idx[0] and demo_idx[1] == agent_idx[1]:
        #     print("True")
        return obs

    def update_states(self, demo_action, agent_action):
        """
        Update the position and velocity with a simple dynamics model
        x = x + x' * dt, x' = x' + u * dt
        Force the position and velocity within bounds
        """
        assert not np.allclose(demo_action, agent_action)

        # print("demo:",demo_action)
        # print("agent",agent_action)
        self.demo_pos += self.demo_vel * self.dt
        self.demo_vel += demo_action * self.dt

        self.demo_pos[self.demo_pos > self.size] = self.size
        self.demo_pos[self.demo_pos < -self.size] = -self.size
        self.demo_vel[self.demo_vel > self.size] = self.size
        self.demo_vel[self.demo_vel < -self.size] = -self.size

        self.agent_pos += self.agent_vel * self.dt
        self.agent_vel += agent_action * self.dt
        self.agent_pos[self.agent_pos > self.size] = self.size
        self.agent_pos[self.agent_pos < -self.size] = -self.size
        self.agent_vel[self.agent_vel > self.size] = self.size
        self.agent_vel[self.agent_vel < -self.size] = -self.size

    def reset(self):
        """
        Randomly spawn a starting location
        """
        # Assuming two agents which has same initial position
        self.demo_pos = np.random.uniform(
            low=-self.size, high=self.size, size=self.pos_dim
        )
        self.agent_pos = self.demo_pos.copy()

        self.demo_vel = np.zeros(self.vel_dim)
        self.agent_vel = np.zeros(self.vel_dim)

        # Keep track of the position in the current episode
        self.demo_pos_list = [self.demo_pos.copy()]
        self.agent_pos_list = [self.agent_pos.copy()]

        # Step count since episode start
        self.demo_step_count = 0
        self.agent_step_count = 0

        #        # Observation is the reward map with the agent position
        #        obs = self.preprocess_obs(self.pos)
        # Observations are the position and velocity
        demo_obs = np.concatenate((self.demo_pos, self.demo_vel))
        agent_obs = np.concatenate((self.agent_pos, self.agent_vel))

        assert np.allclose(demo_obs, agent_obs)

        return demo_obs, agent_obs

    def render(self, mode="human"):
        """
        Visualize the trajectory on the reward map
        """

        if mode == "human" and not self.window:
            import gym_nav.window

            self.window = gym_nav.window.Window("gym_nav")
            self.window.show(block=False)

        img = self.preprocess_obs(self.demo_pos, self.agent_pos)

        if mode == "human":
            self.window.show_img(img)
            self.window.set_caption(self.mission + "Demo: red, Agent: green")
            return img

        elif mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            self.window.close()
        return

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

        # Increase obs_dim for higher resolution
        num = self.obs_dim
        x = np.linspace(-self.size, self.size, num=num)
        y = np.linspace(-self.size, self.size, num=num)

        X, Y = np.meshgrid(x, y)
        Z = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                Z[i, j] = self.eval_gaussian(np.array([X[i, j], Y[i, j]]))
        # rescale values to [0, 255]
        self.reward_min, self.reward_max = np.min(Z), np.max(Z)
        Z = ((Z - self.reward_min) / (self.reward_max - self.reward_min) * 255).astype(
            np.uint8
        )
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
        # shape reward to [-1, 0] to assist learning
        reward_std = (reward - self.reward_min) / (self.reward_max - self.reward_min)
        min_val, max_val = (-20, 0)
        scale_reward = reward_std * (max_val - min_val) + min_val
        return scale_reward

    @staticmethod
    def terminate_condition(pos, threshold):
        x, y = pos
        return abs(x) <= threshold and abs(y) <= threshold

    def get_idx(self, pos):
        idx = ((pos + self.size) / (2 * self.size) * (self.obs_dim - 1)).astype(int)
        return idx
