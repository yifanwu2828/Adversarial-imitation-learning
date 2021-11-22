import sys

import numpy as np
import gym
from stable_baselines3 import SAC
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper

from icecream import ic

class DoneAfterSuccess(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
    
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        if info["is_success"] == 1:
            done = True
        return obs, reward, done, info 

if __name__ == '__main__':
    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    
    
    env_id = "FetchPush-v1"
    
    env = gym.make(env_id)
    env = TimeFeatureWrapper(gym.make(env_id))
    env = DoneAfterSuccess(env)
    
    ic(env.observation_space)
    
    model_path = f"./{env_id}"
    
    model = TQC.load(model_path, env=env, custom_objects=custom_objects,)
    
    
    for i in range(10):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        ep_len = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            episode_reward += reward
            ep_len += 1
            env.render("human")
