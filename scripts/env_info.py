import numpy as np
import gym
import gym_nav


if __name__ == '__main__':
    nav = gym.make('NavEnv-v0')
    inverted_pendulum = gym.make('InvertedPendulum-v2') # survival
    hopper = gym.make('Hopper-v3') # survival
    halfCheetah = gym.make('HalfCheetah-v2') 
    
    for env in [nav, inverted_pendulum, hopper, halfCheetah]:
        print(f"\nEnv: {env}")
        print(f"obs space: {env.observation_space}")
        print(f"act space: {env.action_space}")
        print(f"max ep_len: {env._max_episode_steps}")
    
    # obs = inverted_pendulum.reset()
    # for i in range(1_000):
    #     obs, rew, done, _ = inverted_pendulum.step(inverted_pendulum.action_space.sample())
    #     inverted_pendulum.render()
        
    #     if done:
    #         obs = inverted_pendulum.reset()