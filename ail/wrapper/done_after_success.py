

import gym

class DoneAfterSuccess(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
    
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        if info["is_success"] == 1:
            done = True
        return obs, reward, done, info        

if __name__ == '__main__':
    env = gym.make('FetchPush-v1')
    
    env = DoneAfterSuccess(env)
    obs = env.reset()
    print(obs)
    
    env.step(env.action_space.sample())
    