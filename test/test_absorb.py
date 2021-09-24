import torch as th
import numpy as np
from icecream import ic


def make_absorbing_states(obs: np.ndarray, dones: np.ndarray):
        combined_states = np.hstack([obs, dones])
        absorbing_states = np.zeros(combined_states.shape[1]).reshape(1, -1)
        absorbing_states[:, -1] = 1.0
        is_done = np.all(combined_states, axis=-1, keepdims=True)
        absorbing_obs = np.where(is_done, absorbing_states, combined_states)
        return absorbing_obs

if __name__ == "__main__":
    a = th.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=th.float32)
    a = a.reshape(5,2)
    a = a.numpy()
    
    d = th.tensor([0,0,1,0,1], dtype=th.float32).reshape(5,-1)
    d = d.numpy()
    ic(a)
        
    # ans=th.where(cond, zeros, b)
    # ic(ans)
    ans=make_absorbing_states(a,d)
    ic(ans)