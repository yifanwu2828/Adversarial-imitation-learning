import numpy as np

from icecream import ic

if __name__ == '__main__':
    # DONE=0, NOT_DONE = 1, ABSORBING = -1
    dones = np.array([1, 0, -1, 1, 1, 0, -1], dtype= np.float32).reshape(-1,1)
    remaining_steps = np.array([0, 3,  0, 0, 0, 6, 0], dtype= np.float32).reshape(-1,1)
    # remaining_step = max_ep_step - current step record before absorbing step which is at the step when done = True
    
    
    abs_idx = np.where(dones == -1)[0]
    non_abs_idx = np.where(dones != -1)[0]
    
    ic(abs_idx)
    ic(non_abs_idx)
    
    rng = np.random.default_rng(42)
    
    ic(remaining_steps[abs_idx-1])
    ic(dones)
    ic(abs_idx)
    
    num_true = dones.shape[0] - abs_idx.shape[0]
    num_total_abs = int(remaining_steps.sum())
    num_abs_in_buffer = abs_idx.shape[0]
    n_samples = 5
    
    total_num = num_true + num_total_abs
    ic(num_true)
    ic(num_total_abs)
    ic(total_num)
    
    num_arr = np.arange(dones.shape[0])
    ic(num_arr)
    
    p = np.empty_like(num_arr, dtype=np.float32)
    p[non_abs_idx] = 1/total_num
    ic(p)
    ic(p)
    ic(p.sum())
    
    ind = rng.choice(num_arr, size=n_samples, replace=True, p=p, axis=0, shuffle=True)
    ic(ind)