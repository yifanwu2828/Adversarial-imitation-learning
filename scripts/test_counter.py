import numpy as np
from icecream import ic

if __name__ == '__main__':
    length = 12
    size = 6
    a = np.ones(size) * -1
    counter = 0
    for i in range(size):
        if i < size-1:
            a[i] = i
        else:
            remain = length - (i+1)
            counter += remain
    a_mask = np.where(a==-1)[0]
    idx = a_mask[0]
    ic(a)
    ic(a_mask)
    ic(idx)
    ic(counter)
    n =len(a) + counter
    ic(n)
    p = np.empty_like(a)
    p[: idx] = 1/n
    p[idx]= 1 - np.sum(p[:idx])
    ic(p)
    assert np.sum(p) == 1
    samp = np.random.choice(a, size=size, replace=True, p=p)
    ic(samp)