import torch as th

from icecream import ic
from ail.common.math import pure_discount_cumsum

if __name__ == "__main__":
    gamma = 0.99
    rew = th.tensor([1, 2, 3, 4, 5, 6]).float()
    dones = th.tensor([0, 1, -1, 1, 0, 0]).float()
    steps = th.tensor([3, 0, 0, 0, 2, 5]).float().long()
    idx = th.where(dones == 0)[0]

    ic(idx)
    ic(steps[idx])
    ic(rew[idx])

    ra = 10

    orig_rew = rew.clone()
    lst = []
    for i in idx:
        # t' = t - T
        t_prime = steps[i]
        ic(t_prime)

        power_idx = th.arange(1, t_prime + 1)
        r = th.ones_like(power_idx) * ra
        rT = rew[i]

        # discount gammas
        discounts = th.pow(gamma, power_idx)
        print("\n")

        # discounts * rewards from t to T
        discounted_rewards = discounts * r
        sum_discounted_rewards = th.sum(discounted_rewards)
        sum_discounted_return = sum_discounted_rewards + rT

        print("\n")

        x = [rT]
        x.extend([ra] * t_prime)
        ans = pure_discount_cumsum(x, gamma)
        ic(ans[0])

        ic(sum_discounted_return)
        error = abs(ans[0] - sum_discounted_return.item())
        assert error <= 1e-4, f"error is {error}"

        # rew[i] = sum_discounted_return
        lst.append(sum_discounted_return)

    ic(orig_rew)

    rew[idx] = th.tensor(lst).float()
    ic(rew)
    
    
