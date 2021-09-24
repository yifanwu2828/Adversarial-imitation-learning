import gym
import numpy as np
from icecream import ic

from ail.common.running_stats import RunningStats


def test_running_stat_fake_data():
    for shp in [(3, 4)]:
        li = []
        rs = RunningStats(shape=shp)
        for _ in range(5):
            val = np.random.randn(*shp)
            rs.push(val)
            li.append(val)
            m = np.mean(li, axis=0)
            assert np.allclose(rs.mean, m)
            v = np.square(m) if (len(li) == 1) else np.var(li, ddof=1, axis=0)
            assert np.allclose(rs.var, v)


def test_running_stats_gym():
    import gym

    env = gym.make("Pendulum-v0")
    obs = env.reset()
    # obs = obs.astype(np.float32)
    # ic(obs.shape, obs.dtype)

    rs = RunningStats(obs.shape)
    lst = []
    for i in range(100):
        act = env.action_space.sample()
        obs, *_ = env.step(act)
        lst.append(obs)
        rs.push(obs)
        tmp = np.asarray(lst)
        m = np.mean(tmp, axis=0)
        v = np.square(m) if (len(tmp) == 1) else np.var(tmp, ddof=1, axis=0)
        if i %10 == 0 and i >50:
            ic(tmp.shape)
            ic(m, v)
            ic(rs.mean, rs.var, "\n")
        # ic(rs.mean.astype(np.float32), rs.var.astype(np.float32))
        # ic(rs.mean.dtype, rs.var.dtype)

        np.testing.assert_allclose(rs.mean, m)
        np.testing.assert_allclose(rs.var, v)
    print("Pass")
        