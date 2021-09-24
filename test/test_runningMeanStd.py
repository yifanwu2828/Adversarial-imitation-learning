import numpy as np
from icecream import ic
from ail.common.running_stats import RunningStats, RunningMeanStd


def test_runningmeanstd():
    """Test RunningMeanStd object"""
    for (x_1, x_2, x_3) in [
        # (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2)),
    ]:
        rms = RunningMeanStd(epsilon=1e-3, shape=x_1.shape[1:])
        rs = RunningStats(shape=x_1.shape[1:])
        x_cat = np.concatenate([x_1, x_2, x_3], axis=0)
        
        moments_1 = [x_cat.mean(axis=0), x_cat.var(axis=0)]
        rms.update(x_1)
        rms.update(x_2)
        rms.update(x_3)
        for i in x_cat:
            print(i)
            rs.push(i)
        
        
        moments_2 = [rms.mean, rms.var]
        ic(moments_1)
        ic(moments_2)
        moments_3 = [rs.mean, rs.var]
        ic(moments_3)
        # assert np.allclose(moments_1, moments_2)

if __name__ == '__main__':
    test_runningmeanstd()

