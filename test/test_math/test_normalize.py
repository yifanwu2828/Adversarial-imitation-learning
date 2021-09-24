import numpy as np
from ail.common.math import normalize, unnormalize


x1 = np.arange(1, 100, 5).reshape(-1, 1)


def test_normalize():
    assert abs(np.mean(normalize(x1, x1.mean(), x1.std()))) < 1e-6


def test_unnormalize():
    norm_x = normalize(x1, x1.mean(), x1.std())
    unnorm_x = unnormalize(norm_x, x1.mean(), x1.std())
    assert np.allclose(unnorm_x, x1)
