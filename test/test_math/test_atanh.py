import math
import numpy as np
import torch as th
from ail.common.math import atanh

x = th.tanh(th.tensor([2e6]))
y = th.tanh(th.tensor([-2e6]))


def test_atanh_output():

    res_x = atanh(x).item()
    res_y = atanh(y).item()
    assert not math.isnan(res_x)
    assert not math.isinf(res_x)
    assert not math.isnan(res_y)
    assert not math.isinf(res_y)


def test_atanh_output_accuracy():
    w = th.ones(1, dtype=th.float32)
    a = -0.95 * w
    b = -0.5 * w
    c = -0.1 * w
    d = 0 * w
    x = 0.1 * w
    y = 0.5 * w
    z = 0.95 * w
    assert np.allclose(atanh(a).numpy(), th.atanh(a).numpy())
    assert np.allclose(atanh(b).numpy(), th.atanh(b).numpy())
    assert np.allclose(atanh(c).numpy(), th.atanh(c).numpy())
    assert np.allclose(atanh(d).numpy(), th.atanh(d).numpy())
    assert np.allclose(atanh(x).numpy(), th.atanh(x).numpy())
    assert np.allclose(atanh(x).numpy(), th.atanh(x).numpy())
    assert np.allclose(atanh(y).numpy(), th.atanh(y).numpy())
    assert np.allclose(atanh(z).numpy(), th.atanh(z).numpy())
