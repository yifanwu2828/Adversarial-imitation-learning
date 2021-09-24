import numpy as np
import torch as th
from ail.common.math import squash_logprob_correction

w = th.ones(1)
x = th.tanh(1 * w)
y = th.tanh(-1 * w)
z = th.tanh(0 * w)


def test_squash_logprob_correction_pos_one():

    actions = x
    out = squash_logprob_correction(actions).numpy()
    # log(1 - tanh(x)^2)
    original = th.log(1 - actions.pow(2)).numpy()
    assert np.allclose(out, original)


def test_squash_logprob_correction_neg_one():

    actions = y
    out = squash_logprob_correction(actions).numpy()
    # log(1 - tanh(x)^2)
    original = th.log(1 - actions.pow(2)).numpy()
    assert np.allclose(out, original)


def test_squash_logprob_correction_neg_zero():

    actions = z
    out = squash_logprob_correction(actions).numpy()
    # log(1 - tanh(x)^2)
    original = th.log(1 - actions.pow(2)).numpy()
    assert np.allclose(out, original)
