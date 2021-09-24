import dataclasses
import random
import uuid
from datetime import datetime
from time import sleep
from itertools import zip_longest
from collections import OrderedDict
from typing import Tuple, Dict, Any, Iterable, Union, Optional

import numpy as np
import torch as th
from torch.distributions import Bernoulli


def make_unique_timestamp() -> str:
    """Timestamp, with random uuid added to avoid collisions."""
    ISO_TIMESTAMP = "%Y%m%d_%H%M_%S"
    timestamp = datetime.now().strftime(ISO_TIMESTAMP)
    random_uuid = uuid.uuid4().hex[:3]
    return f"{timestamp}_{random_uuid}"


def set_random_seed(seed: int) -> None:
    """Set random seed to both numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def countdown(t_sec: int) -> None:
    """Countdown t seconds."""
    while t_sec:
        mins, secs = divmod(t_sec, 60)
        time_format = f"{mins: 02d}:{secs: 02d}"
        print(time_format, end="\r")
        sleep(1)
        t_sec -= 1
    print("Done!!")


def get_stats(x: np.ndarray) -> Tuple[np.ndarray, ...]:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return x.mean(), x.std(), x.min(), x.max()  # noqa


def combined_shape(length: int, shape: Optional[Tuple[int, ...]] = None):
    if shape is None:
        return (length,)  # noqa
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def dataclass_quick_asdict(dataclass_instance) -> Dict[str, Any]:
    """
    Extract dataclass to items using `dataclasses.fields` + dict comprehension.
    This is a quick alternative to `dataclasses.asdict`, which expensively and
    undocumentedly deep-copies every numpy array value.
    See https://stackoverflow.com/a/52229565/1091722.
    """
    obj = dataclass_instance
    d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return d


def zip_strict(*iterables: Iterable) -> Iterable:
    """
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.
    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    # ! Slow
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


# From stable baselines
def explained_variance(
    y_pred: Union[np.ndarray, th.Tensor], y_true: Union[np.ndarray, th.Tensor]
) -> Union[np.ndarray, th.Tensor]:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = y_true.var()
    return float("NaN") if var_y == 0 else 1 - (y_true - y_pred).var() / var_y


# Borrow and modify from Imitation Learning
# https://github.com/HumanCompatibleAI/imitation/blob/4008bfee6bb4ad0b1ae18ba5e45d99c1a397d7e5/src/imitation/rewards/common.py#L89-L154
def compute_disc_stats(
    disc_logits: th.Tensor,
    labels: th.Tensor,
    disc_loss: th.Tensor,
) -> Dict[str, float]:
    """
    Train statistics for GAIL/AIRL discriminator, or other binary classifiers.
    :param disc_logits: discriminator logits where expert is 1 and generated is 0
    :param labels: integer labels describing whether logit was for an
            expert (1) or generator (0) sample.
    :param disc_loss: discriminator loss.
    :returns stats: dictionary mapping statistic names for float values.
    """
    with th.no_grad():
        bin_is_exp_pred = disc_logits > 0
        bin_is_exp_true = labels > 0
        bin_is_gen_pred = th.logical_not(bin_is_exp_pred)
        bin_is_gen_true = th.logical_not(bin_is_exp_true)

        int_is_exp_pred = bin_is_exp_pred.long()
        int_is_exp_true = bin_is_exp_true.long()
        float_is_gen_pred = bin_is_gen_pred.float()
        float_is_gen_true = bin_is_gen_true.float()

        explained_var_gen = explained_variance(
            float_is_gen_pred.view(-1), float_is_gen_true.view(-1)
        )

        n_labels = float(len(labels))
        n_exp = float(th.sum(int_is_exp_true))
        n_gen = n_labels - n_exp

        percent_gen = n_gen / float(n_labels) if n_labels > 0 else float("NaN")
        n_gen_pred = int(n_labels - th.sum(int_is_exp_pred))

        if n_labels > 0:
            percent_gen_pred = n_gen_pred / float(n_labels)
        else:
            percent_gen_pred = float("NaN")

        correct_vec = th.eq(bin_is_exp_pred, bin_is_exp_true)
        disc_acc = th.mean(correct_vec.float())

        _n_pred_gen = th.sum(th.logical_and(bin_is_gen_true, correct_vec))
        if n_gen < 1:
            gen_acc = float("NaN")
        else:
            # float() is defensive, since we cannot divide Torch tensors by
            # Python ints
            gen_acc = _n_pred_gen / float(n_gen)

        _n_pred_exp = th.sum(th.logical_and(bin_is_exp_true, correct_vec))
        _n_exp_or_1 = max(1, n_exp)
        exp_acc = _n_pred_exp / float(_n_exp_or_1)

        label_dist = Bernoulli(logits=disc_logits)
        entropy = th.mean(label_dist.entropy())

    pairs = [
        ("disc_loss", float(th.mean(disc_loss))),
        # Accuracy, as well as accuracy on *just* expert examples and *just*
        # generated examples
        ("disc_acc", float(disc_acc)),
        ("disc_acc_gen", float(gen_acc)),
        ("disc_acc_exp", float(exp_acc)),
        # Entropy of the predicted label distribution, averaged equally across
        # both classes (if this drops then disc is very good or has given up)
        ("disc_entropy", float(entropy)),
        # True number of generators and predicted number of generators
        ("proportion_gen_true", float(percent_gen)),
        ("proportion_gen_pred", float(percent_gen_pred)),
        # interpretation:
        # ev=0  =>  might as well have predicted zero
        # ev=1  =>  perfect prediction
        # ev<0  =>  worse than just predicting zero
        ("explained_var_gen", float(explained_var_gen)),
    ]
    return OrderedDict(pairs)
