import random
import dataclasses
from time import sleep
from itertools import zip_longest
from collections import OrderedDict
from typing import Tuple, Dict, Any, Iterable

import numpy as np
import torch as th
from torch.distributions import Bernoulli


def set_random_seed(seed: int) -> None:
    """Set random seed to both numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def countdown(t_sec) -> None:
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


def combined_shape(length: int, shape=None):
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
        bin_is_gen_true = th.logical_not(bin_is_exp_true)

        int_is_exp_pred = bin_is_exp_pred.long()
        int_is_exp_true = bin_is_exp_true.long()

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
    ]
    return OrderedDict(pairs)
