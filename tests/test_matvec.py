import numpy as np
import tensorflow as tf
import torch
from torchrf.rt.utils import tf_matvec


def compare(a, b, transpose_a=False):
    """
    Assert that tf.linalg.matvec and torchrf.rt.utils.tf_matvec give the same answer on
    the given inputs.

    Parameters
    ----------
    a: np.ndarray
    b: np.ndarray
    transpose_a: bool

    Returns
    -------
    Not used
    """
    r1 = tf.linalg.matvec(tf.constant(a), tf.constant(b), transpose_a).numpy()
    r2 = tf_matvec(torch.tensor(a), torch.tensor(b), transpose_a).numpy()
    assert np.abs(r1-r2).max() < 1e-6


compare(np.random.rand(3, 5), np.random.rand(5))
compare(np.random.rand(3, 5), np.random.rand(3), True)
compare(np.random.rand(3, 1, 5, 3, 2), np.random.rand(3, 4, 1, 2))
