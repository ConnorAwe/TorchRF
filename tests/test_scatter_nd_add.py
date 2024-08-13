import numpy as np
import torch
import tensorflow as tf
from torchrf.rt.utils import scatter_nd_add
from torchrf.utils.datalogger import DataLogger
logger = DataLogger().set_mode('break')
logger.set_dir("testdata")
DataLogger.load("scatter_nd_add.json")


def compare(t, ind, update, expect=None, tol=1e-6):
    out_tf = tf.tensor_scatter_nd_add(tf.constant(t), tf.constant(ind), tf.constant(update))
    out_tf = out_tf.numpy()
    if expect is not None:
        assert np.abs(out_tf - expect).max() < tol

    out_t = scatter_nd_add(torch.tensor(t), torch.tensor(ind), torch.tensor(update))
    out_t = out_t.numpy()
    assert np.abs(out_tf - out_t).max() < tol


# Test 1:  From Tensorflow documentation
indices = np.array([[4], [3], [1], [7]], dtype=np.int64)
updates = np.array([9, 10, 11, 12], dtype=np.int64)
tensor = np.ones([8], dtype=np.int64)
expected = np.array([1, 12, 1, 11, 10, 1, 1, 13])
compare(tensor, indices, updates, expected)


# Test 2:  From Tensorflow documentation
indices = np.array([[0], [2]], dtype=np.int64)
updates = np.array([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]], dtype=np.int64)
tensor = np.ones([4, 4, 4], dtype=np.int64)
expected = np.array([[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
                     [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                     [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
                     [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]], dtype=np.int64)
compare(tensor, indices, updates, expected)


# Test 3: From Coverage Map script full run
tensor, indices, updates = logger.get("inputs")
expected = logger.get("output")
compare(tensor, indices, updates, expected)
