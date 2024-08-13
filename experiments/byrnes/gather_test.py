import tensorflow as tf
from torchrf.rt.utils import tf_gather
import numpy as np
import torch

# TODO: Factor out comparison function for list args
# TODO: Full testing, move to tests
params = tf.constant([0, 1, 2, 3, 4, 5])
a = tf.gather(params, 3).numpy()
b = tf_gather(torch.tensor(params.numpy()), 3).numpy()
assert np.all(a == b)

indices = [2, 0, 2, 5]
a = tf.gather(params, indices).numpy()
b = tf_gather(torch.tensor(params.numpy()), indices).numpy()
assert np.all(a == b)
a = tf.gather(params, [[2, 0], [2, 5]]).numpy()
b = tf_gather(torch.tensor(params.numpy()), [[2, 0], [2, 5]]).numpy()
assert np.all(a == b)

params = tf.constant([[0, 1.0, 2.0],
                      [10.0, 11.0, 12.0],
                      [20.0, 21.0, 22.0],
                      [30.0, 31.0, 32.0]])
a = tf.gather(params, indices=[3,1]).numpy()
b = tf_gather(torch.tensor(params.numpy()), indices=[3,1]).numpy()
assert np.all(a == b)
a = tf.gather(params, indices=[2,1], axis=1).numpy()
b = tf_gather(torch.tensor(params.numpy()), indices=[2,1], axis=1).numpy()
assert np.all(a == b)

indices = tf.constant([[0, 2]])
a = tf.gather(params, indices=indices, axis=0)
b = tf_gather(torch.tensor(params.numpy()), indices=torch.tensor(indices.numpy()), axis=0)
assert np.all(a == b)
a = tf.gather(params, indices=indices, axis=1)
b = tf_gather(torch.tensor(params.numpy()), indices=torch.tensor(indices.numpy()), axis=1)
assert np.all(a == b)

params = tf.random.normal(shape=(5, 6, 7, 8))
indices = tf.random.uniform(shape=(10, 11), maxval=7, dtype=tf.int32)
a = tf.gather(params, indices, axis=2)
b = tf_gather(torch.tensor(params.numpy()), torch.tensor(indices.numpy()), axis=2)
assert np.all(a == b)

params = tf.constant([
    [0, 0, 1, 0, 2],
    [3, 0, 0, 0, 4],
    [0, 5, 0, 6, 0]])
indices = tf.constant([
    [2, 4],
    [0, 4],
    [1, 3]])
a = tf.gather(params, indices, axis=1).numpy()
b = tf_gather(torch.tensor(params.numpy()), torch.tensor(indices.numpy()), axis=1).numpy()
assert np.all(a == b)

params = tf.constant([[3, 5, 7],
                      [30, 50, 70],
                      [300, 500, 700]])
indices = tf.constant([[0, 2], [1, 2]])
a = tf.gather(params, indices, axis=0).numpy()
b = tf_gather(torch.tensor(params.numpy()), torch.tensor(indices.numpy()), axis=0).numpy()
assert np.all(a == b)
