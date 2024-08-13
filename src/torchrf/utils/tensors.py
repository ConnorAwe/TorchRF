import torch
import numpy as np

def expand_to_rank(tensor, target_rank, axis=-1):
    """Inserts as many axes to a tensor as needed to achieve a desired rank.

    This operation inserts additional dimensions to a ``tensor`` starting at
    ``axis``, so that so that the rank of the resulting tensor has rank
    ``target_rank``. The dimension index follows Python indexing rules, i.e.,
    zero-based, where a negative index is counted backward from the end.

    Args:
        tensor : A tensor.
        target_rank (int) : The rank of the output tensor.
            If ``target_rank`` is smaller than the rank of ``tensor``,
            the function does nothing.
        axis (int) : The dimension index at which to expand the
               shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
               ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as ``tensor``, with
        ``target_rank``- rank(``tensor``) additional dimensions inserted at the
        index specified by ``axis``.
        If ``target_rank`` <= rank(``tensor``), ``tensor`` is returned.
    """
    num_dims = torch.sym_max(target_rank - len(tensor.shape), 0)
    output = insert_dims(tensor, num_dims, axis)

    return output


def insert_dims(tensor, num_dims, axis=-1):
    """Adds multiple length-one dimensions to a tensor.

    This operation is an extension to TensorFlow`s ``expand_dims`` function.
    It inserts ``num_dims`` dimensions of length one starting from the
    dimension ``axis`` of a ``tensor``. The dimension
    index follows Python indexing rules, i.e., zero-based, where a negative
    index is counted backward from the end.

    Args:
        tensor : A tensor.
        num_dims (int) : The number of dimensions to add.
        axis : The dimension index at which to expand the
               shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
               ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as ``tensor``, with ``num_dims`` additional
        dimensions inserted at the index specified by ``axis``.
    """
    rank = len(tensor.shape)

    axis = axis if axis >= 0 else rank+axis+1
    shape = torch.tensor(tensor.shape)
    new_shape = torch.concat([shape[:axis], torch.ones(num_dims, dtype=torch.int32), shape[axis:]], 0)
    output = torch.reshape(tensor, torch.Size(new_shape))

    return output


def split_dim(tensor, shape, axis):
    """Reshapes a dimension of a tensor into multiple dimensions.

    This operation splits the dimension ``axis`` of a ``tensor`` into
    multiple dimensions according to ``shape``.

    Args:
        tensor : A tensor.
        shape (list or TensorShape): The shape to which the dimension should
            be reshaped.
        axis (int): The index of the axis to be reshaped.

    Returns:
        A tensor of the same type as ``tensor`` with len(``shape``)-1
        additional dimensions, but the same number of elements.
    """
    s = torch.tensor(tensor.shape)
    new_shape = torch.concat([s[:axis], torch.tensor(shape), s[axis+1:]], 0)
    output = torch.reshape(tensor, torch.Size(new_shape))

    return output


def flatten_dims(tensor, num_dims, axis):
    """
    Flattens a specified set of dimensions of a tensor.

    This operation flattens ``num_dims`` dimensions of a ``tensor``
    starting at a given ``axis``.

    Args:
        tensor : A tensor.
        num_dims (int): The number of dimensions
            to combine. Must be larger than two and less or equal than the
            rank of ``tensor``.
        axis (int): The index of the dimension from which to start.

Returns:
    A tensor of the same type as ``tensor`` with ``num_dims``-1 lesser
    dimensions, but the same number of elements.
"""
    out = torch.flatten(tensor, axis, axis + num_dims - 1)
    return out

# def matrix_sqrt(tensor):
#     r""" Computes the square root of a matrix.
#
#     Given a batch of Hermitian positive semi-definite matrices
#     :math:`\mathbf{A}`, returns matrices :math:`\mathbf{B}`,
#     such that :math:`\mathbf{B}\mathbf{B}^H = \mathbf{A}`.
#
#     The two inner dimensions are assumed to correspond to the matrix rows
#     and columns, respectively.
#
#     Args:
#         tensor ([..., M, M]) : A tensor of rank greater than or equal
#             to two.
#
#     Returns:
#         A tensor of the same shape and type as ``tensor`` containing
#         the matrix square root of its last two dimensions.
#
#     Note:
#         If you want to use this function in Graph mode with XLA, i.e., within
#         a function that is decorated with ``@tf.function(jit_compile=True)``,
#         you must set ``torchrf.config.xla_compat=true``.
#         See :py:attr:`~torchrf.config.xla_compat`.
#     """
#     if sn.config.xla_compat and not tf.executing_eagerly():
#         s, u = tf.linalg.eigh(tensor)
#
#         # Compute sqrt of eigenvalues
#         s = tf.abs(s)
#         s = tf.sqrt(s)
#         s = tf.cast(s, u.dtype)
#
#         # Matrix multiplication
#         s = tf.expand_dims(s, -2)
#         return tf.matmul(u*s, u, adjoint_b=True)
#     else:
#         return tf.linalg.sqrtm(tensor)
#
# def matrix_sqrt_inv(tensor):
#     r""" Computes the inverse square root of a Hermitian matrix.
#
#     Given a batch of Hermitian positive definite matrices
#     :math:`\mathbf{A}`, with square root matrices :math:`\mathbf{B}`,
#     such that :math:`\mathbf{B}\mathbf{B}^H = \mathbf{A}`, the function
#     returns :math:`\mathbf{B}^{-1}`, such that
#     :math:`\mathbf{B}^{-1}\mathbf{B}=\mathbf{I}`.
#
#     The two inner dimensions are assumed to correspond to the matrix rows
#     and columns, respectively.
#
#     Args:
#         tensor ([..., M, M]) : A tensor of rank greater than or equal
#             to two.
#
#     Returns:
#         A tensor of the same shape and type as ``tensor`` containing
#         the inverse matrix square root of its last two dimensions.
#
#     Note:
#         If you want to use this function in Graph mode with XLA, i.e., within
#         a function that is decorated with ``@tf.function(jit_compile=True)``,
#         you must set ``torchrf.Config.xla_compat=true``.
#         See :py:attr:`~torchrf.Config.xla_compat`.
#     """
#     if sn.config.xla_compat and not tf.executing_eagerly():
#         s, u = tf.linalg.eigh(tensor)
#
#         # Compute 1/sqrt of eigenvalues
#         s = tf.abs(s)
#         tf.debugging.assert_positive(s, "Input must be positive definite.")
#         s = 1/tf.sqrt(s)
#         s = tf.cast(s, u.dtype)
#
#         # Matrix multiplication
#         s = tf.expand_dims(s, -2)
#         return tf.matmul(u*s, u, adjoint_b=True)
#     else:
#         return tf.linalg.inv(tf.linalg.sqrtm(tensor))
#
# def matrix_inv(tensor):
#     r""" Computes the inverse of a Hermitian matrix.
#
#     Given a batch of Hermitian positive definite matrices
#     :math:`\mathbf{A}`, the function
#     returns :math:`\mathbf{A}^{-1}`, such that
#     :math:`\mathbf{A}^{-1}\mathbf{A}=\mathbf{I}`.
#
#     The two inner dimensions are assumed to correspond to the matrix rows
#     and columns, respectively.
#
#     Args:
#         tensor ([..., M, M]) : A tensor of rank greater than or equal
#             to two.
#
#     Returns:
#         A tensor of the same shape and type as ``tensor``, containing
#         the inverse of its last two dimensions.
#
#     Note:
#         If you want to use this function in Graph mode with XLA, i.e., within
#         a function that is decorated with ``@tf.function(jit_compile=True)``,
#         you must set ``torchrf.Config.xla_compat=true``.
#         See :py:attr:`~torchrf.Config.xla_compat`.
#     """
#     if tensor.dtype in [tf.complex64, tf.complex128] \
#                     and sn.config.xla_compat \
#                     and not tf.executing_eagerly():
#         s, u = tf.linalg.eigh(tensor)
#
#         # Compute inverse of eigenvalues
#         s = tf.abs(s)
#         tf.debugging.assert_positive(s, "Input must be positive definite.")
#         s = 1/s
#         s = tf.cast(s, u.dtype)
#
#         # Matrix multiplication
#         s = tf.expand_dims(s, -2)
#         return tf.matmul(u*s, u, adjoint_b=True)
#     else:
#         return tf.linalg.inv(tensor)
#
# def matrix_pinv(tensor):
#     r""" Computes the Moore–Penrose (or pseudo) inverse of a matrix.
#
#     Given a batch of :math:`M \times K` matrices :math:`\mathbf{A}` with rank
#     :math:`K` (i.e., linearly independent columns), the function returns
#     :math:`\mathbf{A}^+`, such that
#     :math:`\mathbf{A}^{+}\mathbf{A}=\mathbf{I}_K`.
#
#     The two inner dimensions are assumed to correspond to the matrix rows
#     and columns, respectively.
#
#     Args:
#         tensor ([..., M, K]) : A tensor of rank greater than or equal
#             to two.
#
#     Returns:
#         A tensor of shape ([..., K,K]) of the same type as ``tensor``,
#         containing the pseudo inverse of its last two dimensions.
#
#     Note:
#         If you want to use this function in Graph mode with XLA, i.e., within
#         a function that is decorated with ``@tf.function(jit_compile=True)``,
#         you must set ``torchrf.config.xla_compat=true``.
#         See :py:attr:`~torchrf.config.xla_compat`.
#     """
#     inv = matrix_inv(tf.matmul(tensor, tensor, adjoint_a=True))
#     return tf.matmul(inv, tensor, adjoint_b=True)