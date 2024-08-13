#
# SPDX-FileCopyrightText: Copyright (c) 2023 SRI International. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""Layer for simulating an AWGN channel"""

import torch
from tensorflow.keras.layers import Layer
from torchrf.utils import expand_to_rank, complex_normal
from utils import shape

class AWGN(Layer):
    r"""AWGN(dtype=torch.complex64, **kwargs)

    Add complex AWGN to the inputs with a certain variance.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    This layer adds complex AWGN noise with variance ``no`` to the input.
    The noise has variance ``no/2`` per real dimension.
    It can be either a scalar or a tensor which can be broadcast to the shape
    of the input.

    Example
    --------

    Setting-up:

    >>> awgn_channel = AWGN()

    Running:

    >>> # x is the channel input
    >>> # no is the noise variance
    >>> y = awgn_channel((x, no))

    Parameters
    ----------
        dtype : Complex torch.DType
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `torch.complex64`.

    Input
    -----

        (x, no) :
            Tuple:

        x :  Tensor, torch.complex
            Channel input

        no : Scalar or Tensor, tf.float
            Scalar or tensor whose shape can be broadcast to the shape of ``x``.
            The noise power ``no`` is per complex dimension. If ``no`` is a
            scalar, noise of the same variance will be added to the input.
            If ``no`` is a tensor, it must have a shape that can be broadcast to
            the shape of ``x``. This allows, e.g., adding noise of different
            variance to each example in a batch. If ``no`` has a lower rank than
            ``x``, then ``no`` will be broadcast to the shape of ``x`` by adding
            dummy dimensions after the last axis.

    Output
    -------
        y : Tensor with same shape as ``x``, torch.complex
            Channel output
    """

    def __init__(self, dtype=torch.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._real_dtype = torch.DTypes.as_dtype(self._dtype).real_dtype

    def call(self, inputs):

        x, no = inputs

        # Create tensors of real-valued Gaussian noise for each complex dim.
        noise = complex_normal(shape(x), dtype=x.dtype)

        # Add extra dimensions for broadcasting
        no = expand_to_rank(no, x.dim(), axis=-1)

        # Apply variance scaling
        no = no.type(self._real_dtype)
        noise *= torch.sqrt(no).torch(noise.dtype)

        # Add noise to input
        y = x + noise

        return y