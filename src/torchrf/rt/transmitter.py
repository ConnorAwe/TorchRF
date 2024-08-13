#
# SPDX-FileCopyrightText: Copyright (c) 2023 SRI International. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Class impelenting a transmitter
"""

import torch
from .radio_device import RadioDevice


class Transmitter(RadioDevice):
    # pylint: disable=line-too-long
    r"""
    Class defining a transmitter

    Parameters
    ----------
    name : str
        Name

    position : [3], float
        Position :math:`(x,y,z)` [m] as three-dimensional vector

    orientation : [3], float
        Orientation :math:`(\alpha, \beta, \gamma)` [rad] specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.
        Defaults to [0,0,0].

    look_at : [3], float | :class:`~torchrf.rt.Transmitter` | :class:`~torchrf.rt.Receiver` | :class:`~torchrf.rt.Camera` | None
        A position or the instance of a :class:`~torchrf.rt.Transmitter`,
        :class:`~torchrf.rt.Receiver`, or :class:`~torchrf.rt.Camera` to look at.
        If set to `None`, then ``orientation`` is used to orientate the device.

    trainable_position : bool
        Determines if the ``position`` is a trainable variable or not.
        Defaults to `False`.

    trainable_orientation : bool
        Determines if the ``orientation`` is a trainable variable or not.
        Defaults to `False`.

    dtype : torch.complex
        Datatype to be used in internal calculations.
        Defaults to `torch.complex64`.
    """

    def __init__(self,
                 name,
                 position,
                 orientation=(0.,0.,0.),
                 look_at=None,
                 trainable_position=False,
                 trainable_orientation=False,
                 dtype=torch.complex64):

        # Initialize the base class Object
        super().__init__(name=name,
                         position=position,
                         orientation=orientation,
                         look_at=look_at,
                         trainable_position=trainable_position,
                         trainable_orientation=trainable_orientation,
                         dtype=dtype)