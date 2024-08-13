#
# SPDX-FileCopyrightText: Copyright (c) 2023 SRI International. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Abstract Base class for cameras, radios, and scene objects.
"""

from abc import ABC


class Object(ABC):
    # pylint: disable=line-too-long
    r"""Object(name)

    Baseclass for implementing items that are part of the scene.

    Input
    -----
    name : str
        Name of the object
    """

    # The following names are reserved and therefore cannot be used by objects
    RESERVED_NAMES = ('preview',)

    def __init__(self, name):

        # Set the name
        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        if name in Object.RESERVED_NAMES:
            msg = f"Cannot use name '{name}' as it is reserved."
            raise ValueError(msg)
        self._name = name

        # Scene to which this object belongs
        self._scene = None

    @property
    def name(self):
        """
        str (read-only) : Name
        """
        return self._name

    ##############################################
    # Internal methods.
    # Should not be appear in the end user
    # documentation
    ##############################################

    @property
    def scene(self):
        """
        :class:`~torchrf.rt.Scene` : Get/set the scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        self._scene = scene