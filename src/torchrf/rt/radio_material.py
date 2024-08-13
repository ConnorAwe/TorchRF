#
# SPDX-FileCopyrightText: Copyright (c) 2023 SRI International. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Implements a radio material.
A radio material provides the EM radio properties for a specific material.
"""

import torch

from . import scene
from torchrf.constants import DIELECTRIC_PERMITTIVITY_VACUUM, PI
from .scattering_pattern import ScatteringPattern, LambertianPattern


class RadioMaterial:
    # pylint: disable=line-too-long
    r"""
    Class implementing a radio material

    A radio material is defined by its relative permittivity
    :math:`\varepsilon_r` and conductivity :math:`\sigma` (see :eq:`eta`),
    as well as optional parameters related to diffuse scattering, such as the
    scattering coefficient :math:`S`, cross-polarization discrimination
    coefficient :math:`K_x`, and scattering pattern :math:`f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})`.

    We assume non-ionized and non-magnetic materials, and therefore the
    permeability :math:`\mu` of the material is assumed to be equal
    to the permeability of vacuum i.e., :math:`\mu_r=1.0`.

    For frequency-dependent materials, it is possible to
    specify a callback function ``frequency_update_callback`` that computes
    the material properties :math:`(\varepsilon_r, \sigma)` from the
    frequency. If a callback function is specified, the material properties
    cannot be set and the values specified at instantiation are ignored.
    The callback should return `-1` for both the relative permittivity and
    the conductivity if these are not defined for the given carrier frequency.

    The material properties are TensorFlow variables that can be made
    trainable.

    Parameters
    -----------
    name : str
        Unique name of the material

    relative_permittivity : float | `None`
        Relative permittivity of the material.
        Must be larger or equal to 1.
        Defaults to 1. Ignored if ``frequency_update_callback``
        is provided.

    conductivity : float | `None`
        Conductivity of the material [S/m].
        Must be non-negative.
        Defaults to 0.
        Ignored if ``frequency_update_callback``
        is provided.

    scattering_coefficient : float
        Scattering coefficient :math:`S\in[0,1]` as defined in
        :eq:`scattering_coefficient`.
        Defaults to 0.

    xpd_coefficient : float
        Cross-polarization discrimination coefficient :math:`K_x\in[0,1]` as
        defined in :eq:`xpd`.
        Only relevant if ``scattering_coefficient``>0.
        Defaults to 0.

    scattering_pattern : ScatteringPattern
        :class:`~torchrf.rt.ScatteringPattern` to applied.
        Only relevant if ``scattering_coefficient``>0.
        Defaults to `None`, which implies a :class:`~torchrf.rt.LambertianPattern`.

    frequency_update_callback : callable | `None`
        An optional callable object used to obtain the material parameters
        from the scene's :attr:`~torchrf.rt.Scene.frequency`.
        This callable must take as input the frequency [Hz] and
        must return the material properties as a tuple:

        ``(relative_permittivity, conductivity)``.

        If set to `None`, the material properties are constant and equal
        to ``relative_permittivity`` and ``conductivity``.
        Defaults to `None`.

    trainable_relative_permittivity : bool
        Determines if the ``relative_permittivity`` is trainable.
        Only possible if no ``frequency_update_callback``
        is defined.
        Defaults to `False`.

    trainable_conductivity : bool
        Determines if the ``conductivity`` is trainable.
        Only possible if no ``frequency_update_callback``
        is defined.
        Defaults to `False`.

    trainable_scattering_coefficient : bool
        Determines if the ``scattering_coefficient`` is trainable.
        Defaults to `False`.

    trainable_xpd_coefficient : bool
        Determines if the ``xpd_coefficient`` is trainable.
        Defaults to `False`.

    dtype : orch.complex64 or torch.complex128
        Datatype.
        Defaults to `torch.complex64`.
    """

    def __init__(self,
                 name,
                 relative_permittivity=1.0,
                 conductivity=0.0,
                 scattering_coefficient=0.0,
                 xpd_coefficient=0.0,
                 scattering_pattern=None,
                 frequency_update_callback=None,
                 trainable_relative_permittivity=False,
                 trainable_conductivity=False,
                 trainable_scattering_coefficient=False,
                 trainable_xpd_coefficient=False,
                 dtype=torch.complex64):

        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        self._name = name

        if dtype not in (torch.complex64, torch.complex128):
            msg = "`dtype` must be `torch.complex64` or `torch.complex128`"
            raise ValueError(msg)
        self._dtype = dtype
        if dtype == torch.complex64:
            self._rdtype = torch.float32
        elif dtype == torch.complex128:
            self._rdtype = torch.float64

        if scattering_pattern is None:
            scattering_pattern = LambertianPattern(dtype=dtype)

        self._relative_permittivity = torch.ones(0, dtype=self._rdtype)
        self._conductivity = torch.zeros(0, dtype=self._rdtype)
        self.scattering_pattern = scattering_pattern
        self._scattering_coefficient = torch.zeros(0, dtype=self._rdtype)
        self._xpd_coefficient = torch.zeros(0, dtype=self._rdtype)
        self.scattering_coefficient = scattering_coefficient
        self.xpd_coefficient = xpd_coefficient

        if frequency_update_callback is None:
            self.relative_permittivity = relative_permittivity
            self.conductivity = conductivity

        # Save the callback for when the frequency is updated
        self._frequency_update_callback = frequency_update_callback

        # Configure trainability if possible
        self.trainable_relative_permittivity = trainable_relative_permittivity
        self.trainable_conductivity = trainable_conductivity
        self.trainable_scattering_coefficient=trainable_scattering_coefficient
        self.trainable_xpd_coefficient = trainable_xpd_coefficient

        # Run frequency_update_callback to set the properties
        self.frequency_update(scene.Scene().frequency)

        # When loading a scene, the custom materials (i.e., the materials not
        # baked-in torchrf but defined by the user) are not defined yet.
        # If when loading a scene a non-defined material is encountered,
        # then a "placeholder" material is created which is used until the
        # material is defined by the user.
        # Note that propagation simulation cannot be done if placeholders are
        # used.
        self._is_placeholder = False # Is this material a placeholder

        # Set of objects identifiers that use this materials
        self._objects_using = set()


    @property
    def name(self):
        """
        str (read-only) : Name of the radio material
        """
        return self._name

    @property
    def relative_permittivity(self):
        r"""
        torch.float : Get/set the relative permittivity
            :math:`\varepsilon_r` :eq:`eta`
        """
        return self._relative_permittivity.clone()

    @relative_permittivity.setter
    def relative_permittivity(self, v):
        try:
            self._relative_permittivity = v.type(self._rdtype)
        except AttributeError:
            self._relative_permittivity = torch.tensor(v, dtype=self._rdtype)

    @property
    def relative_permeability(self):
        r"""
        torch.float (read-only) : Relative permeability
            :math:`\mu_r` :eq:`mu`.
            Defaults to 1.
        """
        return 1.0.type(self._rdtype)

    @property
    def conductivity(self):
        r"""
        torch.float: Get/set the conductivity
            :math:`\sigma` [S/m] :eq:`eta`
        """
        return self._conductivity.item()

    @conductivity.setter
    def conductivity(self, v):
        try:  # if v was a tensor
            self._conductivity = v.type(self._rdtype)
        except AttributeError:  # if v was a float
            self._conductivity = torch.tensor(v, dtype=self._rdtype)

    @property
    def scattering_coefficient(self):
        r"""
        torch.float: Get/set the scattering coefficient
            :math:`S\in[0,1]` :eq:`scattering_coefficient`.
        """
        return self._scattering_coefficient.item()

    @scattering_coefficient.setter
    def scattering_coefficient(self, v):
        if v>1 or v<0:
            raise ValueError("`scattering_coefficient` must be in [0,1]")
        if self.scattering_pattern is None and v>0:
            raise ValueError("Please configure a `scattering_pattern` first")
        try:
            self._scattering_coefficient = v.type(self._rdtype)
        except AttributeError:
            self._scattering_coefficient = torch.tensor(v, dtype=self._rdtype)

    @property
    def xpd_coefficient(self):
        r"""
        torch.float: Get/set the cross-polarization discrimination coefficient
            :math:`K_x\in[0,1]` :eq:`xpd`.
        """
        return self._xpd_coefficient.item()

    @xpd_coefficient.setter
    def xpd_coefficient(self, v):
        if v>1 or v<0:
            raise ValueError("`xpd_coefficient` must be in [0,1]")
        try:
            self._xpd_coefficient = v.type(self._rdtype)
        except AttributeError:
            self._xpd_coefficient = torch.tensor(v, dtype=self._rdtype)

    @property
    def scattering_pattern(self):
        r"""
        ScatteringPattern: Get/set the ScatteringPattern.
        """
        return self._scattering_pattern

    @scattering_pattern.setter
    def scattering_pattern(self, v):
        if not isinstance(v, ScatteringPattern) and v is not None:
            raise ValueError("Not a valid instanc of ScatteringPattern")
        self._scattering_pattern = v

    @property
    def complex_relative_permittivity(self):
        r"""
        torch.complex (read-only) : Complex relative permittivity
            :math:`\eta` :eq:`eta`
        """
        epsilon_0 = DIELECTRIC_PERMITTIVITY_VACUUM
        eta_prime = self.relative_permittivity
        sigma = self.conductivity
        frequency = scene.Scene().frequency
        omega = 2.*PI*frequency
        a = -torch.nan_to_num(torch.divide(sigma, epsilon_0*omega))
        return torch.complex(eta_prime, a)

    @property
    def trainable_relative_permittivity(self):
        """
        bool : Get/set if the ``relative permittivity``
            is a trainable variable or not.
            Defaults to `False`.
        """
        return self._trainable_relative_permittivity

    @trainable_relative_permittivity.setter
    def trainable_relative_permittivity(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable_relative_permittivity` must be bool")
        if value and self._frequency_update_callback is not None:
            err_msg = "Radio materials with frequency_update_callback" + \
                      " cannot be made trainable."
            raise ValueError(err_msg)
        # pylint: disable=protected-access
        self._relative_permittivity.requires_grad_(value)
        self._trainable_relative_permittivity = value

    @property
    def trainable_conductivity(self):
        """
        bool : Get/set if the ``conductivity``
            is a trainable variable or not.
            Defaults to `False`.
        """
        return self._trainable_conductivity

    @trainable_conductivity.setter
    def trainable_conductivity(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable_conductivity` must be bool")
        if value and self._frequency_update_callback is not None:
            err_msg = "Radio materials with frequency_update_callback" + \
                      " cannot be made trainable."
            raise ValueError(err_msg)
        # pylint: disable=protected-access
        self._conductivity.requires_grad_(value)
        self._trainable_conductivity = value

    @property
    def trainable_scattering_coefficient(self):
        """
        bool : Get/set if the ``scattering_coefficient``
            is a trainable variable or not.
            Defaults to `False`.
        """
        return self._trainable_scattering_coefficient

    @trainable_scattering_coefficient.setter
    def trainable_scattering_coefficient(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable_scattering_coefficient` must be bool")
        # pylint: disable=protected-access
        self._scattering_coefficient.requires_grad_(value)
        self._trainable_scattering_coefficient = value

    @property
    def trainable_xpd_coefficient(self):
        """
        bool : Get/set if the ``xpd_coefficient``
            is a trainable variable or not.
            Defaults to `False`.
        """
        return self._trainable_xpd_coefficient

    @trainable_xpd_coefficient.setter
    def trainable_xpd_coefficient(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable_xpd_coefficient` must be bool")
        # pylint: disable=protected-access
        self._xpd_coefficient.requires_grad_(value)
        self._trainable_xpd_coefficient = value

    @property
    def frequency_update_callback(self):
        """
        callable : Get/set frequency update callback function
        """
        return self._frequency_update_callback

    @frequency_update_callback.setter
    def frequency_update_callback(self, value):
        self._frequency_update_callback = value
        self.frequency_update(scene.Scene().frequency)

    @property
    def well_defined(self):
        """bool : Get if the material is well-defined"""
        # pylint: disable=chained-comparison
        return ((self._conductivity >= 0.)
             and (self.relative_permittivity >= 1.)
             and (0. <= self.scattering_coefficient <= 1.)
             and (0. <= self.xpd_coefficient<= 1.)
             and (0. <= self.scattering_pattern.lambda_ <= 1.))

    @property
    def use_counter(self):
        """
        int : Number of scene objects using this material
        """
        return len(self._objects_using)

    @property
    def is_used(self):
        """bool : Get if the material is used by at least one object of
        the scene"""
        return self.use_counter > 0

    @property
    def using_objects(self):
        """
        [num_using_objects], torch.int : Identifiers of the objects using this
        material
        """
        torch_objects_using = tuple(self._objects_using)
        return torch_objects_using

    ##############################################
    # Internal methods.
    # Should not be documented.
    ##############################################

    def frequency_update(self, fc):
        # pylint: disable=line-too-long
        r"""
        frequency_update(fc)

        Callback for when the frequency is updated

        Input
        ------
        fc : float
            The new value for the frequency [Hz]
        """
        if self._frequency_update_callback is None:
            return

        parameters = self._frequency_update_callback(fc)
        relative_permittivity, conductivity = parameters
        self.relative_permittivity = relative_permittivity
        self.conductivity = conductivity

    def add_object_using(self, object_id):
        """
        Add an object to the set of objects using this material
        """
        self._objects_using.add(object_id)

    def discard_object_using(self, object_id):
        """
        Remove an object from the set of objects using this material
        """
        assert object_id in self._objects_using,\
            f"Object with id {object_id} is not in the set of {self.name}"
        self._objects_using.discard(object_id)

    @property
    def is_placeholder(self):
        return self._is_placeholder

    @is_placeholder.setter
    def is_placeholder(self, v):
        self._is_placeholder = v