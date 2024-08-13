#
# SPDX-FileCopyrightText: Copyright (c) 2023 SRI International. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Implements classes and methods related to antennas.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torchrf.constants import PI
import torch
from collections.abc import Sequence


class Antenna:
    r"""
    Class implementing an antenna

    Creates an antenna object with an either predefined or custom antenna
    pattern. Can be single or dual polarized.

    Parameters
    ----------
    pattern : str, callable, or length-2 sequence of callables
        Antenna pattern. Either one of
        ["iso", "dipole", "hw_dipole", "tr38901"],
        or a callable, or a length-2 sequence of callables defining
        antenna patterns. In the latter case, the antenna is dual
        polarized and each callable defines the antenna pattern
        in one of the two orthogonal polarization directions.
        An antenna pattern is a callable that takes as inputs vectors of
        zenith and azimuth angles of the same length and returns for each
        pair the corresponding zenith and azimuth patterns.

    polarization : str or None
        Type of polarization. For single polarization, must be "V" (vertical)
        or "H" (horizontal). For dual polarization, must be "VH" or "cross".
        Only needed if ``pattern`` is a string.

    polarization_model: int, one of [1,2]
        Polarization model to be used. Options `1` and `2`
        refer to :func:`~torchrf.rt.antenna.polarization_model_1`
        and :func:`~torchrf.rt.antenna.polarization_model_2`,
        respectively.
        Defaults to `2`.

    dtype : torch.complex64 or torch.complex128
        Datatype used for all computations.
        Defaults to `torch.complex64`.

    Example
    -------
    >>> Antenna("tr38901", "VH")
    """

    def __init__(self,
                 pattern,
                 polarization=None,
                 polarization_model=2,
                 dtype=torch.complex64
                 ):

        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError("`dtype` must be torch.complex64 or torch.complex128`")
        self._dtype = dtype = dtype

        if polarization_model not in [1, 2]:
            raise ValueError("`polarization_model` must be 1 or 2")
        self._polarization_model = polarization_model

        # Pattern is provided as string
        if isinstance(pattern, str):

            # Set correct pattern
            if pattern == "iso":
                pattern = iso_pattern
            elif pattern == "dipole":
                pattern = dipole_pattern
            elif pattern == "hw_dipole":
                pattern = hw_dipole_pattern
            elif pattern == "tr38901":
                pattern = tr38901_pattern
            else:
                raise ValueError("Unknown antenna pattern")

            # Set slant angles
            if polarization == "V":
                slant_angles = [0.0]
            elif polarization == "H":
                slant_angles = [PI / 2]
            elif polarization == "VH":
                slant_angles = [0.0, PI / 2]
            elif polarization == "cross":
                slant_angles = [-PI / 4, PI / 4]
            else:
                raise ValueError("Unknown polarization")

            # Create antenna patterns with slant angles
            self._patterns = []
            for sa in slant_angles:
                f = self.pattern_with_slant_angle(pattern, sa)
                self._patterns.append(f)

        # Pattern is a callable
        elif callable(pattern):
            self._patterns = [pattern]

        # Pattern is sequence of callables
        elif isinstance(pattern, Sequence):
            if len(pattern) > 2:
                msg = "An antennta cannot have more than two patterns."
                raise ValueError(msg)
            for p in pattern:
                if not callable(p):
                    msg = "Each element of antenna_pattern must be callable"
                    raise ValueError(msg)
            self._patterns = pattern

        # Unsupported pattern
        else:
            raise ValueError("Unsupported pattern")

    @property
    def patterns(self):
        """
        `list`, `callable` : Antenna patterns for one or two
            polarization directions
        """
        return self._patterns

    def pattern_with_slant_angle(self, pattern, slant_angle):
        """Applies slant angle to antenna pattern"""
        return lambda theta, phi: pattern(theta, phi, slant_angle,
                                          self._polarization_model, self._dtype)


def compute_gain(pattern, dtype=torch.complex64):
    # pylint: disable=line-too-long
    r"""compute_gain(pattern)
    Computes the directivity, gain, and radiation efficiency of an antenna pattern

    Given a function :math:`f:(\theta,\varphi)\mapsto (C_\theta(\theta, \varphi), C_\varphi(\theta, \varphi))`
    describing an antenna pattern :eq:`C`, this function computes the gain :math:`G`,
    directivity :math:`D`, and radiation efficiency :math:`\eta_\text{rad}=G/D`
    (see :eq:`G` and text below).

    Input
    -----
    pattern : callable
        A callable that takes as inputs vectors of zenith and azimuth angles of the same
        length and returns for each pair the corresponding zenith and azimuth patterns.

    Output
    ------
    D : float
        Directivity :math:`D`

    G : float
        Gain :math:`G`

    eta_rad : float
        Radiation efficiency :math:`\eta_\text{rad}`

    Examples
    --------
    >>> compute_gain(tr38901_pattern)
    (<torch.Tensor: shape=(), dtype=float32, numpy=9.606758>,
     <torch.Tensor: shape=(), dtype=float32, numpy=6.3095527>,
     <torch.Tensor: shape=(), dtype=float32, numpy=0.65678275>)
    """

    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("`dtype` must be torch.complex64 or torch.complex128`")
    if dtype == torch.complex64:
        real_dtype = torch.float32
    elif dtype == torch.complex128:
        real_dtype = torch.float64

    # Create angular meshgrid
    theta = torch.linspace(0.0, PI, 1810)
    theta = theta.type(real_dtype)
    phi = torch.linspace(-PI, PI, 3610)
    phi = phi.type(real_dtype)

    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")

    # Compute the gain
    c_theta, c_phi = pattern(theta_grid, phi_grid)
    g = torch.abs(c_theta) ** 2 + torch.abs(c_phi) ** 2

    # Find maximum directional gain
    g_max = torch.max(g)

    # Compute radiation efficiency
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    eta_rad = torch.sum(g * torch.sin(theta_grid) * dtheta * dphi) / (4 * PI)

    # Compute directivity
    d = g_max / eta_rad
    return d, g_max, eta_rad


def visualize(pattern):
    r"""visualize(pattern)
    Visualizes an antenna pattern

    This function visualizes an antenna pattern with the help of three
    figures showing the vertical and horizontal cuts as well as a
    three-dimensional visualization of the antenna gain.

    Input
    -----
    pattern : callable
        A callable that takes as inputs vectors of zenith and azimuth angles
        of the same length and returns for each pair the corresponding zenith
        and azimuth patterns.

    Output
    ------
     : :class:`matplotlib.pyplot.Figure`
        Vertical cut of the antenna gain

     : :class:`matplotlib.pyplot.Figure`
        Horizontal cut of the antenna gain

     : :class:`matplotlib.pyplot.Figure`
        3D visualization of the antenna gain

    Examples
    --------
    >>> fig_v, fig_h, fig_3d = visualize(hw_dipole_pattern)

    .. figure:: ../figures/pattern_vertical.png
        :align: center
        :scale: 80%
    .. figure:: ../figures/pattern_horizontal.png
        :align: center
        :scale: 80%
    .. figure:: ../figures/pattern_3d.png
        :align: center
        :scale: 80%
    """
    # Vertical cut
    theta = np.linspace(0.0, PI, 1000)
    c_theta, c_phi = pattern(theta, np.zeros_like(theta))
    g = np.abs(c_theta) ** 2 + np.abs(c_phi) ** 2
    g = np.where(g == 0, 1e-12, g)
    g_db = 10 * np.log10(g)
    g_db_max = np.max(g_db)
    g_db_min = np.min(g_db)
    if g_db_min == g_db_max:
        g_db_min = -30
    else:
        g_db_min = np.maximum(-60., g_db_min)
    fig_v = plt.figure()
    plt.polar(theta, g_db)
    fig_v.axes[0].set_rmin(g_db_min)
    fig_v.axes[0].set_rmax(g_db_max + 3)
    fig_v.axes[0].set_theta_zero_location("N")
    fig_v.axes[0].set_theta_direction(-1)
    plt.title(r"Vertical cut of the radiation pattern $G(\theta,0)$ ")

    # Horizontal cut
    phi = np.linspace(-PI, PI, 1000)
    c_theta, c_phi = pattern(PI / 2 * torch.ones_like(torch.tensor(phi)),
                             torch.tensor(phi, dtype=torch.float32))
    c_theta = c_theta.numpy()
    c_phi = c_phi.numpy()
    g = np.abs(c_theta) ** 2 + np.abs(c_phi) ** 2
    g = np.where(g == 0, 1e-12, g)
    g_db = 10 * np.log10(g)
    g_db_max = np.max(g_db)
    g_db_min = np.min(g_db)
    if g_db_min == g_db_max:
        g_db_min = -30
    else:
        g_db_min = np.maximum(-60., g_db_min)

    fig_h = plt.figure()
    plt.polar(phi, g_db)
    fig_h.axes[0].set_rmin(g_db_min)
    fig_h.axes[0].set_rmax(g_db_max + 3)
    fig_h.axes[0].set_theta_zero_location("E")
    plt.title(r"Horizontal cut of the radiation pattern $G(\pi/2,\varphi)$")

    # 3D visualization
    theta = np.linspace(0.0, PI, 50)
    phi = np.linspace(-PI, PI, 50)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    c_theta, c_phi = pattern(theta_grid, phi_grid)
    g = np.abs(c_theta) ** 2 + np.abs(c_phi) ** 2
    x = g * np.sin(theta_grid) * np.cos(phi_grid)
    y = g * np.sin(theta_grid) * np.sin(phi_grid)
    z = g * np.cos(theta_grid)

    g = np.maximum(g, 1e-5)
    g_db = 10 * np.log10(g)

    def norm(x, x_max, x_min):
        """Maps input to [0,1] range"""
        x = 10 ** (x / 10)
        x_max = 10 ** (x_max / 10)
        x_min = 10 ** (x_min / 10)
        if x_min == x_max:
            x = np.ones_like(x)
        else:
            x -= x_min
            x /= np.abs(x_max - x_min)
        return x

    g_db_min = np.min(g_db)
    g_db_max = np.max(g_db)

    fig_3d = plt.figure()
    ax = fig_3d.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, alpha=0.7,
                    facecolors=cm.turbo(norm(g_db, g_db_max, g_db_min)))

    sm = cm.ScalarMappable(cmap=plt.cm.turbo)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", location="right",
                        shrink=0.7, pad=0.15)
    xticks = cbar.ax.get_yticks()
    xticklabels = cbar.ax.get_yticklabels()
    xticklabels = g_db_min + xticks * (g_db_max - g_db_min)
    xticklabels = [f"{z:.2f} dB" for z in xticklabels]
    cbar.ax.set_yticks(xticks)
    cbar.ax.set_yticklabels(xticklabels)

    ax.view_init(elev=30., azim=-45)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("z")
    plt.suptitle(
        r"3D visualization of the radiation pattern $G(\theta,\varphi)$")

    return fig_v, fig_h, fig_3d


def polarization_model_1(c_theta, theta, phi, slant_angle):
    # pylint: disable=line-too-long
    r"""Model-1 for polarized antennas from 3GPP TR 38.901

    Transforms a vertically polarized antenna pattern :math:`\tilde{C}_\theta(\theta, \varphi)`
    into a linearly polarized pattern whose direction
    is specified by a slant angle :math:`\zeta`. For example,
    :math:`\zeta=0` and :math:`\zeta=\pi/2` correspond
    to vertical and horizontal polarization, respectively,
    and :math:`\zeta=\pm \pi/4` to a pair of cross polarized
    antenna elements.

    The transformed antenna pattern is given by (7.3-3) [TR38901]_: 
    

    .. math::
        \begin{align}
            \begin{bmatrix}
                C_\theta(\theta, \varphi) \\
                C_\varphi(\theta, \varphi)
            \end{bmatrix} &= \begin{bmatrix}
             \cos(\psi) \\
             \sin(\psi)
            \end{bmatrix} \tilde{C}_\theta(\theta, \varphi)\\
            \cos(\psi) &= \frac{\cos(\zeta)\sin(\theta)+\sin(\zeta)\sin(\varphi)\cos(\theta)}{\sqrt{1-\left(\cos(\zeta)\cos(\theta)-\sin(\zeta)\sin(\varphi)\sin(\theta)\right)^2}} \\
            \sin(\psi) &= \frac{\sin(\zeta)\cos(\varphi)}{\sqrt{1-\left(\cos(\zeta)\cos(\theta)-\sin(\zeta)\sin(\varphi)\sin(\theta)\right)^2}} 
        \end{align}


    Input
    -----
    c_tilde_theta: array_like, complex
        Zenith pattern

    theta: array_like, float
        Zenith angles wrapped within [0,pi] [rad]

    phi: array_like, float
        Azimuth angles wrapped within [-pi, pi) [rad]

    slant_angle: float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    Output
    ------
    c_theta: array_like, complex
        Zenith pattern

    c_phi: array_like, complex
        Azimuth pattern
    """
    if slant_angle == 0:
        return c_theta, torch.zeros_like(c_theta)
    if slant_angle == PI / 2:
        return torch.zeros_like(c_theta), c_theta
    sin_slant = torch.sin(slant_angle).type(theta.dtype)
    cos_slant = torch.cos(slant_angle).type(theta.dtype)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_psi = sin_slant * cos_phi
    cos_psi = cos_slant * sin_theta + sin_slant * sin_phi * cos_theta
    norm = torch.sqrt(1 - (cos_slant * cos_theta - sin_slant * sin_phi * sin_theta) ** 2)
    sin_psi = torch.divide(sin_psi, norm)
    sin_psi[sin_psi != sin_psi] = 0  # Convert to safe division.
    cos_psi = torch.divide(cos_psi, norm)
    cos_psi[cos_psi != cos_psi] = 0  # Convert to safe division.
    c_theta = c_theta * torch.complex(cos_psi, torch.zeros_like(cos_psi))
    c_phi = c_theta * torch.complex(sin_psi, torch.zeros_like(sin_psi))
    return c_theta, c_phi


def polarization_model_2(c, slant_angle):
    # pylint: disable=line-too-long
    r"""Model-2 for polarized antennas from 3GPP TR 38.901

    Transforms a vertically polarized antenna pattern :math:`\tilde{C}_\theta(\theta, \varphi)`
    into a linearly polarized pattern whose direction
    is specified by a slant angle :math:`\zeta`. For example,
    :math:`\zeta=0` and :math:`\zeta=\pi/2` correspond
    to vertical and horizontal polarization, respectively,
    and :math:`\zeta=\pm \pi/4` to a pair of cross polarized
    antenna elements.

    The transformed antenna pattern is given by (7.3-4/5) [TR38901]_: 

    .. math::
        \begin{align}
            \begin{bmatrix}
                C_\theta(\theta, \varphi) \\
                C_\varphi(\theta, \varphi)
            \end{bmatrix} &= \begin{bmatrix}
             \cos(\zeta) \\
             \sin(\zeta)
            \end{bmatrix} \tilde{C}_\theta(\theta, \varphi)
        \end{align}

    Input
    -----
    c_tilde_theta: array_like, complex
        Zenith pattern

    slant_angle: float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    Output
    ------
    c_theta: array_like, complex
        Zenith pattern

    c_phi: array_like, complex
        Azimuth pattern
    """
    c_theta = c * (np.cos(slant_angle) + 0j)
    c_phi = c * (np.sin(slant_angle) + 0j)
    return c_theta, c_phi


def iso_pattern(theta, phi, slant_angle=0.0,
                polarization_model=2, dtype=torch.complex64):
    r"""
    Isotropic antenna pattern with linear polarizarion

    Input
    -----
    theta: array_like, float
        Zenith angles wrapped within [0,pi] [rad]

    phi: array_like, float
        Azimuth angles wrapped within [-pi, pi) [rad]

    slant_angle: float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    polarization_model: int, one of [1,2]
        Polarization model to be used. Options `1` and `2`
        refer to :func:`~torchrf.rt.antenna.polarization_model_1`
        and :func:`~torchrf.rt.antenna.polarization_model_2`,
        respectively.
        Defaults to `2`.

    dtype : torch.complex64 or torch.complex128
        Datatype.
        Defaults to `torch.complex64`.

    Output
    ------
    c_theta: array_like, complex
        Zenith pattern

    c_phi: array_like, complex
        Azimuth pattern


    .. figure:: ../figures/iso_pattern.png
        :align: center
    """
    if dtype == torch.complex64:
        rdtype = torch.float32
    elif dtype == torch.complex128:
        rdtype = torch.float64

    theta = theta.type(rdtype)
    phi = phi.type(rdtype)

    if not theta.shape == phi.shape:
        raise ValueError("theta and phi must have the same shape.")
    if polarization_model not in [1, 2]:
        raise ValueError("polarization_model must be 1 or 2")
    c = torch.ones_like(theta, dtype=dtype)
    if polarization_model == 1:
        return polarization_model_1(c, theta, phi, slant_angle)
    else:
        return polarization_model_2(c, slant_angle)


def dipole_pattern(theta, phi, slant_angle=0.0,
                   polarization_model=2, dtype=torch.complex64):
    r"""
    Short dipole pattern with linear polarizarion (Eq. 4-26a) [Balanis97]_

    Input
    -----
    theta: array_like, float
        Zenith angles wrapped within [0,pi] [rad]

    phi: array_like, float
        Azimuth angles wrapped within [-pi, pi) [rad]

    slant_angle: float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    polarization_model: int, one of [1,2]
        Polarization model to be used. Options `1` and `2`
        refer to :func:`~torchrf.rt.antenna.polarization_model_1`
        and :func:`~torchrf.rt.antenna.polarization_model_2`,
        respectively.
        Defaults to `2`.

    dtype : torch.complex64 or torch.complex128
        Datatype.
        Defaults to `torch.complex64`.

    Output
    ------
    c_theta: array_like, complex
        Zenith pattern

    c_phi: array_like, complex
        Azimuth pattern


    .. figure:: ../figures/dipole_pattern.png
        :align: center
    """
    rdtype = torch.tensor([0], dtype=dtype).real.dtype
    k = np.sqrt(1.5)
    theta = theta.type(rdtype)
    phi = phi.type(rdtype)
    if not theta.shape == phi.shape:
        raise ValueError("theta and phi must have the same shape.")
    if polarization_model not in [1, 2]:
        raise ValueError("polarization_model must be 1 or 2")
    c = k * torch.complex(torch.sin(theta), torch.zeros_like(theta))
    if polarization_model == 1:
        return polarization_model_1(c, theta, phi, slant_angle)
    else:
        return polarization_model_2(c, slant_angle)


def hw_dipole_pattern(theta, phi, slant_angle=0.0,
                      polarization_model=2, dtype=torch.complex64):
    # pylint: disable=line-too-long
    r"""
    Half-wavelength dipole pattern with linear polarizarion (Eq. 4-84) [Balanis97]_

    Input
    -----
    theta: array_like, float
        Zenith angles wrapped within [0,pi] [rad]

    phi: array_like, float
        Azimuth angles wrapped within [-pi, pi) [rad]

    slant_angle: float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    polarization_model: int, one of [1,2]
        Polarization model to be used. Options `1` and `2`
        refer to :func:`~torchrf.rt.antenna.polarization_model_1`
        and :func:`~torchrf.rt.antenna.polarization_model_2`,
        respectively.
        Defaults to `2`.

    dtype : torch.complex64 or torch.complex128
        Datatype.
        Defaults to `torch.complex64`.

    Output
    ------
    c_theta: array_like, complex
        Zenith pattern

    c_phi: array_like, complex
        Azimuth pattern


    .. figure:: ../figures/hw_dipole_pattern.png
        :align: center
    """
    rdtype = torch.tensor([0], dtype=dtype).real.dtype
    k = np.sqrt(1.643)
    theta = theta.type(rdtype)
    phi = phi.type(rdtype)
    if not theta.shape == phi.shape:
        raise ValueError("theta and phi must have the same shape.")
    if polarization_model not in [1, 2]:
        raise ValueError("polarization_model must be 1 or 2")
    c = k * torch.divide(torch.cos(PI / 2 * torch.cos(theta)), torch.sin(theta))
    c[c != c] = 0
    c = torch.complex(c, torch.zeros_like(c))
    if polarization_model == 1:
        return polarization_model_1(c, theta, phi, slant_angle)
    else:
        return polarization_model_2(c, slant_angle)


def tr38901_pattern(theta, phi, slant_angle=0.0,
                    polarization_model=2, dtype=torch.complex64):
    r"""
    Antenna pattern from 3GPP TR 38.901 (Table 7.3-1) [TR38901]_

    Input
    -----
    theta: array_like, float
        Zenith angles wrapped within [0,pi] [rad]

    phi: array_like, float
        Azimuth angles wrapped within [-pi, pi) [rad]

    slant_angle: float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    polarization_model: int, one of [1,2]
        Polarization model to be used. Options `1` and `2`
        refer to :func:`~torchrf.rt.antenna.polarization_model_1`
        and :func:`~torchrf.rt.antenna.polarization_model_2`,
        respectively.
        Defaults to `2`.

    dtype : torch.complex64 or torch.complex128
        Datatype.
        Defaults to `torch.complex64`.

    Output
    ------
    c_theta: array_like, complex
        Zenith pattern

    c_phi: array_like, complex
        Azimuth pattern


    .. figure:: ../figures/tr38901_pattern.png
        :align: center
    """
    rdtype = torch.tensor([0], dtype=dtype).real.dtype
    theta = theta.type(rdtype)
    phi = phi.type(rdtype)

    # Wrap phi to [-PI,PI]
    phi = torch.fmod(phi + PI, 2 * PI) - PI

    if not theta.shape == phi.shape:
        raise ValueError("theta and phi must have the same shape.")
    if polarization_model not in [1, 2]:
        raise ValueError("polarization_model must be 1 or 2")
    theta_3db = phi_3db = (65 / 180 * PI)
    a_max = sla_v = torch.full((12 * ((theta - PI / 2) / theta_3db) ** 2).shape, fill_value=30)
    g_e_max = 8
    a_v = -torch.minimum(12 * ((theta - PI / 2) / theta_3db) ** 2, sla_v)
    a_h = -torch.minimum(12 * (phi / phi_3db) ** 2, a_max)
    a_db = -torch.minimum(-(a_v + a_h), a_max) + g_e_max
    a = 10 ** (a_db / 10)
    c = torch.complex(torch.sqrt(a), torch.zeros_like(a))
    if polarization_model == 1:
        return polarization_model_1(c, theta, phi, slant_angle)
    else:
        return polarization_model_2(c, slant_angle)
