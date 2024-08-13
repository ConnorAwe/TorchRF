#
# SPDX-FileCopyrightText: Copyright (c) 2023 SRI International. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Dataclass that stores paths
"""

import torch
import os
import torch.nn.functional as F

from torchrf.utils.tensors import expand_to_rank, insert_dims
from torchrf.constants import PI
from .utils import dot, r_hat, rank, scatter_nd_update


class Paths:
    # pylint: disable=line-too-long
    r"""
    Paths()

    Stores the simulated propagation paths

    Paths are generated for the loaded scene using
    :meth:`~torchrf.rt.Scene.compute_paths`. Please refer to the
    documentation of this function for further details.
    These paths can then be used to compute channel impulse responses:

    .. code-block:: Python

        paths = scene.compute_paths()
        a, tau = paths.cir()

    where ``scene`` is the :class:`~torchrf.rt.Scene` loaded using
    :func:`~torchrf.rt.load_scene`.
    """

    # Input
    # ------
    # a : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], torch.complex
    #     Channel coefficients :math:`a_i` as defined in :eq:`T_tilde`.
    #     If there are less than `max_num_path` valid paths between a
    #     transmit and receive antenna, the irrelevant elements are
    #     filled with zeros.

    # tau : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float
    #     Propagation delay of each path [s].
    #     If :attr:`~torchrf.rt.Scene.synthetic_array` is `True`, the shape of this tensor
    #     is `[1, num_rx, num_tx, max_num_paths]` as the delays for the
    #     individual antenna elements are assumed to be equal.
    #     If there are less than `max_num_path` valid paths between a
    #     transmit and receive antenna, the irrelevant elements are
    #     filled with -1.

    # theta_t : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float
    #     Zenith  angles of departure :math:`\theta_{\text{T},i}` [rad].
    #     If :attr:`~torchrf.rt.Scene.synthetic_array` is `True`, the shape of this tensor
    #     is `[1, num_rx, num_tx, max_num_paths]` as the angles for the
    #     individual antenna elements are assumed to be equal.

    # phi_t : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float
    #     Azimuth angles of departure :math:`\varphi_{\text{T},i}` [rad].
    #     See description of ``theta_t``.

    # theta_r : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float
    #     Zenith angles of arrival :math:`\theta_{\text{R},i}` [rad].
    #     See description of ``theta_t``.

    # phi_r : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float
    #     Azimuth angles of arrival :math:`\varphi_{\text{T},i}` [rad].
    #     See description of ``theta_t``.

    # types : [batch_size, max_num_paths], torch.int
    #     Type of path:

    #     - 0 : LoS
    #     - 1 : Reflected
    #     - 2 : Diffracted
    #     - 3 : Scattered

    # Types of paths
    LOS = 0
    SPECULAR = 1
    DIFFRACTED = 2
    SCATTERED = 3

    def __init__(self,
                 sources,
                 targets,
                 scene,
                 types=None):

        dtype = scene.dtype
        rdtype = scene.dtype.to_real()
        num_sources = sources.shape[0]
        num_targets = targets.shape[0]

        self._a = torch.zeros(num_targets, num_sources, 0, dtype=dtype)
        self._tau = torch.zeros([num_targets, num_sources, 0], dtype=rdtype)
        self._theta_t = torch.zeros([num_targets, num_sources, 0], dtype=rdtype)
        self._theta_r = torch.zeros([num_targets, num_sources, 0], dtype=rdtype)
        self._phi_t = torch.zeros([num_targets, num_sources, 0], dtype=rdtype)
        self._phi_r = torch.zeros([num_targets, num_sources, 0], dtype=rdtype)
        self._mask = torch.zeros([num_targets, num_sources, 0], dtype=torch.bool)
        self._vertices = torch.zeros([0, num_targets, num_sources, 0, 3], dtype=rdtype)
        self._objects = torch.full([0, num_targets, num_sources, 0], -1, dtype=torch.int32)
        if types is None:
            self._types = torch.full([0], -1)
        else:
            self._types = types

        self._sources = sources
        self._targets = targets
        self._scene = scene

        # Is the direction reversed?
        self._reverse_direction = False
        # Normalize paths delays?
        self._normalize_delays = False

    def export(self, filename):
        r"""
        export(filename)

        Saves the paths as an OBJ file for visualisation, e.g., in Blender

        Input
        ------
        filename : str
            Path and name of the file
        """
        vertices = self.vertices
        objects = self.objects
        sources = self.sources
        targets = self.targets
        mask = self.mask

        # Content of the obj file
        r = ''
        offset = 0
        for rx in range(vertices.shape[1]):
            tgt = targets[rx].numpy()
            for tx in range(vertices.shape[2]):
                src = sources[tx].numpy()
                for p in range(vertices.shape[3]):

                    # If the path is masked, skip it
                    if not mask[rx, tx, p]:
                        continue

                    # Add a comment to describe this path
                    r += f'# Path {p} from tx {tx} to rx {rx}' + os.linesep
                    # Vertices and intersected objects
                    vs = vertices[:, rx, tx, p].numpy()
                    objs = objects[:, rx, tx, p].numpy()

                    depth = 0
                    # First vertex is the source
                    r += f"v {src[0]:.8f} {src[1]:.8f} {src[2]:.8f}" + os.linesep
                    # Add intersection points
                    for v, o in zip(vs, objs):
                        # Skip if no intersection
                        if o == -1:
                            continue
                        r += f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}" + os.linesep
                        depth += 1
                    r += f"v {tgt[0]:.8f} {tgt[1]:.8f} {tgt[2]:.8f}" + os.linesep

                    # Add the connections
                    for i in range(1, depth + 2):
                        v0 = i + offset
                        v1 = i + offset + 1
                        r += f"l {v0} {v1}" + os.linesep

                    # Prepare for the next path
                    r += os.linesep
                    offset += depth + 2

        # Save the file
        # pylint: disable=unspecified-encoding
        with open(filename, 'w') as f:
            f.write(r)

    @property
    def a(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], torch.complex : Passband channel coefficients :math:`a_i` of each path as defined in :eq:`H_final`.
        """
        return self._a

    @a.setter
    def a(self, v):
        self._a = v

    @property
    def tau(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float : Propagation delay :math:`\tau_i` [s] of each path as defined in :eq:`H_final`.
        """
        return self._tau

    @tau.setter
    def tau(self, v):
        self._tau = v

    @property
    def theta_t(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float : Zenith  angles of departure [rad]
        """
        return self._theta_t

    @theta_t.setter
    def theta_t(self, v):
        self._theta_t = v

    @property
    def phi_t(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float : Azimuth angles of departure [rad]
        """
        return self._phi_t

    @phi_t.setter
    def phi_t(self, v):
        self._phi_t = v

    @property
    def theta_r(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float : Zenith angles of arrival [rad]
        """
        return self._theta_r

    @theta_r.setter
    def theta_r(self, v):
        self._theta_r = v

    @property
    def phi_r(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float : Azimuth angles of arrival [rad]
        """
        return self._phi_r

    @phi_r.setter
    def phi_r(self, v):
        self._phi_r = v

    @property
    def types(self):
        """
        [batch_size, max_num_paths], torch.int : Type of the paths:

        - 0 : LoS
        - 1 : Reflected
        - 2 : Diffracted
        - 3 : Scattered
        """
        return self._types

    @types.setter
    def types(self, v):
        self._types = v

    @property
    def sources(self):
        # pylint: disable=line-too-long
        """
        [num_sources, 3], torch.float : Sources from which rays (paths) are emitted
        """
        return self._sources

    @sources.setter
    def sources(self, v):
        self._sources = v

    @property
    def targets(self):
        # pylint: disable=line-too-long
        """
        [num_targets, 3], torch.float : Targets at which rays (paths) are received
        """
        return self._targets

    @targets.setter
    def targets(self, v):
        self._targets = v

    @property
    def normalize_delays(self):
        """
        bool : Set to `True` to normalize path delays such that the first path
        between any pair of antennas of a transmitter and receiver arrives at
        ``tau = 0``. Defaults to `True`.
        """
        return self._normalize_delays

    @normalize_delays.setter
    def normalize_delays(self, v):
        if v == self._normalize_delays:
            return

        if ~v and self._normalize_delays:
            self.tau += self._min_tau
        else:
            self.tau -= self._min_tau
        self._normalize_delays = v

    def apply_doppler(self, sampling_frequency, num_time_steps,
                      tx_velocities=(0., 0., 0.), rx_velocities=(0., 0., 0.)):
        # pylint: disable=line-too-long
        r"""
        Apply Doppler shifts corresponding to input transmitters and receivers
        velocities.

        This function replaces the last dimension of the tensor storing the
        paths coefficients ``a``, which stores the the temporal evolution of
        the channel, with a dimension of size ``num_time_steps`` computed
        according to the input velocities.

        Time evolution of the channel coefficient is simulated by computing the
        Doppler shift due to movements of the transmitter and receiver. If we denote by
        :math:`\mathbf{v}_{\text{T}}\in\mathbb{R}^3` and :math:`\mathbf{v}_{\text{R}}\in\mathbb{R}^3`
        the velocity vectors of the transmitter and receiver, respectively, the Doppler shifts are computed as

        .. math::

            f_{\text{T}, i} &= \frac{\hat{\mathbf{r}}(\theta_{\text{T},i}, \varphi_{\text{T},i})^\mathsf{T}\mathbf{v}_{\text{T}}}{\lambda}\qquad \text{[Hz]}\\
            f_{\text{R}, i} &= \frac{\hat{\mathbf{r}}(\theta_{\text{R},i}, \varphi_{\text{R},i})^\mathsf{T}\mathbf{v}_{\text{R}}}{\lambda}\qquad \text{[Hz]}

        for arbitrary path :math:`i`, where :math:`(\theta_{\text{T},i}, \varphi_{\text{T},i})` are the AoDs,
        :math:`(\theta_{\text{R},i}, \varphi_{\text{R},i})` are the AoAs, and :math:`\lambda` is the wavelength.
        This leads to the time-dependent path coefficient

        .. math ::

            a_i(t) = a_i e^{j2\pi(f_{\text{T}, i}+f_{\text{R}, i})t}.

        Note that this model is only valid as long as the AoDs, AoAs, and path delay do not change.

        When this function is called multiple times, it overwrites the previous
        time steps dimension.

        Input
        ------
        sampling_frequency : float
            Frequency [Hz] at which the channel impulse response is sampled

        num_time_steps : int
            Number of time steps.

        tx_velocities : [batch_size, num_tx, 3] or broadcastable, torch.float | `None`
            Velocity vectors :math:`(v_\text{x}, v_\text{y}, v_\text{z})` of all
            transmitters [m/s].
            Defaults to `[0,0,0]`.

        rx_velocities : [batch_size, num_tx, 3] or broadcastable, torch.float | `None`
            Velocity vectors :math:`(v_\text{x}, v_\text{y}, v_\text{z})` of all
            receivers [m/s].
            Defaults to `[0,0,0]`.
        """

        dtype = self._scene.dtype
        rdtype = dtype.real_dtype
        zeror = torch.zeros(0, dtype=rdtype)
        two_pi = 2. * PI.type(rdtype)

        tx_velocities = tx_velocities.type(rdtype)
        tx_velocities = expand_to_rank(tx_velocities, 3, 0)
        if tx_velocities.shape[2] != 3:
            raise ValueError("Last dimension of `tx_velocities` must equal 3")

        if rx_velocities is None:
            rx_velocities = [0., 0., 0.]
        rx_velocities = rx_velocities.type(rdtype)
        rx_velocities = expand_to_rank(rx_velocities, 3, 0)
        if rx_velocities.shape[2] != 3:
            raise ValueError("Last dimension of `rx_velocities` must equal 3")

        sampling_frequency = sampling_frequency.type(rdtype)
        if sampling_frequency <= 0.0:
            raise ValueError("The sampling frequency must be positive")

        num_time_steps = num_time_steps.type(torch.int32)
        if num_time_steps <= 0:
            msg = "The number of time samples must a positive integer"
            raise ValueError(msg)

        # Drop previous time step dimension, if any
        if torch.linalg.matrix_rank(self.a) == 7:
            self.a = self.a[..., 0]

        # [batch_size, num_rx, num_tx, max_num_paths, 3]
        # or
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 3]
        k_t = r_hat(self.theta_t, self.phi_t)
        k_r = r_hat(self.theta_r, self.phi_r)

        if self._scene.synthetic_array:
            # [batch_size, num_rx, 1, num_tx, 1, max_num_paths, 3]
            k_t = torch.unsqueeze(torch.unsqueeze(k_t, dim=2), dim=4)
            # [batch_size, num_rx, 1, num_tx, 1, max_num_paths, 3]
            k_r = torch.unsqueeze(torch.unsqueeze(k_r, dim=2), dim=4)

        # Expand rank of the speed vector for broadcasting with k_r
        # [batch_dim, 1, 1, num_tx, 1, 1, 3]
        tx_velocities = insert_dims(insert_dims(tx_velocities, 2, 1), 2, 4)
        # [batch_dim, num_rx, 1, 1, 1, 1, 3]
        rx_velocities = insert_dims(rx_velocities, 4, 1)

        # Generate time steps
        # [num_time_steps]
        ts = torch.arange(0, num_time_steps, dtype=rdtype)
        ts = ts / sampling_frequency

        # Compute the Doppler shift
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        tx_ds = two_pi * dot(tx_velocities, k_t) / self._scene.wavelength
        rx_ds = two_pi * dot(rx_velocities, k_r) / self._scene.wavelength
        ds = tx_ds + rx_ds
        # Expand for the time sample dimension
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 1]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 1]
        ds = torch.unsqueeze(ds, dim=-1)
        # Expand time steps for broadcasting
        # [1, 1, 1, 1, 1, 1, num_time_steps]
        ts = expand_to_rank(ts, torch.linalg.matrix_rank(ds), 0)
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths,
        #   num_time_steps]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, num_time_steps]
        ds = ds * ts
        exp_ds = torch.exp(torch.complex(zeror, ds))

        # Apply Doppler shift
        # Expand with time dimension
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 1]
        a = torch.unsqueeze(self.a, dim=-1)
        if self._scene.synthetic_array:
            # Broadcast is not supported by Torch for such high rank tensors.
            # We therefore do it manually
            # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths,
            #   num_time_steps]
            a = torch.tile(a, [1, 1, 1, 1, 1, 1, exp_ds.shape[6]])
        # [batch_dim, num_rx,  num_rx_ant, num_tx, num_tx_ant, max_num_paths,
        #   num_time_steps]
        a = a * exp_ds

        self.a = a

    @property
    def reverse_direction(self):
        r"""
        bool : If set to `True`, swaps receivers and transmitters
        """
        return self._reverse_direction

    @reverse_direction.setter
    def reverse_direction(self, v):

        if v == self._reverse_direction:
            return

        if torch.linalg.matrix_rank(self.a) == 6:
            self.a = torch.permute(self.a, (0, 3, 4, 1, 2, 5))
        else:
            self.a = torch.permute(self.a, (0, 3, 4, 1, 2, 5, 6))

        if self._scene.synthetic_array:
            self.tau = torch.permute(self.tau, (0, 2, 1, 3))
            self.theta_t = torch.permute(self.theta_t, (0, 2, 1, 3))
            self.phi_t = torch.permute(self.phi_t, (0, 2, 1, 3))
            self.theta_r = torch.permute(self.theta_r, (0, 2, 1, 3))
            self.phi_r = torch.permute(self.phi_r, (0, 2, 1, 3))
        else:
            self.tau = torch.permute(self.tau, (0, 3, 4, 1, 2, 5))
            self.theta_t = torch.permute(self.theta_t, (0, 3, 4, 1, 2, 5))
            self.phi_t = torch.permute(self.phi_t, (0, 3, 4, 1, 2, 5))
            self.theta_r = torch.permute(self.theta_r, (0, 3, 4, 1, 2, 5))
            self.phi_r = torch.permute(self.phi_r, (0, 3, 4, 1, 2, 5))

        self._reverse_direction = v

    def cir(self, los=True, reflection=True, diffraction=True, scattering=True):
        # pylint: disable=line-too-long
        r"""
        Returns the baseband equivalent channel impulse response :eq:`h_b`
        which can be used for link simulations by other torchrf components.

        The baseband equivalent channel coefficients :math:`a^{\text{b}}_{i}`
        are computed as :

        .. math::
            a^{\text{b}}_{i} = a_{i} e^{-j2 \pi f \tau_{i}}

        where :math:`i` is the index of an arbitrary path, :math:`a_{i}`
        is the passband path coefficient (:attr:`~torchrf.rt.Paths.a`),
        :math:`\tau_{i}` is the path delay (:attr:`~torchrf.rt.Paths.tau`),
        and :math:`f` is the carrier frequency.

        Note: For the paths of a given type to be returned (LoS, reflection, etc.), they
        must have been previously computed by :meth:`~torchrf.rt.Scene.compute_paths`, i.e.,
        the corresponding flags must have been set to `True`.

        Input
        ------
        los : bool
            If set to `False`, LoS paths are not returned.
            Defaults to `True`.

        reflection : bool
            If set to `False`, specular paths are not returned.
            Defaults to `True`.

        diffraction : bool
            If set to `False`, diffracted paths are not returned.
            Defaults to `True`.

        scattering : bool
            If set to `False`, scattered paths are not returned.
            Defaults to `True`.

        Output
        -------
        a : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], torch.complex
            Path coefficients

        tau : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], torch.float
            Path delays
        """

        # Select only the desired effects
        types = self.types[0]
        # [max_num_paths]
        selection_mask = torch.full(list(types.size()), False)
        if los:
            selection_mask = torch.logical_or(selection_mask,
                                              types == Paths.LOS)
        if reflection:
            selection_mask = torch.logical_or(selection_mask,
                                              types == Paths.SPECULAR)
        if diffraction:
            selection_mask = torch.logical_or(selection_mask,
                                              types == Paths.DIFFRACTED)
        if scattering:
            selection_mask = torch.logical_or(selection_mask,
                                              types == Paths.SCATTERED)

        # Extract selected paths
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths,
        #   num_time_steps]
        a = self.a[...,  torch.where(selection_mask)[0], :]
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        #   or [batch_size, num_rx, num_tx, max_num_paths]
        tau = self.tau[..., torch.where(selection_mask)[0]]

        # Compute baseband CIR
        # [batch_size, num_rx, 1/num_rx_ant, num_tx, 1/num_tx_ant,
        #   max_num_paths, num_time_steps, 1]
        if self._scene.synthetic_array:
            tau_ = torch.unsqueeze(tau, dim=2)
            tau_ = torch.unsqueeze(tau_, dim=4)
        else:
            tau_ = tau
        tau_ = torch.unsqueeze(tau_, dim=-1)
        phase = torch.complex(torch.zeros_like(tau_),
                              -2 * PI * self._scene.frequency * tau_)
        a = a * torch.exp(phase)

        return a, tau

    #######################################################
    # Internal methods and properties
    #######################################################

    @property
    def mask(self):
        # pylint: disable=line-too-long
        """
        [num_targets, num_sources, max_num_paths], torch.bool : Mask indicating if a path is valid
        """
        return self._mask

    @mask.setter
    def mask(self, v):
        self._mask = v

    @property
    def vertices(self):
        # pylint: disable=line-too-long
        """
        [max_depth, num_targets, num_sources, max_num_paths, 3], torch.float : Positions of intersection points.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, v):
        self._vertices = v

    @property
    def objects(self):
        # pylint: disable=line-too-long
        """
        [max_depth, num_targets, num_sources, max_num_paths], torch.int : Indices of the intersected scene objects
        or wedges. Paths with depth lower than ``max_depth`` are padded with `-1`.
        """
        return self._objects

    @objects.setter
    def objects(self, v):
        self._objects = v

    def merge(self, more_paths):
        r"""
        Merge ``more_paths`` with the current paths and returns the so-obtained
        instance. `self` is not updated.

        Input
        -----
        more_paths : :class:`~torchrf.rt.Paths`
            First set of paths to merge
        """

        dtype = self._scene.dtype
        more_vertices = more_paths.vertices
        more_objects = more_paths.objects
        more_types = more_paths.types

        # The paths to merge must have the same number of sources and targets
        assert more_paths.targets.shape[0] == self.targets.shape[0], \
            "Paths to merge must have same number of targets"
        assert more_paths.sources.shape[0] == self.sources.shape[0], \
            "Paths to merge must have same number of targets"

        # Pad the paths with the lowest depth
        padding = self.vertices.shape[0] - more_vertices.shape[0]
        if padding > 0:
            more_vertices = F.pad(more_vertices, (0, 0, 0, 0, 0, 0, 0, 0, 0, padding), value=0.0)
            more_objects = F.pad(more_objects, (0, 0, 0, 0, 0, 0, 0, padding), value=-1)
        elif padding < 0:

            padding = -padding
            if self.vertices.shape[-2] > 0:
                self.vertices = F.pad(self.vertices, (0, 0, 0, 0, 0, 0, 0, 0, padding, 0), value=0)
            else:
                self.vertices = torch.zeros(self.vertices.shape[0] + padding,
                                            self.vertices.shape[1],
                                            self.vertices.shape[2],
                                            self.vertices.shape[3],
                                            self.vertices.shape[4], dtype=self.vertices.dtype)
            if self.objects.shape[-1] > 0:
                self.objects = F.pad(self.objects, (0, 0, 0, 0, 0, 0, 0, padding), value=-1)
            else:
                self.objects = -torch.ones(self.objects.shape[0] + padding,
                                           self.objects.shape[1],
                                           self.objects.shape[2],
                                           self.objects.shape[3], dtype=self.objects.dtype)


        # Merge types
        if rank(self.types) == 0:
            merged_types = torch.full((self.vertices.shape[3],), self.types)
        else:
            merged_types = self.types

        if rank(torch.tensor(more_types)) == 0:
            more_types = torch.full((more_vertices.shape[3],), more_types)

        self.types = torch.cat((merged_types, more_types), dim=0)

        # Concatenate all
        self.a = torch.cat([self.a, more_paths.a], dim=2)
        self.tau = torch.cat([self.tau, more_paths.tau], dim=2)
        self.theta_t = torch.cat([self.theta_t, more_paths.theta_t], dim=2)
        self.phi_t = torch.cat([self.phi_t, more_paths.phi_t], dim=2)
        self.theta_r = torch.cat([self.theta_r, more_paths.theta_r], dim=2)
        self.phi_r = torch.cat([self.phi_r, more_paths.phi_r], dim=2)
        self.mask = torch.cat([self.mask, more_paths.mask], dim=2)
        self.vertices = torch.cat([self.vertices, more_vertices], dim=3)
        self.objects = torch.cat([self.objects, more_objects], dim=3)

        return self

    def finalize(self):
        """
        This function must be call to finalize the creation of the paths.
        This function:

        - Flags the LoS paths

        - Computes the smallest delay for delay normalization
        """

        self.set_los_path_type()

        tau = self.tau
        if self._scene.synthetic_array:
            min_tau = torch.amin(torch.abs(tau), dim=2, keepdim=True)
        else:
            min_tau = torch.amin(torch.abs(tau), dim=(1, 3, 4), keepdim=True)
        self._min_tau = min_tau

        # Add dummy-dimension for batch_size
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.a = torch.unsqueeze(self.a, dim=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.tau = torch.unsqueeze(self.tau, dim=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.theta_t = torch.unsqueeze(self.theta_t, dim=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.phi_t = torch.unsqueeze(self.phi_t, dim=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.theta_r = torch.unsqueeze(self.theta_r, dim=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.phi_r = torch.unsqueeze(self.phi_r, dim=0)
        # [1, max_num_paths]
        self.types = torch.unsqueeze(self.types, dim=0)

        # Add the time steps dimension
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 1]
        self.a = torch.unsqueeze(self.a, dim=-1)

        # Normalize delays
        self.normalize_delays = True

    def set_los_path_type(self):
        """
        Flags paths that do not hit any objects to as LoS ones.
        """

        if self.objects.shape[3] > 0:
            # [num_targets, num_sources, num_paths]
            los_path = (self.objects == -1).all(0)
            # [num_targets, num_sources, num_paths]
            los_path = torch.logical_and(los_path, self.mask)
            # [num_paths]
            los_path = los_path.any(0).any(0)
            # [[1]]
            los_path_index = torch.where(los_path)[0]
            if los_path_index.shape[0] > 0:
                self.types = scatter_nd_update(self.types, los_path_index, torch.tensor([Paths.LOS]))
                # tf.tensor_scatter_nd_update(self.types,
                #                                         los_path_index,
                #                                         [Paths.LOS])
