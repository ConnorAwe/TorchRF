#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Ray tracing algorithm that uses the image method to compute all pure reflection
paths.
"""

import mitsuba as mi
import drjit as dr
import torch
import torch.nn.functional as F
from torchrf.constants import SPEED_OF_LIGHT, PI
from torchrf.utils.tensors import expand_to_rank, insert_dims, split_dim
from .paths import Paths
from .utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec, \
    normalize, moller_trumbore, component_transform, mi_to_torch_tensor, \
    compute_field_unit_vectors, reflection_coefficient, fibonacci_lattice, \
    cot, cross, rotation_matrix, acos_diff, scatter_nd_update, cast, \
    divide_no_nan, random_uniform, reshape, size, fresnel_sin, fresnel_cos, \
    gather_nd, normalize, band_part, gather_nd_torch, gather_batch, matvec
from .solver_base import SolverBase
from .scattering_pattern import ScatteringPattern


class PathsTmpData:
    r"""
    Class used to temporarily store values for paths calculation.
    """

    def __init__(self, sources, targets, dtype):

        self.sources = sources
        self.targets = targets
        self.dtype = dtype
        rdtype = dtype.to_real()
        num_sources = sources.shape[0]
        num_targets = targets.shape[0]

        # [max_depth, num_targets, num_sources, max_num_paths, 3] or
        # [max_depth, num_targets, num_sources, max_num_paths, 2, 3], torch.float
        #     Reflected or scattered paths: Normals to the primitives at the
        #     intersection points.
        #     Diffracted paths: Normals to the two primitives forming the wedge.
        self.normals = None

        # [max_depth + 1, num_targets, num_sources, max_num_paths, 3], torch.float
        #   Direction of arrivals.
        #   The last item (k_i[max_depth]) correspond to the direction of
        #   arrival at the target. Therefore, k_i is a tensor of length
        #   `max_depth + 1`, where `max_depth` is the number of maximum
        #   interaction (which could be zero if only LoS is requested).
        self.k_i = None

        # [max_depth, num_targets, num_sources, max_num_paths, 3], torch.float
        #   Direction of departures at interaction points.
        #   We do not need the direction of departure at the source, as it
        #   is the same as k_i[0].
        self.k_r = None

        # [max_depth+1, num_targets, num_sources, max_num_paths] or
        # [max_depth, num_targets, num_sources, max_num_paths] for scattering,
        # torch.float
        #     Lengths in meters of the paths segments
        self.total_distance = None

        # [num_targets, num_sources, max_num_paths, 2, 2] or
        # [num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 2, 2],
        # torch.complex
        #     Channel transition matrix
        # These are initialized to emtpy tensors to handle cases where no
        # paths are found
        self.mat_t = torch.zeros([num_targets, num_sources, 0, 2, 2], dtype=dtype)

        # [num_targets, num_sources, max_num_paths, 3], torch.float
        #   Direction of departure. This vector is normalized and pointing
        #   awat from the radio device.
        # These are initialized to emtpy tensors to handle cases where no
        # paths are found
        self.k_tx = torch.zeros([num_targets, num_sources, 0, 3],
                                dtype=rdtype)

        # [num_targets, num_sources, max_num_paths, 3], torch.float
        #   Direction of arrival. This vector is normalized and pointing
        #   awat from the radio device.
        # These are initialized to emtpy tensors to handle cases where no
        # paths are found
        self.k_rx = torch.zeros([num_targets, num_sources, 0, 3],
                                dtype=rdtype)

        # [max_depth, num_targets, num_sources, max_num_paths], torch.bool
        #   This parameter is specific to scattering.
        #   For scattering, every path prefix is a potential final path.
        #   This tensor is a mask which indicates for every path prefix if it
        #   is a valid path.
        self.scat_prefix_mask = None

        # [max_depth, num_targets, num_sources, max_num_paths, 2, 2], torch.complex
        #   This parameter is specific to scattering.
        #   For scattering, every path prefix is a potential final path.
        #   This tensor stores the transition matrices for reflection
        #   corresponding to every interaction up to the scattering point.
        self.scat_prefix_mat_t = None

        # [max_depth, num_targets, num_sources, max_num_paths, 3], torch.float
        #   For every intersection point between the paths and the scene,
        #   gives the direction of the scattered ray, i.e., points towards the
        #   targets.
        self.scat_prefix_k_s = None

        # [num_targets, num_sources, max_num_paths]
        #   This parameter is specific to scattering.
        #   Stores the index of the last hit object for retreiving the
        #   scattering properties of the objects
        self.scat_last_objects = None

        # [num_targets, num_sources, max_num_paths, 3]
        #   This parameter is specific to scattering.
        #   Stores the incoming vector for the last interaction, i.e., the
        #   one that scatters the field
        self.scat_last_k_i = None

        # [num_targets, num_sources, max_num_paths, 3]
        #   This parameter is specific to scattering.
        #   Stores the outgoing vector for the last interaction, i.e., the
        #   direction of the scattered ray.
        self.scat_k_s = None

        # [num_targets, num_sources, max_num_paths, 3]
        #   This parameter is specific to scattering.
        #   Stores the normals to the last interaction point, i.e., the
        #   scattering point
        self.scat_last_normals = None

        # [num_targets, num_sources, max_num_paths]
        #   This parameter is specific to scattering.
        #   Stores the distance from the sources to the scattering points.
        self.scat_src_2_last_int_dist = None

        # [num_targets, num_sources, max_num_paths]
        #   This parameter is specific to scattering.
        #   Stores the distance from the scattering points to the targets.
        self.scat_2_target_dist = None

    def merge_ktx_krx_matt(self, other):
        r"""
        Returns a new structure with only the fields `k_tx`, `k_rx`, and
        `mat_t` set to the concatenation of `self` and
        `other_paths_tmp_data`.
        """

        new_obj = PathsTmpData(self.sources, self.targets, self.dtype)
        if self.mat_t.shape[2] == 0:
            new_obj.mat_t = other.mat_t
        else:
            new_obj.mat_t = torch.cat([self.mat_t, other.mat_t], dim=2)

        if self.k_tx.shape[2] == 0:
            new_obj.k_tx = other.k_tx
        else:
            new_obj.k_tx = torch.cat([self.k_tx, other.k_tx], dim=2)

        if self.k_rx.shape[2] == 0:
            new_obj.k_rx = other.k_rx
        else:
            new_obj.k_rx = torch.cat([self.k_rx, other.k_rx], dim=2)

        return new_obj


class SolverPaths(SolverBase):
    # pylint: disable=line-too-long
    r"""SolverPaths(scene, solver=None, dtype=torch.complex64)

    Generates propagation paths consisting of the line-of-sight (LoS) paths,
    specular, and diffracted paths for the currently loaded scene.

    The main inputs of the solver are:

    * A set of sources, from which rays are emitted.

    * A set of targets, at which rays are received.

    * A maximum depth, corresponding to the maximum number of reflections. A
    depth of zero corresponds to LoS.

    Generation of paths is carried-out for every link, i.e., for every pair of
    source and target.

    The genration of specular paths consists in three steps:

    1. A list of candidate paths is generated. A candidate consists in a
    sequence of primitives on which a ray emitted by a source sequentially
    reflects until it reaches a target.

    2. The image method is applied to every candidates (in parallel) to discard
    candidates that do not correspond to valid paths, either because they
    are obstructed by another object in the scene, or because a reflection on
    one of the primitive in the sequence is impossible (reflection point outside
    of the primitive).

    3. For the valid paths, Fresnel coefficients for reflections are computed,
    considering the materials of the intersected objects, to compute transfer
    matrices for every paths.

    For diffracted paths, after step 1.:

    2. The wedges of primitives in LoS are selected, i.e., primitives for which
    a direct connection with the sources was found at step 1

    3. The intersection point of the diffracted path on the wedge is computed.
    This is the point that minimizes the total length of the paths.
    Paths for which the diffraction point does not belong to the finite wedge
    are discarded.

    4. Obstruction test: Paths that are blocked are discarded.

    5. The transmition matrices are computed, as well as the delays and angles
    of arrival and departure.

    The output of the solver consists in, for every valid path that was found:

    * A transfer matrix, which is a 2x2 complex-valued matrix that describes the
    linear transformation incurred by the emitted field. The two dimensions
    correspond to the two polarization components (S and P).

    * A delay

    * Azimuth and zenith angles of arrival

    * Azimuth and zenith angles of departure

    Concerning the first step, two search methods are available for the
    listing of candidates:

    * Exhaustive search, which lists all possible combinations of primitives up
    to the requested maximum depth. This method is deterministic and ensures
    that all paths are found. However, its complexity increases exponentially
    with the number of primitives and with the maximum depth. Therefore, it
    only works for scenes of low complexity and/or for small depth values.

    * Fibonacci sampling, which find candidates by shooting and bouncing rays,
    and such that initial directions of rays shot from the sources are arranged
    in a Fibonacci lattice on the unit sphere. At every intersection with a
    primitive, the rays are bounced assuming perfectly specular reflections
    until the maximum depth is reached. The intersected primitives makes the
    candidate. This method can be applied to very large scenes. However, there
    is no guarantee that all possible paths are found.

    Note: Only triangle mesh are supported.

    Parameters
    -----------
    scene : :class:`~torchrf.rt.Scene`
        torchrf RT scene

    solver : :class:`~torchrf.rt.SolverBase` | None
        Another solver from which to re-use some structures to avoid useless
        compute and memory use

    dtype : torch.complex64 | torch.complex128
        Datatype for all computations, inputs, and outputs.
        Defaults to `torch.complex64`.

    Input
    ------
    max_depth : int
        Maximum depth (i.e., number of interaction with objects in the scene)
        allowed for tracing the paths.

    sources : [num_sources, 3], torch.float
        Coordinates of the sources.

    targets : [num_targets, 3], torch.float
        Coordinates of the targets.

    method : str ("exhaustive"|"fibonacci")
        Method to be used to list candidate paths.
        The "exhaustive" method tests all possible combination of primitives as
        paths. This method is not compatible with scattering.
        The "fibonacci" method uses a shoot-and-bounce approach to find
        candidate chains of primitives. Intial rays direction are arranged
        in a Fibonacci lattice on the unit sphere. This method can be
        applied to very large scenes. However, there is no guarantee that
        all possible paths are found.

    num_samples: int
        Number of random rays to trace in order to generate candidates.
        A large sample count may exhaust GPU memory.

    los : bool
        If set to `True`, then the LoS paths are computed.

    reflection : bool
        If set to `True`, then the reflected paths are computed.

    diffraction : bool
        If set to `True`, then the diffracted paths are computed.

    scattering : bool
        if set to `True`, then the scattered paths are computed.
        Only works with the Fibonacci method.

    scat_keep_prob : float
        Probability with which to keep scattered paths.
        This is helpful to reduce the number of scattered paths computed,
        which might be prohibitively high in some setup.
        Must be in the range (0,1).

    edge_diffraction : bool
        If set to `False`, only diffraction on wedges, i.e., edges that
        connect two primitives, is considered.

    Output
    -------
    paths : Paths
        The computed paths.
    """

    def __call__(self, max_depth, method, num_samples, los, reflection,
                 diffraction, scattering, scat_keep_prob, edge_diffraction):

        scat_keep_prob = cast(scat_keep_prob, self._rdtype)
        # Disable scattering if the probability of keeping a path is 0
        scattering = torch.logical_and(torch.tensor(scattering),
                                       torch.gt(scat_keep_prob, torch.zeros_like(scat_keep_prob)))

        # If reflection and scattering are disabled, no need for a max_depth
        # higher than 1.
        # This clipping can save some compute for the shoot-and-bounce
        if (not reflection) and (not scattering):
            max_depth = torch.minimum(max_depth, 1)

        # Rotation matrices corresponding to the orientations of the radio
        # devices
        # rx_rot_mat : [num_rx, 3, 3]
        # tx_rot_mat : [num_tx, 3, 3]
        rx_rot_mat, tx_rot_mat = self._get_tx_rx_rotation_matrices()
        #################################################
        # Prepares the sources (from which rays are shot)
        # and targets (which capture the rays)
        #################################################

        if not self._scene.synthetic_array:
            # Relative positions of the antennas of the transmitters and
            # receivers
            # rx_rel_ant_pos: [num_rx, rx_array_size, 3], torch.float
            #     Relative positions of the receivers antennas
            # tx_rel_ant_pos: [num_tx, rx_array_size, 3], torch.float
            #     Relative positions of the transmitters antennas
            rx_rel_ant_pos, tx_rel_ant_pos = \
                self._get_antennas_relative_positions(rx_rot_mat, tx_rot_mat)

        # Number of receive antennas (not counting for dual polarization)
        tx_array_size = self._scene.tx_array.array_size
        # Number of transmit antennas (not counting for dual polarization)
        rx_array_size = self._scene.rx_array.array_size

        # Transmitters and receivers positions
        # [num_tx, 3]
        tx_pos = [tx.position for tx in self._scene.transmitters.values()]
        tx_pos = torch.stack(tx_pos, dim=0)
        # [num_rx, 3]
        rx_pos = [rx.position for rx in self._scene.receivers.values()]
        rx_pos = torch.stack(rx_pos, dim=0)

        if self._scene.synthetic_array:
            # With synthetic arrays, each radio device corresponds to a single
            # endpoint (source or target)
            # [num_sources = num_tx, 3]
            sources = tx_pos
            # [num_targets = num_rx, 3]
            targets = rx_pos
        else:
            # [num_tx, tx_array_size, 3]
            sources = torch.unsqueeze(tx_pos, dim=1) + tx_rel_ant_pos
            # [num_sources = num_tx*tx_array_size, 3]
            sources = reshape(sources, [-1, 3])
            # [num_rx, rx_array_size, 3]
            targets = torch.unsqueeze(rx_pos, dim=1) + rx_rel_ant_pos
            # [num_targets = num_rx*rx_array_size, 3]
            targets = reshape(targets, [-1, 3])

        #################################################
        # Extract the material properties of the scene
        #################################################

        # Returns: relative_permittivities, denoted by `etas`,
        # scattering_coefficients, xpd_coefficients,
        # alpha_r, alpha_i and lambda_
        object_properties = self._build_scene_object_properties_tensors()
        etas = object_properties[0]
        scattering_coefficient = object_properties[1]
        xpd_coefficient = object_properties[2]
        alpha_r = object_properties[3]
        alpha_i = object_properties[4]
        lambda_ = object_properties[5]

        ##############################################
        # Generate candidate paths
        ##############################################

        # Candidate paths are generated according to the specified `method`.
        if method == 'exhaustive':
            if scattering:
                msg = "The exhaustive method is not compatible with scattering"
                raise ValueError(msg)
            # List all possible sequences of primitives with length up to
            # ``max_depth``
            # candidates: [max_depth, num_samples], int
            #     All possible candidate paths with depth up to ``max_depth``.
            # los_candidates: [num_samples], int
            #     Primitives in LoS. For the exhaustive method, this is the
            #     list of all the primitives in the scene.
            candidates, los_prim = self._list_candidates_exhaustive(max_depth,
                                                                    los, reflection)
        elif method == 'fibonacci':
            # Sample sequences of primitives using shoot-and-bounce
            # with length up to ``max_depth`` and by arranging the initial
            # rays direction in a Fibonacci lattice on the unit sphere.
            # candidates: [max_depth, num paths], int
            #     All unique candidate paths found, with depth up to
            #       ``max_depth``.
            # los_candidates: [num_samples], int
            #     Candidate primitives found in LoS.
            # candidates_scat : [max_depth, num_sources, num_paths_per_source]
            #       Sequence of primitives hit at `hit_points`.
            # hit_points : [max_depth, num_sources, num_paths_per_source, 3]
            #     Coordinates of the intersection points.
            output = self._list_candidates_fibonacci(max_depth,
                                                     sources, num_samples, los, reflection,
                                                     scattering)
            candidates = output[0]
            los_prim = output[1]
            candidates_scat = output[2]
            hit_points = output[3]


        else:
            raise ValueError(f"Unknown method '{method}'")

        # Create empty paths object
        all_paths = Paths(sources=sources,
                          targets=targets,
                          scene=self._scene)
        # Create empty objects for storing tensors that are required to compute
        # paths, but that will not be returned to the user
        all_paths_tmp = PathsTmpData(sources, targets, self._dtype)
        ##############################################
        # LoS and Specular paths
        ##############################################
        if los or reflection:
            spec_paths = Paths(sources=sources, targets=targets,
                               types=Paths.SPECULAR, scene=self._scene)
            spec_paths_tmp = PathsTmpData(sources, targets, self._dtype)

            # Using the image method, computes the non-obstructed specular paths
            # interacting with the ``candidates`` primitives
            self._spec_image_method(candidates, spec_paths, spec_paths_tmp)

            # Compute paths length, delays, angles and directions of arrivals
            # and departures for the specular paths
            spec_paths, spec_paths_tmp = \
                self._compute_directions_distances_delays_angles(spec_paths,
                                                                 spec_paths_tmp, False)
            # Compute the EM transition matrices
            spec_paths, spec_paths_tmp = \
                self._spec_transition_matrices(etas,
                                               scattering_coefficient,
                                               spec_paths, spec_paths_tmp,
                                               False)

            all_paths = all_paths.merge(spec_paths)
            all_paths_tmp = all_paths_tmp.merge_ktx_krx_matt(spec_paths_tmp)
            # For tests
            all_paths.spec_tmp = spec_paths_tmp

        ############################################
        # Diffracted paths
        ############################################
        diff_paths = Paths(sources=sources, targets=targets,
                           scene=self._scene, types=Paths.DIFFRACTED)
        diff_paths_tmp = PathsTmpData(sources, targets, self._dtype)

        if (los_prim is not None) and diffraction:

            # Get the candidate wedges for diffraction
            # Note: Only one-order diffraction is supported. Therefore, we
            # restrict the candidate wedges to the ones of primitives in
            # line-of-sight with the transmitter
            # candidate_wedges : [num_candidate_wedges], int
            #     Candidate wedges indices
            diff_wedges_indices = self._wedges_from_primitives(los_prim,
                                                               edge_diffraction)

            # Discard paths for which at least one of the transmitter or
            # receiver is inside the wedge.
            # diff_wedges_indices : [num_targets, num_sources, max_num_paths]
            #   Indices of the intersected wedges
            diff_wedges_indices = self._discard_obstructing_wedges_and_corners(
                diff_wedges_indices, targets,
                sources)

            # Compute the intersection points with the wedges, and discard paths
            # for which the intersection point is not on the finite wedge.
            # diff_wedges_indices : [num_targets, num_sources, max_num_paths]
            #   Indices of the intersected wedges
            # diff_vertices : [num_targets, num_sources, max_num_paths, 3]
            #   Position of the intersection point on the wedges
            diff_wedges_indices, diff_vertices = \
                self._compute_diffraction_points(targets, sources,
                                                 diff_wedges_indices)

            # Discard obstructed diffracted paths
            # Only check for wedge visibility if there is at least one candidate
            # diffracted path
            if diff_wedges_indices.shape[2] > 0:  # Number of diff. paths > 0
                # Discard obstructed paths
                diff_wedges_indices, diff_vertices = \
                    self._check_wedges_visibility(targets, sources,
                                                  diff_wedges_indices,
                                                  diff_vertices)

            diff_paths.objects = torch.unsqueeze(diff_wedges_indices, dim=0)
            diff_paths.vertices = torch.unsqueeze(diff_vertices, dim=0)

            # Select only the valid paths
            diff_paths = self._gather_valid_diff_paths(diff_paths)

            # Computes paths length, delays, angles and directions of arrivals
            # and departures for the specular paths
            diff_paths, diff_paths_tmp = \
                self._compute_directions_distances_delays_angles(diff_paths,
                                                                 diff_paths_tmp, False)

            # Compute the transition matrices
            diff_paths = self._compute_diffraction_transition_matrices(etas,
                                                                       scattering_coefficient,
                                                                       diff_paths, diff_paths_tmp)
            all_paths = all_paths.merge(diff_paths)
            all_paths_tmp = all_paths_tmp.merge_ktx_krx_matt(diff_paths_tmp)
            # For tests
            all_paths.diff_tmp = diff_paths_tmp

        ############################################
        # Scattered paths
        ############################################
        if scattering and candidates_scat.shape[0] > 0:
            scat_paths, scat_paths_tmp = \
                self._scat_test_rx_blockage(targets, sources, candidates_scat,
                                            hit_points)
            scat_paths, scat_paths_tmp = \
                self._compute_directions_distances_delays_angles(scat_paths,
                                                                 scat_paths_tmp,
                                                                 True)

            scat_paths, scat_paths_tmp = \
                self._scat_discard_crossing_paths(scat_paths, scat_paths_tmp,
                                                  scat_keep_prob)

            # Compute transition matrices up to the scattering point
            scat_paths, scat_paths_tmp = \
                self._spec_transition_matrices(etas, scattering_coefficient,
                                               scat_paths, scat_paths_tmp,
                                               True)

            # Extract the valid prefixes as paths
            scat_paths, scat_paths_tmp = self._scat_prefixes_2_paths(scat_paths,
                                                                     scat_paths_tmp)

            all_paths = all_paths.merge(scat_paths)
            all_paths_tmp = all_paths_tmp.merge_ktx_krx_matt(scat_paths_tmp)
            # Add all fields specific to scattering
            all_paths_tmp.scat_last_objects = scat_paths_tmp.scat_last_objects
            all_paths_tmp.scat_last_k_i = scat_paths_tmp.scat_last_k_i
            all_paths_tmp.scat_k_s = scat_paths_tmp.scat_k_s
            all_paths_tmp.scat_last_normals = scat_paths_tmp.scat_last_normals
            all_paths_tmp.scat_src_2_last_int_dist \
                = scat_paths_tmp.scat_src_2_last_int_dist
            all_paths_tmp.scat_2_target_dist = scat_paths_tmp.scat_2_target_dist
            # For tests
            all_paths.scat_tmp = scat_paths_tmp

        #################################################
        # Splitting the sources (targets) dimension into
        # transmitters (receivers) and antennas, or
        # applying the synthetic arrays
        #################################################

        # If not using synthetic array, then the paths for the different
        # antenna elements were generated and reshaping is needed.
        # Otherwise, expand with the antenna dimensions.
        if self._scene.synthetic_array:
            # [num_rx, num_tx, 2, 2]
            mat_t = all_paths_tmp.mat_t
            # [num_rx, 1, num_tx, 1, max_num_paths, 2, 2]
            mat_t = torch.unsqueeze(torch.unsqueeze(mat_t, dim=1), dim=3)
            all_paths_tmp.mat_t = mat_t
        else:
            num_rx = len(self._scene.receivers)
            num_tx = len(self._scene.transmitters)
            max_num_paths = all_paths.vertices.shape[3]
            batch_dims = [num_rx, rx_array_size, num_tx, tx_array_size,
                          max_num_paths]
            # [num_rx, tx_array_size, num_tx, tx_array_size, max_num_paths]
            all_paths.tau = reshape(all_paths.tau, batch_dims)
            all_paths.theta_t = reshape(all_paths.theta_t, batch_dims)
            all_paths.phi_t = reshape(all_paths.phi_t, batch_dims)
            all_paths.theta_r = reshape(all_paths.theta_r, batch_dims)
            all_paths.phi_r = reshape(all_paths.phi_r, batch_dims)
            # [num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 2,2]
            all_paths_tmp.mat_t = reshape(all_paths_tmp.mat_t,
                                          batch_dims + [2, 2])
            # [num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 3]
            all_paths_tmp.k_tx = reshape(all_paths_tmp.k_tx, batch_dims + [3])
            all_paths_tmp.k_rx = reshape(all_paths_tmp.k_rx, batch_dims + [3])

        ####################################################
        # Compute the channel coefficients
        ####################################################
        all_paths = self._compute_paths_coefficients(rx_rot_mat,
                                                     tx_rot_mat,
                                                     all_paths,
                                                     all_paths_tmp,
                                                     num_samples,
                                                     scattering_coefficient,
                                                     xpd_coefficient,
                                                     etas, alpha_r, alpha_i,
                                                     lambda_, scat_keep_prob)

        # If using synthetic array, adds the antenna dimentions by applying
        # synthetic phase shifts
        if self._scene.synthetic_array:
            all_paths = self._apply_synthetic_array(rx_rot_mat, tx_rot_mat,
                                                    all_paths, all_paths_tmp)

        ##################################################
        # If not using synthetic arrays, tile the AoAs,
        # AoDs, and delays to handle dual-polarization
        ##################################################
        if not self._scene.synthetic_array:
            num_rx_patterns = len(self._scene.rx_array.antenna.patterns)
            num_tx_patterns = len(self._scene.tx_array.antenna.patterns)
            # [num_rx, 1,rx_array_size, num_tx, 1,tx_array_size, max_num_paths]
            tau = torch.unsqueeze(torch.unsqueeze(all_paths.tau, dim=2),
                                  dim=5)
            theta_t = torch.unsqueeze(torch.unsqueeze(all_paths.theta_t, dim=2),
                                      dim=5)
            phi_t = torch.unsqueeze(torch.unsqueeze(all_paths.phi_t, dim=2),
                                    dim=5)
            theta_r = torch.unsqueeze(torch.unsqueeze(all_paths.theta_r, dim=2),
                                      dim=5)
            phi_r = torch.unsqueeze(torch.unsqueeze(all_paths.phi_r, dim=2),
                                    dim=5)
            # [num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns,
            #   tx_array_size, max_num_paths]
            tau = torch.repeat(tau, [1, num_rx_patterns, 1, 1,
                                     num_tx_patterns, 1, 1])
            theta_t = torch.repeat(theta_t, [1, num_rx_patterns, 1, 1,
                                             num_tx_patterns, 1, 1])
            phi_t = torch.repeat(phi_t, [1, num_rx_patterns, 1, 1,
                                         num_tx_patterns, 1, 1])
            theta_r = torch.repeat(theta_r, [1, num_rx_patterns, 1, 1,
                                             num_tx_patterns, 1, 1])
            phi_r = torch.repeat(phi_r, [1, num_rx_patterns, 1, 1,
                                         num_tx_patterns, 1, 1])
            # [num_rx, num_rx_ant = num_rx_patterns*num_rx_ant,
            #   ... num_tx, num_tx_ant = num_tx_patterns*tx_array_size,
            #   ... max_num_paths]
            # all_paths.tau = flatten_dims(flatten_dims(tau, 2, 1), 2, 3)
            # all_paths.theta_t = flatten_dims(flatten_dims(theta_t, 2, 1), 2, 3)
            # all_paths.phi_t = flatten_dims(flatten_dims(phi_t, 2, 1), 2, 3)
            # all_paths.theta_r = flatten_dims(flatten_dims(theta_r, 2, 1), 2, 3)
            # all_paths.phi_r = flatten_dims(flatten_dims(phi_r, 2, 1), 2, 3)
            all_paths.tau = tau.flatten(1, 2).flatten(3, 4)
            all_paths.theta_t = theta_t.flatten(1, 2).flatten(3, 4)
            all_paths.phi_t = phi_t.flatten(1, 2).flatten(3, 4)
            all_paths.theta_r = theta_r.flatten(1, 2).flatten(3, 4)
            all_paths.phi_r = phi_r.flatten(1, 2).flatten(3, 4)

        # For test
        all_paths.all_tmp = all_paths_tmp

        return all_paths

    ##################################################################
    # Methods for finding candiate primitives and edges for reflected
    # and diffracted paths
    ##################################################################

    def _list_candidates_exhaustive(self, max_depth, los, reflection):
        r"""
        Generate all possible candidate paths made of reflections only and the
        LoS.

        The number of candidate paths equals

            num_triangles**max_depth + 1

        where the additional path (+1) is the LoS.

        This can easily exhaust GPU memory if the number of triangles in the
        scene or the `max_depth` are too large.

        Input
        ------
        max_depth: int
            Maximum number of reflections.
            Set to 0 for LoS only.

        los : bool
            Set if the LoS paths are computed.

        reflection : bool
            Set if the reflected paths are computed.

        Output
        -------
        candidates: [max_depth, num_samples], int
            All possible candidate paths with depth up to ``max_depth``.
            Entries correspond to primitives indices.
            For paths with depth lower than ``max_depth``, -1 is used as
            padding value.
            The first path is the LoS one if LoS is requested.

        los_candidates: [num_samples], int or `None`
            Candidates in LoS. For the exhaustive method, this is the list of
            all candidates. `None` is returned if ``max_depth`` is 0 or for
            empty scenes.
        """
        # Number of triangles
        n_prims = self._primitives.shape[0]

        # List of all triangles
        # [n_prims]
        all_prims = torch.arange(n_prims, dtype=torch.int32)

        # Empty scene or reflection disabled
        if (not reflection) or (n_prims == 0):
            if los:
                # Only LoS is added as candidate
                return torch.full([0, 1], -1), all_prims
            else:
                # No candidates
                return torch.full([0, 0], -1), all_prims

        # If reflection is disabled,

        # Number of candidate paths made of reflections only
        # num_samples = n_prims + n_prims^2 + ... + n_prims^max_depth
        if n_prims == 0:
            num_samples = 0
        elif n_prims == 1:
            num_samples = max_depth
        else:
            num_samples = (n_prims * (n_prims ** max_depth - 1)) // (n_prims - 1)
        # Add LoS path
        if los:
            num_samples += 1
        # Tensor of all possible reflections
        # Shape : [max_depth , num_samples]
        # It is transposed to fit the expected output shape at the end of this
        # function.
        # all_candidates[i,j] correspond to the triangle index intersected
        # by the i^th path for at j^th reflection.
        # The first column corresponds to LoS, i.e., no interaction.
        # -1 is used as padding value for path with depth lower than
        # max_depth.
        # Initialized with -1.
        all_candidates = torch.full([num_samples, max_depth], -1)
        # The next loop fill all_candidates with the list of intersected
        # primitives for all possible paths made of reflections only.
        # It starts from the paths with the 1 reflection, up to max_depth.
        # The variable `offset` corresponds to the index offset for storing the
        # paths in all_candidates.
        if los:
            # `offset` is initialized to 1 as the first path (depth = 0)
            # corresponds to LoS
            offset = 1
        else:
            # No LoS, `offset` is initialized to 0
            offset = 0
        for depth in range(1, max_depth + 1):
            # Enumerate all possible interactions for this depth
            # List of `depth` tensors with shape
            # [n_prims, ..., n_prims] and rank `depth`
            candidates = torch.meshgrid(*([all_prims] * depth), indexing='ij')

            # Reshape to
            # [n_prims**depth,depth]
            candidates = torch.stack([reshape(c, [-1]) for c in candidates],
                                     dim=1)

            # Pad with -1 for paths shorter than max_depth
            # [n_prims**depth,max_depth]
            candidates = torch.nn.functional.pad(candidates, [[0, 0], [0, max_depth - depth]],
                                                 mode='constant', value=-1)

            # Update all_candidates
            # Number of candidate paths for this depth
            num_candidates = candidates.shape[0]
            # Corresponding row indices in the all_candidates tensor
            indices = torch.arange(offset, offset + num_candidates, dtype=torch.int32)
            indices = torch.unsqueeze(indices, -1)
            # all_candidates : [max_depth , num_samples]
            all_candidates = scatter_nd_update(all_candidates,
                                               indices, candidates)

            # Prepare for next iteration
            offset += num_candidates

        # Transpose to fit the expected output shape.
        # [max_depth, num_samples]
        all_candidates = all_candidates.T

        # Primitives in LoS
        if max_depth > 0:
            los_candidates = all_prims
        else:
            los_candidates = None

        return all_candidates, los_candidates

    def _list_candidates_fibonacci(self, max_depth, sources, num_samples,
                                   los, reflection, scattering):
        r"""
        Generate potential candidate paths made of reflections only and the
        LoS. Rays direction are arranged in a Fibonacci lattice on the unit
        sphere.

        This can be used when the triangle count or maximum depth make the
        exhaustive method impractical.

        A budget of ``num_samples`` rays is split equally over the given
        sources. Starting directions are sampled uniformly at random.
        Paths are simulated until the maximum depth is reached.
        We record all sequences of primitives hit and the prefixes of these
        sequences, and return unique sequences.

        Input
        ------
        max_depth: int
            Maximum number of reflections.
            Set to 0 for LoS only.

        sources : [num_sources, 3], torch.float
            Coordinates of the sources.

        num_samples: int
            Number of rays to trace in order to generate candidates.
            A large sample count may exhaust GPU memory.

        los : bool
            If set to `True`, then the LoS paths are computed.

        reflection : bool
            If set to `True`, then the reflected paths are computed.

        scattering : bool
            if set to `True`, then the scattered paths are computed

        Output
        -------
        candidates_ref: [max_depth, num paths], int
            Unique sequence of hitted primitives, with depth up to ``max_depth``.
            Entries correspond to primitives indices.
            For paths with depth lower than max_depth, -1 is used as
            padding value.
            The first path is the LoS one if LoS is requested.

        los_candidates: [num_samples], int or `None`
            Primitives in LoS. `None` is returned if ``max_depth`` is 0.

        candidates_scat : [max_depth, num_sources, num_paths_per_source], int
            Sequence of primitives hit at `hit_points`. Compared to
            `candidates_ref`, it does not need to be unique, as the
            intersection points are different for every sequence, and is
            dependant on the source, as the intersection point are specific to
            the sources positions.

        hit_points : [max_depth, num_sources, num_paths_per_source, 3], torch.float
            Intersection points.
        """
        mask_t = dr.mask_t(self._mi_scalar_t)

        # Ensure that sample count can be distributed over the emitters
        num_sources = sources.shape[0]
        samples_per_source = int(dr.ceil(num_samples / num_sources))
        num_samples = num_sources * samples_per_source

        # List of candidates
        candidates = []

        # Hit points
        hit_points = []

        # Is the scene empty?
        is_empty = dr.shape(self._shape_indices)[0] == 0

        # Only shoot if the scene is not empty
        if not is_empty:

            # Keep track of which paths are still active
            active = dr.full(mask_t, True, num_samples)

            # Initial ray: Arranged in a Fibonacci lattice on the unit
            # sphere.
            # [samples_per_source, 3]
            lattice = fibonacci_lattice(samples_per_source, self._rdtype)
            source_i = dr.linspace(self._mi_scalar_t, 0, num_sources,
                                   num=num_samples, endpoint=False)
            sampled_d = self._mi_vec_t(torch.tile(lattice, [num_sources, 1]))
            source_i = mi.Int32(source_i)
            sources_dr = self._mi_tensor_t(sources)
            ray = mi.Ray3f(
                o=dr.gather(self._mi_vec_t, sources_dr.array, source_i),
                d=sampled_d,
            )

            for depth in range(max_depth):
                # Intersect ray against the scene to find the next hitted
                # primitive
                si = self._mi_scene.ray_intersect(ray, active)

                active &= si.is_valid()

                # Record which primitives were hit
                shape_i = dr.gather(mi.Int32, self._shape_indices,
                                    dr.reinterpret_array_v(mi.UInt32, si.shape),
                                    active)
                offsets = dr.gather(mi.Int32, self._prim_offsets, shape_i, active)
                prims_i = dr.select(active, offsets + si.prim_index, -1)
                candidates.append(prims_i)

                # Record the hit point
                hit_p = ray.o + si.t * ray.d
                hit_points.append(hit_p)

                # Prepare the next interaction, assuming purely specular
                # reflection
                ray = si.spawn_ray(si.to_world(mi.reflect(si.wi)))

        # For diffraction, we need only primitives in LoS
        # [num_los_primitives]
        if len(candidates) > 0:
            # max_depth > 0 or empty scene
            los_primitives = reshape(mi_to_torch_tensor(mi.Int32(candidates[0]), dtype=torch.int32), [-1])
            los_primitives = torch.unique(los_primitives)
            los_primitives = torch.gather(los_primitives, 0,
                                          torch.where(los_primitives != -1)[0])
        else:
            # max_depth == 0
            los_primitives = None

        reflection = reflection and (max_depth > 0) and (len(candidates) > 0)
        scattering = scattering and (max_depth > 0) and (len(candidates) > 0)

        if scattering or reflection:
            # Stack all found interactions along the depth dimension
            # [max_depth, num_samples]
            candidates = torch.stack([mi_to_torch_tensor(mi.Int32(r), torch.int32)
                                      for r in candidates], dim=0)

        if reflection:
            # [max_depth, num_samples]
            candidates_ref = candidates
            # Compute the actual max_depth
            # [max_depth]
            useless_step = torch.all(torch.eq(candidates_ref, -1), dim=1)
            # ()
            max_depth_ref = torch.where(torch.any(useless_step),
                                        torch.argmax(cast(useless_step, torch.int32)),
                                        max_depth)
            # [max_depth, num_samples]
            candidates_ref = candidates_ref[:max_depth_ref]
        else:
            # No candidates
            candidates_ref = torch.full([0, 0], -1)
            max_depth_ref = 0

        if scattering:
            # [max_depth, num_samples, 3]
            hit_points = torch.stack([mi_to_torch_tensor(r, self._rdtype)
                                      for r in hit_points])
            # [max_depth, num_sources, samples_per_source, 3]
            hit_points = reshape(hit_points,
                                 [max_depth, num_sources, samples_per_source, 3])
            # [max_depth, num_sources, samples_per_source]
            candidates_scat = reshape(candidates,
                                      [max_depth, num_sources, samples_per_source])
            # Flag indicating no hits
            # [max_depth, num_sources, samples_per_source]
            no_hit = torch.eq(candidates_scat, -1)
            # Compute the actual max_depth
            # [max_depth]
            useless_step = no_hit.all(1).all(1)
            # ()
            max_depth_scat = torch.where(torch.any(useless_step),
                                         torch.argmax(cast(useless_step, torch.int32)),
                                         max_depth)
            # [max_depth, num_sources, samples_per_source, 3]
            hit_points = hit_points[:max_depth_scat]
            # [max_depth, num_sources, samples_per_source]
            candidates_scat = candidates_scat[:max_depth_scat]
            # [max_depth, num_sources, samples_per_source]
            no_hit = no_hit[:max_depth_scat]
            # Remove useless paths
            # [samples_per_source]
            useful_samples = torch.logical_not(no_hit.all(0).all(0))
            useful_samples_index = torch.nonzero(useful_samples, as_tuple=False)[:, 0]
            # [max_depth, num_sources, num_paths_per_source, 3]
            hit_points = hit_points[:, :, useful_samples_index]
            # [max_depth, num_sources, num_paths_per_source]
            candidates_scat = candidates_scat[:, :, useful_samples_index]
            # [max_depth, num_sources, num_paths_per_source]
            no_hit = no_hit[:, :, useful_samples_index]

            # Zero the hit masked points
            # [max_depth, num_sources, num_paths, 3]
            hit_points = torch.where(torch.unsqueeze(no_hit, dim=-1),
                                     torch.zeros_like(hit_points),
                                     hit_points)
        else:
            # No hit points
            hit_points = torch.full([0, num_sources, 1, 3],
                                    cast(0., self._rdtype))
            candidates_scat = torch.full([0, num_sources, 1], False)
            max_depth_scat = 0

        if not reflection and not scattering:
            max_depth = 0
        # Remove duplicates
        if max_depth_ref > 0:
            candidates_ref, _ = torch.unique(candidates_ref, sorted=False, return_inverse=True, dim=1)

        # Add line-of-sight to list of candidates for reflection if
        # required
        if los:
            candidates_ref = torch.cat([torch.full([max_depth_ref, 1], -1),
                                        candidates_ref],
                                       dim=1)
        else:
            # Ensure there is no LoS by removing all paths corresponding
            # to no hits
            # [num_samples]
            is_nlos = torch.logical_not(torch.all(candidates_ref == -1,
                                                  dim=0))
            is_nlos_ind = torch.nonzero(is_nlos, as_tuple=False)[:, 0]
            candidates_ref = candidates_ref[:, is_nlos_ind]

        # The previous shoot and bounce process does not do next-event
        # estimation, and continues to trace until max_depth reflections occurs
        # or the ray does not intersect any primitive.
        # Therefore, we extend the set of rays with the prefixes of all
        # rays in `results_tf` to ensure we don't miss shorter paths than the
        # ones found.
        candidates_ref_ = [candidates_ref]
        for depth in range(1, max_depth_ref):
            # Extract prefix of length depth
            # [depth, num_samples]
            prefix = candidates_ref[:depth]
            # Pad with -1, i.e., not intersection
            # [max_depth, num_samples]
            prefix = torch.nn.functional.pad(prefix, (0, 0, 0, max_depth_ref - depth), value=-1)
            # Add to the list of rays
            candidates_ref_.insert(0, prefix)
        # [max_depth, num_samples]
        candidates_ref = torch.cat(candidates_ref_, dim=1)
        # TODO: figure out where this got set to int64 after fib3 and fix it there.
        candidates_ref = candidates_ref.to(torch.int32)

        # Extending the rays with prefixes might have created duplicates.
        # Remove duplicates
        if candidates_ref.shape[0] > 0:
            candidates_ref, _ = torch.unique(candidates_ref, sorted=False, return_inverse=True, dim=1)
        return candidates_ref, los_primitives, candidates_scat, hit_points

    ##################################################################
    # Methods used for computing the specular paths
    ##################################################################

    ### The following functions implement the image methods

    def _spec_image_method_phase_1(self, candidates, sources):
        r"""
        Implements the first phase of the image method.

        Starting from the sources, mirror each point against the
        given candidate primitive. At this stage, we do not carry
        any verification about the visibility of the ray.
        Loop through the max_depth interactions. All candidate paths are
        processed in parallel.

        Input
        ------
        candidates: [max_depth, num_samples], torch.int
            Set of candidate paths with depth up to ``max_depth``.
            For paths with depth lower than ``max_depth``, -1 must be used as
            padding value.
            The first path is the LoS one if LoS is requested.

        sources : [num_sources, 3], torch.float
            Positions of the sources from which rays (paths) are emitted

        Output
        -------
        mirrored_vertices : [max_depth, num_sources, num_samples, 3], torch.float
            Mirrored points coordinates

        tri_p0 : [max_depth, num_sources, num_samples, 3], torch.float
            Coordinates of the first vertex of potentially hitted triangles

        normals : [max_depth, num_sources, num_samples, 3], torch.float
            Normals to the potentially hitted triangles
        """

        # Max depth
        max_depth = candidates.shape[0]

        # Number of candidates
        num_samples = candidates.shape[1]

        # Number of sources and number of receivers
        num_sources = len(sources)

        # Sturctures are filled by the following loop
        # Indicates if a path is discarded
        # [num_samples]
        valid = torch.full([num_samples], True)
        # Coordinates of the first vertex of potentially hitted triangles
        # [max_depth, num_sources, num_samples, 3]
        tri_p0 = torch.zeros([max_depth, num_sources, num_samples, 3],
                             dtype=self._rdtype)
        # Coordinates of the mirrored vertices
        # [max_depth, num_sources, num_samples, 3]
        mirrored_vertices = torch.zeros([max_depth, num_sources, num_samples, 3],
                                        dtype=self._rdtype)
        # Normals to the potentially hitted triangles
        # [max_depth, num_sources, num_samples, 3]
        normals = torch.zeros([max_depth, num_sources, num_samples, 3],
                              dtype=self._rdtype)

        # Position of the last interaction.
        # It is initialized with the sources position
        # Add an additional dimension for broadcasting with the paths
        # [num_sources, 1, xyz : 1]
        current = torch.unsqueeze(sources, dim=1)
        current = torch.tile(current, [1, num_samples, 1])
        # Index of the last hit primitive
        prev_prim_idx = torch.full([num_samples], -1)
        if max_depth > 0:
            for depth in torch.arange(max_depth):

                # Primitive indices with which paths interact at this depth
                # [num_samples]
                prim_idx = candidates[depth]

                # Flag indicating which paths are still active, i.e., should be
                # tested.
                # Paths that are shorter than depth are marked as inactive
                # [num_samples]
                active = torch.ne(prim_idx, -1)

                # Break the loop if no active paths
                # Could happen with empty scenes, where we have only LoS
                if torch.logical_not(torch.any(active)):
                    break

                # Eliminate paths that go through the same prim twice in a row
                # [num_samples]
                valid = torch.logical_and(
                    valid,
                    torch.logical_or(~active, torch.ne(prim_idx, prev_prim_idx))
                )

                # On CPU, indexing with -1 does not work. Hence we replace -1
                # by 0.
                # This makes no difference on the resulting paths as such paths
                # are not flagged as active.
                # valid_prim_idx = prim_idx
                valid_prim_idx = torch.where(prim_idx == -1, 0, prim_idx)

                # Mirroring of the current point with respected to the
                # potentially hitted triangle.
                # We need the coordinate of the first vertex of the potentially
                # hitted triangle.
                # To get this, we build the indexing tensor to gather only the
                # coordinate of the first index
                # [[num_samples, 1]]
                p0_index = torch.unsqueeze(valid_prim_idx, dim=1)
                p0_index = torch.nn.functional.pad(p0_index, (0, 1), mode='constant', value=0)  # First vertex
                # [num_samples, xyz : 3]
                p0 = self._primitives[p0_index[:, 0], p0_index[:, 1], :]
                # Expand rank and tile to broadcast with the number of
                # transmitters
                # [num_sources, num_samples, xyz : 3]
                p0 = torch.unsqueeze(p0, dim=0)
                p0 = torch.tile(p0, [num_sources, 1, 1])
                # Gather normals to potentially intersected triangles
                # [num_samples, xyz : 3]
                normal = self._normals[valid_prim_idx]
                # Expand rank and tile to broadcast with the number of
                # transmitters
                # [1, num_samples, xyz : 3]
                normal = torch.unsqueeze(normal, dim=0)
                normal = torch.tile(normal, [num_sources, 1, 1])

                # Distance between the current intersection point (or sources)
                # and the plane the triangle is part of.
                # Note: `dist` is signed to compensate for backfacing normals
                # whenn needed.
                # [num_sources, num_samples, 1]
                dist = dot(current, normal, keepdim=True) \
                       - dot(p0, normal, keepdim=True)
                # Coordinates of the mirrored point
                # [num_sources, num_samples, xyz : 3]
                mirrored = current - 2. * dist * normal

                # Store these results
                # [max_depth, num_sources, num_samples, 3]
                mirrored_vertices = scatter_nd_update(
                    mirrored_vertices, torch.tensor([depth])[None], mirrored[None])
                # [max_depth, num_sources, num_samples, 3]
                tri_p0 = scatter_nd_update(tri_p0, torch.tensor([depth])[None], p0[None])
                # [max_depth, num_sources, num_samples, 3]
                normals = scatter_nd_update(normals,
                                            torch.tensor([depth])[None], normal[None])

                # Prepare for the next interaction
                # [num_sources, num_samples, xyz : 3]
                current = mirrored
                # [num_samples]
                prev_prim_idx = prim_idx

        return mirrored_vertices, tri_p0, normals

    def _spec_image_method_phase_21(self, depth, candidates, valid,
                                    mirrored_vertices, tri_p0, normals, current,
                                    num_targets, num_sources):
        # pylint: disable=line-too-long
        r"""
        Implement the first part of phase 2 of the image method:

        For a given ``depth``:
        - Computes the intersection point with the ``depth``th primitive of the
        sequence of candidates for a ray originating from ``current``
        - Checks that the intersection point is within the primitive
        - Ensures the normal points toward the ``current`` point
        - Prepares the ray to test for blockage between ``current`` point and
        thecomputed intersection point

        The obstruction test is note performed in this function as it uses
        Mitsuba.

        Input
        -----
        depth : int
            Current interaction number

        candidates: [max_depth, num_samples], torch.int
            Set of candidate paths with depth up to ``max_depth``.
            For paths with depth lower than ``max_depth``, -1 must be used as
            padding value.
            The first path is the LoS one if LoS is requested.

        valid : [num_targets, num_sources, num_samples], torch.bool
            Mask indicating the valid paths

        mirrored_vertices : [max_depth, num_sources, num_samples, 3], torch.float
            Mirrored points

        tri_p0 : [max_depth, num_sources, num_samples, 3], torch.float
            Coordinates of the first vertex of potentially hitted triangles

        normals : [max_depth, num_sources, num_samples, 3], torch.float
            Normals to the potentially hitted triangles

        current : [num_targets, 1, 1, xyz : 3], torch.float
            Positions of the last interactions

        num_targets : int
            Number of targets

        num_sources : int
            Number of sources

        Output
        ------
        valid : [num_targets, num_sources, num_samples], torch.bool
            Mask indicating the valid paths

        current : [num_targets, num_sources, num_samples, 3], torch.float
            Positions of the last interactions

        p : [num_targets, num_sources, num_samples, 3], torch.float
            Intersection point on the ``depth`` primitive

        n : [num_targets, num_sources, num_samples, 3], torch.float
            Normals to the primitive at the ``depth`` intersection point

        maxt : [num_targets, num_sources, num_samples], torch.float
            Distance from current to intersection point

        d : [num_targets, num_sources, num_samples, 3], torch.float
            Ray direction to test for blockage between ``curent`` and the
            intersection point

        active : [num_samples], torch.bool
            Mask indicating paths that are not active, i.e., didn't start yet
        """

        # Number of candidates at this stage
        num_samples = candidates.shape[1]

        # Primitive indices with which paths interact at this depth
        # [num_samples]
        prim_idx = candidates[depth]

        # Next mirrored point
        # [num_sources, num_samples, 3]
        next_pos = mirrored_vertices[depth][None]
        # Expand rank for broadcasting
        # [1, num_sources, num_samples, 3]

        # Since paths can have different depths, we have to mask out paths
        # that have not started yet.
        # [num_samples]
        active = torch.ne(prim_idx, -1)

        # Expand rank to broadcast with receivers and transmitters
        # [1, 1, num_samples]
        active = expand_to_rank(active, 3, axis=0)

        # Invalid paths are marked as inactive
        # [num_targets, num_sources, num_samples]
        active = torch.logical_and(active, valid)

        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # valid_prim_idx = prim_idx
        valid_prim_idx = torch.where(prim_idx == -1, 0, prim_idx)

        # Trace a direct line from the current position to the next path
        # vertex.

        # Ray direction
        # [num_targets, num_sources, num_samples, 3]
        d, _ = normalize(next_pos - current)

        # Find where it intersects the primitive that we mirrored against.
        # If that falls out of the primitive, this whole path is invalid.

        # Vertices forming the triangle.
        # [num_sources, num_samples, xyz : 3]
        p0 = tri_p0[depth]
        # Expand rank to broadcast with the target dimension
        # [1, num_sources, num_samples, xyz : 3]
        p0 = torch.unsqueeze(p0, dim=0)
        # Build the indexing tensor to gather only the coordinate of the
        # second index
        # [[num_samples, 1]]
        p1_index = torch.unsqueeze(valid_prim_idx, dim=1)
        p1_index = torch.nn.functional.pad(p1_index, (0, 1), mode='constant', value=1)  # Second vertex
        # [num_samples, xyz : 3]
        p1 = self._primitives[p1_index[:, 0], p1_index[:, 1], :]
        # Expand rank to broadcast with the target and sources
        # dimensions
        # [1, 1, num_samples, xyz : 3]
        p1 = expand_to_rank(p1, 4, axis=0)
        # Build the indexing tensor to gather only the coordinate of the
        # third index
        # [[num_samples, 1]]
        p2_index = torch.unsqueeze(valid_prim_idx, dim=1)
        p2_index = torch.nn.functional.pad(p2_index, (0, 1), mode='constant', value=2)  # Third vertex
        # [num_samples, xyz : 3]
        p2 = gather_nd(self._primitives, p2_index)
        # Expand rank to broadcast with the target and sources
        # dimensions
        # [1, 1, num_samples, xyz : 3]
        p2 = expand_to_rank(p2, 4, axis=0)
        # Intersection test.
        # We use the Moeller Trumbore algorithm
        # t : [num_targets, num_sources, num_samples]
        # hit : [num_targets, num_sources, num_samples]
        t, hit = moller_trumbore(current, d, p0, p1, p2, SolverBase.EPSILON)
        # [num_targets, num_sources, num_samples]
        valid = torch.logical_and(valid, torch.logical_or(~active, hit))

        # Force normal to point towards our current position
        # [num_sources, num_samples, 3]
        n = normals[depth]
        # Add dimension for broadcasting with receivers
        # [1, num_sources, num_samples, 3]
        n = torch.unsqueeze(n, dim=0)
        # Force to point towards current position
        # [num_targets, num_sources, num_samples, 3]
        s = torch.sign(dot(n, current - p0, keepdim=True))
        n = n * s
        # Intersection point
        # [num_targets, num_sources, num_samples, 3]
        t = torch.unsqueeze(t, dim=3)
        p = current + t * d

        # Prepare obstruction test.
        # There should be no obstruction between the actual
        # interaction point and the current point.
        # We use Mitsuba to test for obstruction efficiently.
        # We only compute here the origin and direction of the ray

        # Ensure current is already broadcasted
        # [num_targets, num_sources, num_samples, 3]
        current = torch.broadcast_to(current, [num_targets, num_sources,
                                               num_samples, 3])
        # Distance from current to intersection point
        # [num_targets, num_sources, num_samples]
        maxt = torch.norm(current - p, dim=-1)

        output = (
            valid,
            current,
            p,
            n,
            maxt,
            d,
            active
        )
        return output

    def _spec_image_method_phase_22(self, depth, valid, mirrored_vertices,
                                    current, blk, num_targets, num_sources, maxt, p, active):
        r"""
        Implement the second part of phase 2 of the image method:

        - Discards paths that are blocked
        - Discards paths for which ``current`` point and the ``next_pos`` are
            not on the same side, as this would mean that the path is going
            through the surface

        Input
        -----
        depth : int
            Current interaction number

        valid : [num_targets, num_sources, num_samples], torch.bool
            Mask indicating the valid paths

        mirrored_vertices : [max_depth, num_sources, num_samples, 3], torch.float
            Mirrored points

        current : [num_targets, num_sources, num_samples, 3], torch.float
            Positions of the last interactions

        blk : [num_targets*num_sources*num_samples], torch.bool
            Mask indicating which blocked paths

        num_targets : int
            Number of targets

        num_sources : int
            Number of sources

        maxt : [num_targets, num_sources, num_samples], torch.float
            Distance from current to intersection point

        p : [num_targets, num_sources, num_samples, 3], torch.float
            Intersection point on the ``depth`` primitive

        active : [num_targets, num_sources, num_samples], torch.bool
            Flag indicating which paths are active

        Output
        ------
        valid : [num_targets, num_sources, num_samples], torch.bool
            Mask indicating the valid paths

        current : [num_targets, num_sources, num_samples, 3], torch.float
            Positions of the last interactions
        """

        # Number of candidates at this stage
        num_samples = valid.shape[2]

        # Next mirrored point
        # [num_sources, num_samples, 3]
        next_pos = mirrored_vertices[depth]
        # Expand rank for broadcasting
        # [1, num_sources, num_samples, 3]
        next_pos = torch.unsqueeze(next_pos, dim=0)

        # Discard paths if blocked
        # [num_targets, num_sources, num_samples]
        blk = reshape(blk, [num_targets, num_sources, num_samples])
        valid = torch.logical_and(valid, torch.logical_or(~active, ~blk))

        # Discard paths for which the shooted ray has zero-length, i.e.,
        # when two consecutive intersection points have the same location,
        # or when the source and target have the same locations (RADAR).
        # [num_targets, num_sources, num_samples]
        blk = torch.lt(maxt, SolverBase.EPSILON)
        # [num_targets, num_sources, num_samples]
        valid = torch.logical_and(valid, torch.logical_or(~active, ~blk))

        # We must also ensure that the current point and the next_pos are
        # not on the same side, as this would mean that the path is going
        # through the surface

        # Vector from the intersection point to the current point
        # [num_targets, num_sources, num_samples, 3]
        v1 = current - p
        # Vector from the intersection point to the next point
        # [num_targets, num_sources, num_samples, 3]
        v2 = next_pos - p
        # Compute the scalar product. It must be negative, as we are using
        # the image (next_pos)
        # [num_targets, num_sources, num_samples]
        blk = dot(v1, v2)
        blk = torch.ge(blk, torch.zeros_like(blk))
        valid = torch.logical_and(valid, torch.logical_or(~active, ~blk))

        # Update active state
        # [num_targets, num_sources, num_samples]
        active = torch.logical_and(active, valid)
        # Prepare for next path segment
        # [num_targets, num_sources, num_samples, 3]
        current = torch.where(torch.unsqueeze(active, dim=-1), p, current)

        output = (
            valid,
            current
        )
        return output

    def _spec_image_method_phase_23(self, current, sources, num_targets):
        r"""
        Implements the third step of phase 2 of the image method.

        Prepares the rays for testing blockage between the last interaction
        point and the sources.

        Input
        ------
        current : [num_targets, num_sources, num_samples, 3], torch.float
            Positions of the last interactions

        sources : [num_sources, 3], torch.float
            Sources from which rays (paths) are emitted

        num_targets : int
            Number of targets

        Output
        ------
        current : [num_targets, num_sources, num_samples, 3], torch.float
            Positions of the last interactions

        d : [num_targets, num_sources, num_samples, 3], torch.float
            Ray direction between the last interaction point and the sources

        maxt : [num_targets, num_sources, num_samples], torch.float
            Distances between the last interaction point and the sources
        """

        # Number of candidates at this stage
        num_samples = current.shape[2]

        num_sources = sources.shape[0]

        # Check visibility to the transmitters
        # [1, num_sources, 1, 3]
        sources_ = torch.unsqueeze(torch.unsqueeze(sources, dim=0),
                                   dim=2)
        # Direction vector and distance to the transmitters
        # d : [num_targets, num_sources, num_samples, 3]
        # maxt : [num_targets, num_sources, num_samples]
        d, maxt = normalize(sources_ - current)
        # Ensure current is already broadcasted
        # [num_targets, num_sources, num_samples, 3]
        current = torch.broadcast_to(current, [num_targets, num_sources,
                                               num_samples, 3])
        d = torch.broadcast_to(d, [num_targets, num_sources, num_samples, 3])
        maxt = torch.broadcast_to(maxt, [num_targets, num_sources, num_samples])

        return current, d, maxt

    def _spec_image_method_phase_3(self, candidates, valid, num_targets,
                                   num_sources, path_vertices, path_normals, blk):
        # pylint: disable=line-too-long
        r"""
        Implements the third phase of the image method.

        Post-process the valid paths, from transmitters to receivers, to put
        them in the expected output format.

        Input
        -----
        candidates: [max_depth, num_samples], torch.int
            Set of candidate paths with depth up to ``max_depth``.
            For paths with depth lower than ``max_depth``, -1 must be used as
            padding value.
            The first path is the LoS one if LoS is requested.

        valid : [num_targets, num_sources, num_samples], torch.bool
            Mask indicating the valid paths

        num_targets : int
            Number of targets

        num_sources : int
            Number of sources

        path_vertices : [max_depth, num_targets, num_sources, num_samples, xyz : 3]
            Positions of the intersection points

        path_normals : [max_depth, num_targets, num_sources, num_samples, 3]
            Normals to the surface at the intersection points

        blk : [num_targets*num_sources*num_samples], torch.bool
            Mask indicating which blocked paths

        Output
        ------
        mask : [num_targets, num_sources, max_num_paths], torch.bool
             Mask indicating if a path is valid

        valid_vertices : [max_depth, num_targets, num_sources, max_num_paths, 3], torch.float
            Positions of intersection points.

        valid_objects : [max_depth, num_targets, num_sources, max_num_paths], torch.int
            Indices of the intersected scene objects or wedges.
            Paths with depth lower than ``max_depth`` are padded with `-1`.

        valid_normals : [max_depth, num_targets, num_sources, max_num_paths, 3], torch.float
            Normals to the primitives at the intersection points.
        """

        # Discard blocked paths
        # [num_targets, num_sources, num_samples]
        valid = torch.logical_and(valid, ~blk)

        # Max depth
        max_depth = candidates.shape[0]

        # If at least one link has the LoS paths flagged as valid,
        # then we keep the LoS paths for all links.
        # This makes the tracking of paths types easier.
        # The LoS paths will be masked for links for which it is obstructed.
        # Note that there is only one entry that corresponds to LoS paths
        # [1, num_samples]
        is_los = torch.eq(candidates, -1).any(0, True)
        # Only keep the LoS path if it is valid (i.e., not obstructed) for at
        # least one link
        # [1, 1, num_samples]
        is_los = is_los.unsqueeze(0)
        is_los = torch.logical_and(is_los, valid).any(0).any(0)[None, None]

        # Build indices for keeping only valid path
        # A path is kept if it's valid or the LoS, and there is at least one link
        # for which LoS is not obstructed
        # [num_targets, num_sources, num_samples]
        keep = torch.logical_or(valid, is_los)
        # [num_targets, num_sources]
        num_paths = torch.sum(keep.to(torch.int32), dim=-1)
        # Maximum number of paths
        max_num_paths = torch.max(num_paths)
        # [num_valid, 3]
        gather_indices = torch.nonzero(keep, as_tuple=False)
        # [num_targets, num_sources, num_samples]
        path_indices = torch.cumsum(keep.to(torch.int32), dim=-1)
        # [num_valid]
        path_indices = gather_nd(path_indices, gather_indices)- 1

        # [3, num_valid]
        scatter_indices = gather_indices.t()
        if scatter_indices.numel() != 0:
            # [3, num_valid]
            scatter_indices[2] = path_indices
        # [num_valid, 3]
        scatter_indices = scatter_indices.t()
        gather_indices = torch.nonzero(keep, as_tuple=False)

        # Mask of valid paths
        # [num_targets, num_sources, max_num_paths]
        mask = torch.full((num_targets, num_sources, max_num_paths), False, dtype=torch.bool)
        # [num_keep_paths]
        mask_ = gather_nd(valid, gather_indices)
        # [num_targets, num_sources, max_num_paths]
        mask = scatter_nd_update(mask, scatter_indices, mask_)

        # Locations of the interactions
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        valid_vertices = torch.zeros((max_depth, num_targets, num_sources, max_num_paths, 3), dtype=self._rdtype)

        # Normals at the intersection points
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        valid_normals = torch.zeros((max_depth, num_targets, num_sources, max_num_paths, 3), dtype=self._rdtype)

        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_primitives = torch.full((max_depth, num_targets, num_sources, max_num_paths), -1, dtype=torch.int32)

        if max_depth > 0:
            for depth in torch.arange(max_depth, dtype=torch.int64):
                # Indices for storing the valid vertices/normals/primitives for this depth
                scatter_indices_ = F.pad(scatter_indices, (1, 0, 0, 0), mode='constant', value=depth)

                # Location of the interactions
                # Extract only the valid paths
                # [num_targets, num_sources, num_samples, 3]
                vertices_ = path_vertices[depth]
                # [total_num_valid_paths, 3]
                vertices_ = gather_nd(vertices_, gather_indices)
                # Store the valid intersection points
                # [max_depth, num_targets, num_sources, max_num_paths, 3]
                valid_vertices = scatter_nd_update(valid_vertices, scatter_indices_, vertices_)

                # Normals at the interactions
                # Extract only the valid paths
                # [num_targets, num_sources, num_samples, 3]
                normals_ = path_normals[depth]
                # [total_num_valid_paths, 3]
                normals_ = gather_nd(normals_, gather_indices)
                # Store the valid normals
                # [max_depth, num_targets, num_sources, max_num_paths, 3]
                valid_normals = scatter_nd_update(valid_normals, scatter_indices_, normals_)

                # Intersected primitives
                # Extract only the valid paths
                # [num_samples]
                primitives_ = candidates[depth]
                # [total_num_valid_paths]
                primitives_ = torch.gather(primitives_, 0, gather_indices[:, 2])
                # Store the valid primitives
                # [max_depth, num_targets, num_sources, max_num_paths]
                valid_primitives = scatter_nd_update(valid_primitives, scatter_indices_, primitives_)


        # Add a dummy entry to primitives_2_objects with value -1 for invalid
        # reflection.
        # Invalid reflection, i.e., corresponding to paths with a depth lower
        # than max_depth, will be assigned -1 as index of the intersected
        # shape.
        # [num_samples + 1]
        primitives_2_objects = F.pad(self._primitives_2_objects, (0, 1), mode='constant', value=-1)
        # Replace all -1 by num_samples
        num_samples = self._primitives_2_objects.shape[0]
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_primitives = torch.where(torch.eq(valid_primitives, -1), num_samples, valid_primitives)
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_objects = primitives_2_objects[valid_primitives]

        # Actual maximum depth
        if max_depth > 0:
            # Limit the depth to the actual max_depth
            # [max_depth]
            useless_depth = torch.eq(valid_objects, -1).all(1).all(1).all(1)

            max_depth = torch.where(useless_depth.any(), torch.argmax(useless_depth.to(torch.int32)), max_depth)
            max_depth = max_depth.clamp_min(1)
            # [max_depth, num_targets, num_sources, max_num_paths, 3]
            valid_vertices = valid_vertices[:max_depth]
            # [max_depth, num_targets, num_sources, max_num_paths, 3]
            valid_normals = valid_normals[:max_depth]
            # [max_depth, num_targets, num_sources, max_num_paths]
            valid_objects = valid_objects[:max_depth]

        return mask, valid_vertices, valid_objects, valid_normals

    def _spec_image_method(self, candidates, paths, spec_paths_tmp):
        # pylint: disable=line-too-long
        r"""
        Evaluates a list of candidate paths ``candidates`` and keep only the
        valid ones, i.e., the non-obstricted ones with valid reflections only,
        using the image method.

        Input
        -----
        candidates: [max_depth, num_samples], torch.int
            Set of candidate paths with depth up to ``max_depth``.
            For paths with depth lower than ``max_depth``, -1 must be used as
            padding value.
            The first path is the LoS one if LoS is requested.

        paths : :class:`~torchrf.rt.Paths`
            Paths to update
        """

        sources = paths.sources
        targets = paths.targets

        # Max depth
        max_depth = candidates.shape[0]

        # Number of sources and number of receivers
        num_sources = len(sources)
        num_targets = len(targets)

        # --- Phase 1
        # Starting from the sources, mirror each point against the
        # given candidate primitive. At this stage, we do not carry
        # any verification about the visibility of the ray.
        # Loop through the max_depth interactions. All candidate paths are
        # processed in parallel.
        #
        # mirrored_vertices : [max_depth, num_sources, num_samples, 3], torch.float
        #     Mirrored points coordinates
        #
        # tri_p0 : [max_depth, num_sources, num_samples, 3], torch.float
        #     Coordinates of the first vertex of potentially hitted triangles
        #
        # normals : [max_depth, num_sources, num_samples, 3], torch.float
        #     Normals to the potentially hitted triangles
        mirrored_vertices, tri_p0, normals = \
            self._spec_image_method_phase_1(candidates, sources)

        # --- Phase 2

        # Number of candidates at this stage
        num_samples = candidates.shape[1]

        # Starting from the receivers, go over the vertices in reverse
        # and check that connections are possible.

        # Mask indicating which paths are valid
        # [num_targets, num_sources, num_samples]
        valid = torch.full([num_targets, num_sources, num_samples], True)
        # Positions of the last interactions.
        # Initialized with the positions of the receivers.
        # Add two additional dimensions for broadcasting with transmitters and
        # paths.
        # [num_targets, 1, 1, xyz : 3]
        current = expand_to_rank(targets, 4, axis=1)
        # Positions of the interactions.
        # [max_depth, num_targets, num_sources, num_samples, xyz : 3]
        # path_vertices = torch.zeros([max_depth, num_targets, num_sources,
        #                             num_samples, 3], dtype=self._rdtype)
        path_vertices = []
        # Normals at the interactions.
        # [max_depth, num_targets, num_sources, num_samples, xyz : 3]
        path_normals = []
        for depth in torch.arange(max_depth - 1, -1, -1):
            # The following call:
            # - Computes the intersection point with the ``depth``th primitive
            #   of the sequence of candidates for a ray originating from
            #   ``current``
            # - Checks that the intersection point is within the primitive
            # - Ensures the normal points toward the ``current`` point
            # - Prepares the ray to test for blockage between ``depth-1``th
            # point and the current point
            output = self._spec_image_method_phase_21(depth, candidates, valid,
                                                      mirrored_vertices, tri_p0, normals, current, num_targets,
                                                      num_sources)
            # [num_targets, num_sources, num_samples]
            #   Mask indicating the valid paths
            valid = output[0]
            # [num_targets, 1, 1, xyz : 3]
            #   Positions of the last interactions
            current = output[1]
            # : [num_targets, num_sources, num_samples, 3], torch.float
            #     Intersection point on the ``depth`` primitive
            path_vertices_ = output[2]
            # : [num_targets, num_sources, num_samples, 3], torch.float
            #    Normals to the primitive at the ``depth`` intersection point
            path_normals_ = output[3]
            # maxt : [num_targets, num_sources, num_samples], torch.float
            #     Distance from current to intersection point
            maxt = output[4]
            # d : [num_targets, num_sources, num_samples, 3], torch.float
            #     Ray direction to test for blockage between ``curent`` and the
            #     intersection point
            d = output[5]
            # [num_targets, num_sources, num_samples]
            #   Mask indicating paths that are not active, i.e., didn't start
            #   yet
            active = output[6]

            # Test for obstruction using Mitsuba
            # As Mitsuba only hanldes a single batch dimension, we flatten the
            # batch dims [num_targets, num_sources, num_samples]
            # [num_targets*num_sources*num_samples]
            blk = self._test_obstruction(reshape(current, [-1, 3]),
                                         reshape(d, [-1, 3]),
                                         reshape(maxt, [-1]))

            # The following call:
            # - Discards paths that are blocked
            # - Discards paths for which ``current`` point and the ``next_pos``
            #   are not on the same side, as this would mean that the path is
            #   going through the surface
            output = self._spec_image_method_phase_22(depth, valid,
                                                      mirrored_vertices, current, blk, num_targets, num_sources, maxt,
                                                      path_vertices_, active)
            # [num_targets, num_sources, num_samples]
            #   Mask indicating the valid paths
            valid = output[0]
            # [num_targets, num_sources, num_samples, xyz : 3]
            #   Positions of the last interactions
            current = output[1]

            path_vertices.append(path_vertices_)
            path_normals.append(path_normals_)

        path_vertices.reverse()
        path_normals.reverse()
        path_vertices = torch.stack(path_vertices, dim=0)
        path_normals = torch.stack(path_normals, dim=0)

        # Prepares the rays for testing blockage between the last
        # interaction point and the sources.
        #
        # current : [num_targets, num_sources, num_samples, 3], torch.float
        #     Positions of the last interactions
        #
        # d : [num_targets, num_sources, num_samples, 3], torch.float
        #     Ray direction between the last interaction point and the sources
        #
        # maxt : [num_targets, num_sources, num_samples], torch.float
        #     Distances between the last interaction point and the sources
        current, d, maxt = self._spec_image_method_phase_23(current, sources,
                                                            num_targets)

        # Test for obstruction using Mitsuba
        # [num_targets*num_sources*num_samples]
        val = self._test_obstruction(reshape(current, [-1, 3]),
                                     reshape(d, [-1, 3]),
                                     reshape(maxt, [-1]))
        # [num_targets, num_sources, num_samples, 3]
        blk = val.reshape(maxt.shape)
        # Discard paths for which the shooted ray has zero-length, i.e., when
        # two consecutive intersection points have the same location, or when
        # the source and target have the same locations (RADAR).
        # [num_targets, num_sources, num_samples]
        blk = torch.logical_or(blk, torch.lt(maxt, SolverBase.EPSILON))

        # --- Phase 3
        # Post-process the valid paths, from transmitters to receivers, to put
        # them in the expected output format.
        #
        # mask : [num_targets, num_sources, max_num_paths], torch.bool
        #      Mask indicating if a path is valid
        #
        # valid_vertices : [max_depth, num_targets, num_sources,
        #                   max_num_paths, 3], torch.float
        #     Positions of intersection points.
        #
        # valid_objects : [max_depth, num_targets, num_sources,
        #                   max_num_paths], torch.int
        #     Indices of the intersected scene objects or wedges.
        #     Paths with depth lower than ``max_depth`` are padded with `-1`.
        #
        # valid_normals : [max_depth, num_targets, num_sources,
        #                   max_num_paths, 3], torch.float
        #     Normals to the primitives at the intersection points.
        mask, valid_vertices, valid_objects, valid_normals = \
            self._spec_image_method_phase_3(candidates, valid, num_targets,
                                            num_sources, path_vertices, path_normals, blk)

        # Update the object storing the paths
        paths.mask = mask
        paths.vertices = valid_vertices
        paths.objects = valid_objects

        spec_paths_tmp.normals = valid_normals

    ### Transition matrices

    def _spec_transition_matrices(self, relative_permittivity,
                                  scattering_coefficient,
                                  paths, paths_tmp, scattering):
        # pylint: disable=line-too-long
        """
        Compute the transfer matrices, delays, angles of departures, and angles
        of arrivals, of paths from a set of valid reflection paths and the
        EM properties of the materials.

        Input
        ------
        relative_permittivity : [num_shape], torch.complex
            Tensor containing the relative permittivity of all shapes

        scattering_coefficient : [num_shape], torch.float
            Tensor containing the scattering coefficients of all shapes

        paths : :class:`~torchrf.rt.Paths`
            Paths to update

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Addtional quantities required for paths computation

        scattering : bool
            Set to `True` if computing the scattered paths.

        Output
        -------
        paths : :class:`~torchrf.rt.Paths`
            Updated paths
        """

        vertices = paths.vertices
        targets = paths.targets
        sources = paths.sources
        objects = paths.objects
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        normals = paths_tmp.normals
        k_i = paths_tmp.k_i
        k_r = paths_tmp.k_r
        total_distance = paths_tmp.total_distance

        # Maximum depth
        max_depth = vertices.shape[0]
        # Number of targets
        num_targets = targets.shape[0]
        # Number of sources
        num_sources = sources.shape[0]
        # Maximum number of paths
        max_num_paths = objects.shape[3]

        # Flag that indicates if a ray is valid
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_ray = torch.ne(objects, -1)

        # Tensor with relative perimittivities values for all reflection points
        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_object_idx = torch.where(objects == -1, 0, objects).to(torch.int64)
        # [max_depth, num_targets, num_sources, max_num_paths]
        if relative_permittivity.shape[0] == 0:
            etas = torch.zeros_like(valid_object_idx, dtype=self._dtype)
            scattering_coefficient = torch.zeros_like(valid_object_idx,
                                                      dtype=self._rdtype)
        else:
            etas = relative_permittivity[valid_object_idx]
            scattering_coefficient = scattering_coefficient[valid_object_idx]

        # Compute cos(theta) at each reflection point
        # [max_depth, num_targets, num_sources, max_num_paths]
        cos_theta = -dot(k_i[:max_depth], normals)

        # Compute e_i_s, e_i_p, e_r_s, e_r_p at each reflection point
        # all : [max_depth, num_targets, num_sources, max_num_paths,3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s, e_i_p, e_r_s, e_r_p = compute_field_unit_vectors(k_i[:max_depth],
                                                                k_r, normals, SolverBase.EPSILON)

        # Compute r_s, r_p at each reflection point
        # [max_depth, num_targets, num_sources, max_num_paths]
        r_s, r_p = reflection_coefficient(etas, cos_theta)

        # Multiply the reflection coefficients with the
        # reflection reduction factor
        # [max_depth, num_targets, num_sources, max_num_paths]
        reduction_factor = torch.sqrt(1 - scattering_coefficient ** 2)
        reduction_factor = torch.complex(reduction_factor,
                                         torch.zeros_like(reduction_factor))

        # Compute the field transfer matrix.
        # It is initialized with the identity matrix of size 2 (S and P
        # polarization components)
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = (torch.eye(n=2, dtype=self._dtype).reshape(1, 1, 1, 2, 2)
                 .repeat(num_targets, num_sources, max_num_paths, 1, 1))
        # For scattering, we keep track of the transition matrix for every
        # prefix of the path
        if scattering:
            # [max_depth, num_targets, num_sources, max_num_paths, 2, 2]
            prefix_mat_t = (torch.eye(n=2, dtype=self._dtype).reshape(1, 1, 1, 1, 2, 2)
                            .repeat(max_depth, num_targets, num_sources, max_num_paths, 1, 1))
        # Initialize last field unit vector with outgoing ones
        # [num_targets, num_sources, max_num_paths, 3]
        last_e_r_s = theta_hat(theta_t, phi_t)
        last_e_r_p = phi_hat(phi_t)
        for depth in torch.arange(0, max_depth):

            # Is this a valid reflection?
            # [num_targets, num_sources, max_num_paths]
            valid = valid_ray[depth]

            # [num_targets, num_sources, max_num_paths]
            reduction_factor_ = reduction_factor[depth]
            # [num_targets, num_sources, max_num_paths, 1, 1]
            reduction_factor_ = insert_dims(reduction_factor_, 2, -1)

            # Early stopping if no active rays
            if not torch.any(valid):
                break

            # Add dimension for broadcasting with coordinates
            # [num_targets, num_sources, max_num_paths, 1]
            valid_ = torch.unsqueeze(valid, dim=-1)

            # Change of basis matrix
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_cob = component_transform(last_e_r_s, last_e_r_p,
                                          e_i_s[depth], e_i_p[depth])
            mat_cob = torch.complex(mat_cob, torch.zeros_like(mat_cob))
            # Only apply transform if valid reflection
            # [num_targets, num_sources, max_num_paths, 1, 1]
            valid__ = torch.unsqueeze(valid_, dim=-1)
            # [num_targets, num_sources, max_num_paths, 2, 2]
            e = torch.where(valid__, torch.matmul(mat_cob, mat_t), mat_t)
            # Only update ongoing direction for next iteration if this
            # reflection is valid and if this is not the last step
            last_e_r_s = torch.where(valid_, e_r_s[depth], last_e_r_s)
            last_e_r_p = torch.where(valid_, e_r_p[depth], last_e_r_p)

            # Fresnel coefficients
            # [num_targets, num_sources, max_num_paths, 2]
            r = torch.stack([r_s[depth], r_p[depth]], -1)
            # Set the coefficients to one if non-valid reflection
            # [num_targets, num_sources, max_num_paths, 2]
            r = torch.where(valid_, r, torch.ones_like(r))
            # Add a dimension to broadcast with mat_t
            # [num_targets, num_sources, max_num_paths, 2, 1]
            r = torch.unsqueeze(r, dim=-1)
            # Apply Fresnel coefficient
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = r * e

            if scattering:
                # [max_depth, num_targets, num_sources, max_num_paths, 2, 2]
                prefix_mat_t[depth] = mat_t

            # Apply the reduction factor
            # [num_targets, num_sources, max_num_paths, 2]
            reduction_factor_ = torch.where(valid__, reduction_factor_,
                                            torch.ones_like(reduction_factor_))
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = mat_t * reduction_factor_

        # Move to the targets frame
        # This is not done for scattering as we stop the last interaction point
        if not scattering:
            # Transformation matrix
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_cob = component_transform(last_e_r_s, last_e_r_p,
                                          theta_hat(theta_r, phi_r),
                                          phi_hat(phi_r))
            mat_cob = torch.complex(mat_cob, torch.zeros_like(mat_cob))
            # Apply transformation
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = torch.matmul(mat_cob, mat_t)

        # Divide by total distance to account for propagation loss
        # [num_targets, num_sources, max_num_paths, 1, 1]
        # or
        # [max_depth, num_targets, num_sources, max_num_paths, 1, 1]
        total_distance = expand_to_rank(total_distance, mat_t.dim(),
                                        axis=total_distance.dim())
        total_distance = torch.complex(total_distance,
                                       torch.zeros_like(total_distance))
        if scattering:
            # Due to the high rank of these tensor, some reshape is required
            # so that tensorflow can do the division.
            # [max_depth, num_targets, num_sources, max_num_paths, 2, 2]
            prefix_mat_t = divide_no_nan(
                reshape(prefix_mat_t,
                        [max_depth * num_targets, num_sources, -1, 2, 2]),
                reshape(total_distance,
                        [max_depth * num_targets, num_sources, -1, 1, 1]))
            prefix_mat_t = reshape(prefix_mat_t,
                                   [max_depth, num_targets, num_sources, -1, 2, 2])
        else:
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = divide_no_nan(mat_t, total_distance)

        # Set invalid paths to 0 and stores the transition matrices
        # This is not done for scattering as further processing is required
        if not scattering:
            # Expand masks to broadcast with the field components
            # [num_targets, num_sources, max_num_paths, 1, 1]
            mask_ = expand_to_rank(paths.mask, 5, axis=3)
            # Zeroing coefficients corresponding to non-valid paths
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = torch.where(mask_, mat_t, torch.zeros_like(mat_t))
            paths_tmp.mat_t = mat_t
        else:
            # Expand masks to broadcast with the field components
            # [max_depth, num_targets, num_sources, max_num_paths, 1, 1]
            mask_ = expand_to_rank(paths_tmp.scat_prefix_mask, 6, axis=4)
            # Zeroing coefficients corresponding to non-valid paths
            # [max_depth, num_targets, num_sources, max_num_paths, 2, 2]
            prefix_mat_t = torch.where(mask_, prefix_mat_t,
                                       torch.zeros_like(prefix_mat_t))
            paths_tmp.scat_prefix_mat_t = prefix_mat_t

        return paths, paths_tmp

    ##################################################################
    # Methods used for computing the diffracted paths
    ##################################################################

    def _discard_obstructing_wedges_and_corners(self, candidate_wedges, targets,
                                                sources):
        r"""
        Discard wedges for which at least one of the source or target are
        "inside" the wedge

        Inputs
        ------
        candidate_wedges : [num_candidate_wedges], int
            Candidate wedges.
            Entries correspond to wedges indices.

        targets : [num_targets, 3], torch.float
            Coordinates of the targets.

        sources : [num_sources, 3], torch.float
            Coordinates of the sources.

        Output
        -------
        wedges_indices : [num_targets, num_sources, max_num_paths], torch.int
            Indices of the wedges that interacted with the diffracted paths
        """

        epsilon = cast(SolverBase.EPSILON, self._rdtype)

        # [num_candidate_wedges, 3]
        origins = self._wedges_origin[candidate_wedges]

        # Expand to broadcast with sources/targets and 0/n faces
        # [1, num_candidate_wedges, 1, 3]
        origins = torch.unsqueeze(origins, dim=0)
        origins = torch.unsqueeze(origins, dim=2)

        # Normals
        # [num_candidate_wedges, 2, 3]
        # [:,0,:] : 0-face
        # [:,1,:] : n-face
        normals = self._wedges_normals[candidate_wedges]
        # Expand to broadcast with the sources or targets
        # [1, num_candidate_wedges, 2, 3]
        normals = torch.unsqueeze(normals, dim=0)

        # Expand to broadcast with candidate and 0/n faces wedges
        # [num_sources, 1, 1, 3]
        sources = expand_to_rank(sources, 4, 1)
        # [num_targets, 1, 1, 3]
        targets = expand_to_rank(targets, 4, 1)
        # Sources/Targets vectors
        # [num_sources, num_candidate_wedges, 1, 3]
        u_t = sources - origins
        # [num_targets, num_candidate_wedges, 1, 3]
        u_r = targets - origins

        # [num_sources, num_candidate_wedges, 2]
        sources_valid_half_space = dot(u_t, normals)
        sources_valid_half_space = torch.gt(sources_valid_half_space,
                                            torch.full(sources_valid_half_space.shape, epsilon))
        # [num_sources, num_candidate_wedges]
        sources_valid_half_space = torch.any(sources_valid_half_space,
                                             dim=2)
        # Expand to broadcast with targets
        # [1, num_sources, num_candidate_wedges]
        sources_valid_half_space = torch.unsqueeze(sources_valid_half_space,
                                                   dim=0)

        # [num_targets, num_candidate_wedges, 2]
        targets_valid_half_space = dot(u_r, normals)
        targets_valid_half_space = torch.gt(targets_valid_half_space,
                                            torch.full(targets_valid_half_space.shape, epsilon))
        # [num_targets, num_candidate_wedges]
        targets_valid_half_space = torch.any(targets_valid_half_space,
                                             dim=2)
        # Expand to broadcast with sources
        # [num_targets, 1, num_candidate_wedges]
        targets_valid_half_space = torch.unsqueeze(targets_valid_half_space,
                                                   dim=1)

        # [num_targets, num_sources, max_num_paths = num_candidate_wedges]
        mask = torch.logical_and(sources_valid_half_space,
                                 targets_valid_half_space)

        # Discard paths with no valid link
        # [max_num_paths]
        valid_paths = torch.where(mask.any(0).any(0))[0]
        # [num_targets, num_sources, max_num_paths]
        mask = mask[:, :, valid_paths]
        # [max_num_paths]
        wedges_indices = candidate_wedges[valid_paths]
        # Set invalid wedges to -1
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = torch.where(mask, wedges_indices, -1)

        return wedges_indices

    def _compute_diffraction_points(self, targets, sources, wedges_indices):
        r"""
        Compute the interaction points on the wedges that minimizes the path
        length, and masks the wedges for which the interaction points is
        not on the finite wedge.

        Note: This calculation is done in double-precision (64bit).

        Input
        ------
        targets : [num_targets, 3], torch.float
            Coordinates of the targets.

        sources : [num_sources, 3], torch.float
            Coordinates of the sources.

        wedges_indices : [num_targets, num_sources, max_num_paths], torch.int
            Indices of the wedges that interacted with the diffracted paths

        Output
        -------
        wedges_indices : [num_targets, num_sources, max_num_paths], torch.int
            Indices of the wedges that interacted with the diffracted paths

        vertices : [num_targets, num_sources, max_num_paths, 3], torch.float
            Coordinates of the interaction points on the intersected wedges
        """

        sources = cast(sources, torch.float64)
        targets = cast(targets, torch.float64)

        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # [max_num_paths]
        valid_wedges_idx = torch.where(wedges_indices == -1, 0, wedges_indices)

        # [max_num_paths, 3]
        origins = self._wedges_origin[valid_wedges_idx]
        origins = cast(origins, torch.float64)
        # [1, 1, max_num_paths, 3]
        origins = expand_to_rank(origins, 4, 0)
        # [max_num_paths, 3]
        e_hat = self._wedges_e_hat[valid_wedges_idx]
        e_hat = cast(e_hat, torch.float64)
        # [1, 1, max_num_paths, 3]
        e_hat = expand_to_rank(e_hat, 4, 0)
        # [max_num_paths]
        wedges_length = self._wedges_length[valid_wedges_idx]
        wedges_length = cast(wedges_length, torch.float64)
        # [1, 1, max_num_paths]
        wedges_length = expand_to_rank(wedges_length, 3, 0)

        # Expand to broadcast with paths and sources/targets
        # [1, num_sources, 1, 3]
        sources = torch.unsqueeze(torch.unsqueeze(sources, dim=1), dim=0)
        # [num_targets, 1, 1, 3]
        targets = insert_dims(targets, 2, 1)
        # Sources/Targets vectors
        # [1, num_sources, max_num_paths, 3]
        u_t = origins - sources
        # [num_targets, 1, max_num_paths, 3]
        u_r = origins - targets

        # Quantites required for the computation of the interaction points
        # [1, num_sources, max_num_paths]
        a = dot(u_t, e_hat)
        # [num_targets, 1, max_num_paths]
        b = dot(u_r, e_hat)
        # [1, num_sources, max_num_paths]
        c = dot(u_t, u_t)
        # [num_targets, 1, max_num_paths]
        d = dot(u_r, u_r)

        # Quantites required for the computation of the interaction points
        # [num_targets, num_sources, max_num_paths]
        alpha = -torch.square(a) + torch.square(b) + c - d
        beta = 2. * (a * torch.square(b) - b * torch.square(a) + b * c - a * d)
        gamma = torch.square(b) * c - torch.square(a) * d

        # Normalized quantites to improve numerical preicion, only valid if
        # alpha != 0
        # [num_targets, num_sources, max_num_paths]
        beta_norm = divide_no_nan(beta, alpha)
        gamma_norm = divide_no_nan(gamma, alpha)
        delta = torch.square(beta_norm) - 4. * gamma_norm

        # Because of numerical imprecision, delta could be slighlty smaller than
        # 0
        # [num_targets, num_sources, max_num_paths]
        delta = torch.where(torch.lt(delta, torch.zeros_like(delta)),
                            torch.zeros_like(delta), delta)

        # Four possible outcomes depending on the value of the previous
        # quantities.
        # Values of t that minimizes the path length for each outcome.
        # [num_targets, num_sources, max_num_paths]
        t_min_1 = -a
        t_min_2 = -divide_no_nan(gamma, beta)
        t_min_3 = (-beta_norm + torch.sqrt(delta)) * 0.5
        t_min_4 = (-beta_norm - torch.sqrt(delta)) * 0.5
        # Condition for each outcome to be selected
        # If a == b and c == d, then set to t_min_1
        # [num_targets, num_sources, max_num_paths]
        cond_1 = torch.logical_and(torch.isclose(a, b),
                                   torch.isclose(c, d))
        # If cond_1 does not hold and alpha == 0, then set to t_min_2
        # [num_targets, num_sources, max_num_paths]
        cond_2 = torch.logical_and(torch.logical_not(cond_1),
                                   torch.isclose(alpha,
                                                 torch.zeros_like(alpha)))
        # If neither cond_1 nor cond_2 holds, then set to t_min_3 or t_min_4
        # depending on the signs of t+a and t+b
        # [num_targets, num_sources, max_num_paths]
        not_cond_12 = torch.logical_and(torch.logical_not(cond_1),
                                        torch.logical_not(cond_2))
        # [num_targets, num_sources, max_num_paths]
        t_min_3a = t_min_3 + a
        t_min_3b = t_min_3 + b
        # [num_targets, num_sources, max_num_paths]
        cond_3 = torch.logical_and(not_cond_12,
                                   torch.le(torch.sign(t_min_3a) * torch.sign(t_min_3b), 0.0))
        # If none of conditions 1, 2, or 3 are satisfied, then all is left is
        # t_min_4
        # [num_targets, num_sources, max_num_paths]
        cond_4 = torch.logical_and(not_cond_12, torch.logical_not(cond_3))
        # Assign t_min according to the previously computed conditions
        # [num_targets, num_sources, max_num_paths]
        t_min = torch.zeros_like(cond_1, dtype=torch.float64)
        t_min = torch.where(cond_1, t_min_1, t_min)
        t_min = torch.where(cond_2, t_min_2, t_min)
        t_min = torch.where(cond_3, t_min_3, t_min)
        t_min = torch.where(cond_4, t_min_4, t_min)

        # Mask paths for which the interaction point is not on the finite
        # wedge
        # [num_targets, num_sources, max_num_paths]
        mask_ = torch.logical_and(
            torch.ge(t_min, torch.zeros_like(t_min)),
            torch.le(t_min, wedges_length))
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = torch.where(mask_, wedges_indices, -1)

        # Interaction points
        # Expand to broadcast with coordinates
        # [num_targets, num_sources, max_num_paths, 1]
        t_min = torch.unsqueeze(t_min, dim=3)
        # [num_targets, num_sources, max_num_paths, 3]
        inter_point = origins + t_min * e_hat

        # Discard wedges with no valid paths
        # [max_num_paths]
        used_wedges = torch.nonzero(torch.ne(wedges_indices, -1).any(0).any(0), as_tuple=False)[:, 0]
        # [num_targets, num_sources, max_num_paths, 3]
        inter_point = inter_point[:, :, used_wedges]
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = wedges_indices[:, :, used_wedges]

        # Back to the required precision
        inter_point = cast(inter_point, self._rdtype)

        return wedges_indices, inter_point

    def _check_wedges_visibility(self, targets, sources, wedges_indices,
                                 vertices):
        r"""
        Discard the wedges that are not valid due to obstruction by updating the
        mask and removing the wedges related to no valid links.

        Input
        ------
        targets : [num_targets, 3], torch.float
            Coordinates of the targets.

        sources : [num_sources, 3], torch.float
            Coordinates of the sources.

        wedges_indices : [num_targets, num_sources, max_num_paths], torch.int
            Indices of the wedges that interacted with the diffracted paths

        vertices : [num_targets, num_sources, max_num_paths, 3], torch.float
            Coordinates of the interaction points on the intersected wedges

        Output
        -------
        wedges_indices : [num_targets, num_sources, max_num_paths], torch.int
            Indices of the wedges that interacted with the diffracted paths

        vertices : [num_targets, num_sources, max_num_paths, 3], torch.float
            Coordinates of the interaction points on the intersected wedges
        """

        max_num_paths = vertices.shape[2]
        num_sources = sources.shape[0]
        num_targets = targets.shape[0]

        # Broadcast sources and targets with wedges diffraction point
        # [1, num_sources, 1, 3]
        sources = torch.unsqueeze(sources, dim=0)
        sources = torch.unsqueeze(sources, dim=2)
        # [num_targets, num_sources, max_num_paths, 3]
        sources = torch.broadcast_to(sources, vertices.shape)
        # Flatten
        # batch_size = num_targets*num_sources*max_num_paths
        # [batch_size, 3]
        sources = reshape(sources, [-1, 3])
        # [num_targets, 1, 1, 3]
        targets = expand_to_rank(targets, vertices.dim(), 1)
        # [num_targets, num_sources, max_num_paths, 3]
        targets = torch.broadcast_to(targets, vertices.shape)
        # Flatten
        # [batch_size, 3]
        targets = reshape(targets, [-1, 3])

        # Flatten interaction points
        # [batch_size, 3]
        wedges_points = reshape(vertices, [-1, 3])

        # Check visibility between transmitter and wedge
        # Ray origin
        # d : [batch_size, 3]
        # maxt : [batch_size]
        d, maxt = normalize(wedges_points - sources, dim=1)
        # [batch_size]
        valid_t2w = torch.logical_not(self._test_obstruction(sources, d, maxt))

        # Check visibility between wedge and receiver
        # Ray origin
        # d : [batch_size, 3]
        # maxt : [batch_size]
        d, maxt = normalize(wedges_points - targets, dim=1)
        # [batch_size]
        valid_w2r = torch.logical_not(self._test_obstruction(targets, d, maxt))

        # Mask obstructed wedges
        # [batch_size]
        valid = torch.logical_and(valid_t2w, valid_w2r)
        # [num_targets, num_sources, max_num_paths]
        valid = reshape(valid, [num_targets, num_sources, max_num_paths])
        # Set wedge indices of blocked paths to -1
        wedges_indices = torch.where(valid, wedges_indices, -1)

        # Discard wedges not involved in any link
        # [max_num_paths]
        used_wedges = torch.nonzero(torch.ne(wedges_indices, -1).any(0).any(0), as_tuple=False)[:, 0]
        # [num_targets, num_sources, max_num_paths, 3]
        vertices = vertices[:, :, used_wedges]
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = wedges_indices[:, :, used_wedges]

        return wedges_indices, vertices

    def _gather_valid_diff_paths(self, paths):
        r"""
        Extracts only valid diffracted paths to reduce memory consumption when
        having multiple links with different number of valid paths.

        Input
        ------
        paths : :class:`~torchrf.rt.Paths`
            Paths to update

        Output
        ------
        paths : :class:`~torchrf.rt.Paths`
            Updated paths
        """

        # [num_targets, num_sources, max_num_candidates]
        wedges_indices = paths.objects[0]
        # [num_targets, num_sources, max_num_candidates, 3]
        vertices = paths.vertices[0]
        # [num_sources, 3]
        sources = paths.sources
        # [num_targets, 3]
        targets = paths.targets

        num_sources = sources.shape[0]
        num_targets = targets.shape[0]

        # [num_targets, num_sources, max_num_candidates]
        valid = torch.ne(wedges_indices, -1)

        # [num_targets, num_sources]
        num_paths = torch.sum(cast(valid, torch.int32), dim=-1)
        # Maximum number of valid paths
        # ()
        max_num_paths = torch.max(num_paths)

        # Build indices for keeping only valid path
        # [num_valid_paths, 3]
        gather_indices = torch.nonzero(valid, as_tuple=False)
        # [num_targets, num_sources, max_num_candidates]
        path_indices = torch.cumsum(valid.to(torch.int32), dim=-1)
        # [num_valid_paths]
        path_indices = gather_nd(path_indices, gather_indices) - 1
        scatter_indices = torch.permute(gather_indices, [1, 0])
        if not size(scatter_indices) == 0:
            scatter_indices[2] = path_indices
        # [num_valid_paths, 3]
        scatter_indices = torch.permute(scatter_indices, [1, 0])
        gather_indices = torch.nonzero(valid, as_tuple=False)

        # Mask of valid paths
        # [num_targets, num_sources, max_num_paths]
        mask = torch.full([num_targets, num_sources, max_num_paths], False)
        mask = scatter_nd_update(mask, scatter_indices,
                                 torch.full([scatter_indices.shape[0]], True))

        # Locations of the interactions
        # [num_targets, num_sources, max_num_paths, 3]
        valid_vertices = torch.zeros([num_targets, num_sources, max_num_paths, 3],
                                     dtype=self._rdtype)
        # [total_num_valid_paths, 3]
        vertices = gather_nd(vertices, gather_indices)
        valid_vertices = scatter_nd_update(valid_vertices,
                                           scatter_indices, vertices)

        # Intersected wedges
        # [num_targets, num_sources, max_num_paths]
        valid_wedges_indices = torch.full([num_targets, num_sources,
                                           max_num_paths], -1)
        # [total_num_valid_paths]
        wedges_indices = gather_nd(wedges_indices, gather_indices)
        valid_wedges_indices = scatter_nd_update(valid_wedges_indices,
                                                 scatter_indices, wedges_indices)

        # [1, num_targets, num_sources, max_num_candidates]
        paths.objects = torch.unsqueeze(valid_wedges_indices, dim=0)
        # [1, num_targets, num_sources, max_num_candidates, 3]
        paths.vertices = torch.unsqueeze(valid_vertices, dim=0)
        # [num_targets, num_sources, max_num_candidates]
        paths.mask = mask

        return paths

    def _compute_diffraction_transition_matrices(self,
                                                 relative_permittivity,
                                                 scattering_coefficient,
                                                 paths,
                                                 paths_tmp):
        # pylint: disable=line-too-long
        """
        Compute the transition matrices for diffracted rays.

        Input
        ------
        relative_permittivity : [num_shape], torch.complex
            Tensor containing the complex relative permittivity of all shape
            in the scene

        scattering_coefficient : [num_shape], torch.float
            Tensor containing the scattering coefficients of all shapes

        paths : :class:`~torchrf.rt.Paths`
            Paths to update

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Addtional quantities required for paths computation

        Output
        ------
        paths : :class:`~torchrf.rt.Paths`
            Updated paths
        """

        mask = paths.mask
        targets = paths.targets
        sources = paths.sources
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        normals = paths_tmp.normals

        def f(x):
            """F(x) Eq.(88) in [ITUR_P526]
            """
            sqrt_x = torch.sqrt(x)
            sqrt_pi_2 = torch.sqrt(cast(PI, x.dtype) / 2.)

            # Fresnel integral
            arg = sqrt_x / sqrt_pi_2
            s = fresnel_sin(arg)
            c = fresnel_cos(arg)
            f = torch.complex(s, c)

            zero = cast(0, x.dtype)
            one = cast(1, x.dtype)
            two = cast(2, f.dtype)
            factor = torch.complex(sqrt_pi_2 * sqrt_x, zero)
            factor = factor * torch.exp(torch.complex(zero, x))
            res = torch.complex(one, one) - two * f

            return factor * res

        wavelength = self._scene.wavelength
        k = 2. * PI / wavelength

        # [num_targets, num_sources, max_num_paths, 3]
        diff_points = paths.vertices[0]
        # [num_targets, num_sources, max_num_paths]
        wedges_indices = paths.objects[0]

        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # [num_targets, num_sources, max_num_paths]
        valid_wedges_idx = torch.where(wedges_indices == -1, 0, wedges_indices)

        # Normals
        # [num_targets, num_sources, max_num_paths, 2, 3]
        normals = self._wedges_normals[valid_wedges_idx]

        # Compute the wedges angle
        # [num_targets, num_sources, max_num_paths]
        wedges_angle = PI - torch.acos(dot(normals[..., 0, :], normals[..., 1, :]))
        n = (2. * PI - wedges_angle) / PI

        # [num_targets, num_sources, max_num_paths, 3]
        e_hat = self._wedges_e_hat[valid_wedges_idx]

        # Reshape sources and targets
        # [1, num_sources, 1, 3]
        sources = reshape(sources, [1, -1, 1, 3])
        # [num_targets, 1, 1, 3]
        targets = reshape(targets, [-1, 1, 1, 3])

        # Extract surface normals
        # [num_targets, num_sources, max_num_paths, 3]
        n_0_hat = normals[..., 0, :]
        # [num_targets, num_sources, max_num_paths, 3]
        n_n_hat = normals[..., 1, :]

        # Relative permitivities
        # [num_targets, num_sources, max_num_paths, 2]
        objects_indices = self._wedges_objects[valid_wedges_idx]
        # [num_targets, num_sources, max_num_paths, 2]
        etas = relative_permittivity[objects_indices]
        # [num_targets, num_sources, max_num_paths]
        eta_0 = etas[..., 0]
        eta_n = etas[..., 1]

        # Get scattering coefficients
        # [num_targets, num_sources, max_num_paths, 2]
        scattering_coefficient = scattering_coefficient[objects_indices]
        # [num_targets, num_sources, max_num_paths]
        scattering_coefficient_0 = scattering_coefficient[..., 0]
        scattering_coefficient_n = scattering_coefficient[..., 1]

        # Compute s_prime_hat, s_hat, s_prime, s
        # s_prime_hat : [num_targets, num_sources, max_num_paths, 3]
        # s_prime : [num_targets, num_sources, max_num_paths]
        s_prime_hat, s_prime = normalize(diff_points - sources)
        # s_hat : [num_targets, num_sources, max_num_paths, 3]
        # s : [num_targets, num_sources, max_num_paths]
        s_hat, s = normalize(targets - diff_points)

        # Compute phi_prime_hat, beta_0_prime_hat, phi_hat, beta_0_hat
        # [num_targets, num_sources, max_num_paths, 3]
        phi_prime_hat, _ = normalize(cross(s_prime_hat, e_hat))
        # [num_targets, num_sources, max_num_paths, 3]
        beta_0_prime_hat = cross(phi_prime_hat, s_prime_hat)

        # [num_targets, num_sources, max_num_paths, 3]
        phi_hat_, _ = normalize(-cross(s_hat, e_hat))
        beta_0_hat = cross(phi_hat_, s_hat)

        # Compute tangent vector t_0_hat
        # [num_targets, num_sources, max_num_paths, 3]
        t_0_hat = cross(n_0_hat, e_hat)

        # Compute s_t_prime_hat and s_t_hat
        # [num_targets, num_sources, max_num_paths, 3]
        s_t_prime_hat, _ = normalize(s_prime_hat
                                     - dot(s_prime_hat, e_hat, keepdim=True) * e_hat)
        # [num_targets, num_sources, max_num_paths, 3]
        s_t_hat, _ = normalize(s_hat - dot(s_hat, e_hat, keepdim=True) * e_hat)

        # Compute phi_prime and phi
        # [num_targets, num_sources, max_num_paths]
        phi_prime = PI - \
                    (PI - acos_diff(-dot(s_t_prime_hat, t_0_hat))) \
                    * torch.sign(-dot(s_t_prime_hat, n_0_hat))
        # [num_targets, num_sources, max_num_paths]
        phi = PI - (PI - acos_diff(dot(s_t_hat, t_0_hat))) \
              * torch.sign(dot(s_t_hat, n_0_hat))

        # Compute field component vectors for reflections at both surfaces
        # [num_targets, num_sources, max_num_paths, 3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_0, e_i_p_0, e_r_s_0, e_r_p_0 = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_0_hat,  # *sign(-dot(s_t_prime_hat, n_0_hat, keepdim=True)),
            SolverBase.EPSILON
        )
        # [num_targets, num_sources, max_num_paths, 3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_n, e_i_p_n, e_r_s_n, e_r_p_n = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_n_hat,  # *sign(-dot(s_t_prime_hat, n_n_hat, keepdim=True)),
            SolverBase.EPSILON
        )

        # Compute Fresnel reflection coefficients for 0- and n-surfaces
        # [num_targets, num_sources, max_num_paths]
        r_s_0, r_p_0 = reflection_coefficient(eta_0, torch.abs(torch.sin(phi_prime)))
        r_s_n, r_p_n = reflection_coefficient(eta_n, torch.abs(torch.sin(n * PI - phi)))

        # Multiply the reflection coefficients with the
        # corresponding reflection reduction factor
        reduction_factor_0 = torch.sqrt(1 - scattering_coefficient_0 ** 2)
        reduction_factor_0 = torch.complex(reduction_factor_0,
                                           torch.zeros_like(reduction_factor_0))
        reduction_factor_n = torch.sqrt(1 - scattering_coefficient_n ** 2)
        reduction_factor_n = torch.complex(reduction_factor_n,
                                           torch.zeros_like(reduction_factor_n))
        r_s_0 *= reduction_factor_0
        r_p_0 *= reduction_factor_0
        r_s_n *= reduction_factor_n
        r_p_n *= reduction_factor_n

        # Compute matrices R_0, R_n
        # [num_targets, num_sources, max_num_paths, 2, 2]
        w_i_0 = component_transform(phi_prime_hat,
                                    beta_0_prime_hat,
                                    e_i_s_0,
                                    e_i_p_0)
        w_i_0 = torch.complex(w_i_0, torch.zeros_like(w_i_0))
        # [num_targets, num_sources, max_num_paths, 2, 2]
        w_r_0 = component_transform(e_r_s_0,
                                    e_r_p_0,
                                    phi_hat_,
                                    beta_0_hat)
        w_r_0 = torch.complex(w_r_0, torch.zeros_like(w_r_0))
        # [num_targets, num_sources, max_num_paths, 2, 1]
        r_0 = torch.unsqueeze(torch.stack([r_s_0, r_p_0], -1), -1) * w_i_0
        # [num_targets, num_sources, max_num_paths, 2, 1]
        r_0 = -torch.matmul(w_r_0, r_0)

        # [num_targets, num_sources, max_num_paths, 2, 2]
        w_i_n = component_transform(phi_prime_hat,
                                    beta_0_prime_hat,
                                    e_i_s_n,
                                    e_i_p_n)
        w_i_n = torch.complex(w_i_n, torch.zeros_like(w_i_n))
        # [num_targets, num_sources, max_num_paths, 2, 2]
        w_r_n = component_transform(e_r_s_n,
                                    e_r_p_n,
                                    phi_hat_,
                                    beta_0_hat)
        w_r_n = torch.complex(w_r_n, torch.zeros_like(w_r_n))
        # [num_targets, num_sources, max_num_paths, 2, 1]
        r_n = torch.unsqueeze(torch.stack([r_s_n, r_p_n], -1), -1) * w_i_n
        # [num_targets, num_sources, max_num_paths, 2, 1]
        r_n = -torch.matmul(w_r_n, r_n)

        # Compute D_1, D_2, D_3, D_4
        # [num_targets, num_sources, max_num_paths]
        phi_m = phi - phi_prime
        phi_p = phi + phi_prime

        # [num_targets, num_sources, max_num_paths]
        cot_1 = cot((PI + phi_m) / (2 * n))
        cot_2 = cot((PI - phi_m) / (2 * n))
        cot_3 = cot((PI + phi_p) / (2 * n))
        cot_4 = cot((PI - phi_p) / (2 * n))

        def n_p(beta, n):
            return torch.round((beta + PI) / (2. * n * PI))

        def n_m(beta, n):
            return torch.round((beta - PI) / (2. * n * PI))

        def a_p(beta, n):
            return 2 * torch.cos((2. * n * PI * n_p(beta, n) - beta) / 2.) ** 2

        def a_m(beta, n):
            return 2 * torch.cos((2. * n * PI * n_m(beta, n) - beta) / 2.) ** 2

        d_mul = (-torch.exp(-1j*torch.tensor(PI, dtype=self._rdtype)/4.) /
                 (2 * n * torch.sqrt(2*torch.tensor(PI, dtype=self._rdtype)*k)).to(self._dtype))

        # [num_targets, num_sources, max_num_paths]
        ell = s_prime * s / (s_prime + s)

        # [num_targets, num_sources, max_num_paths]
        cot_1 = torch.complex(cot_1, torch.zeros_like(cot_1))
        cot_2 = torch.complex(cot_2, torch.zeros_like(cot_2))
        cot_3 = torch.complex(cot_3, torch.zeros_like(cot_3))
        cot_4 = torch.complex(cot_4, torch.zeros_like(cot_4))
        d_1 = d_mul * cot_1 * f(k * ell * a_p(phi_m, n))
        d_2 = d_mul * cot_2 * f(k * ell * a_m(phi_m, n))
        d_3 = d_mul * cot_3 * f(k * ell * a_p(phi_p, n))
        d_4 = d_mul * cot_4 * f(k * ell * a_m(phi_p, n))

        # [num_targets, num_sources, max_num_paths, 1, 1]
        d_1 = d_1[..., None, None]
        d_2 = d_2[..., None, None]
        d_3 = d_3[..., None, None]
        d_4 = d_4[..., None, None]

        # [num_targets, num_sources, max_num_paths]
        spreading_factor = torch.sqrt(1.0 / (s * s_prime * (s_prime + s)))
        spreading_factor = torch.complex(spreading_factor,
                                         torch.zeros_like(spreading_factor))
        # [num_targets, num_sources, max_num_paths, 1, 1]
        spreading_factor = reshape(spreading_factor, d_1.shape)

        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = (d_1 + d_2) * torch.eye(2, 2, dtype=self._dtype)[None, None, None]
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t += d_3 * r_n + d_4 * r_0
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t *= -spreading_factor

        # Convert from/to GCS
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r

        mat_from_gcs = component_transform(
            theta_hat(theta_t, phi_t), phi_hat(phi_t),
            phi_prime_hat, beta_0_prime_hat)
        mat_from_gcs = torch.complex(mat_from_gcs, torch.zeros_like(mat_from_gcs))

        mat_to_gcs = component_transform(phi_hat_, beta_0_hat,
                                         theta_hat(theta_r, phi_r), phi_hat(phi_r))
        mat_to_gcs = torch.complex(mat_to_gcs, torch.zeros_like(mat_to_gcs))

        mat_t = torch.matmul(mat_t, mat_from_gcs)
        mat_t = torch.matmul(mat_to_gcs, mat_t)

        # Set invalid paths to 0
        # Expand masks to broadcast with the field components
        # [num_targets, num_sources, max_num_paths, 1, 1]
        mask_ = expand_to_rank(mask, 5, axis=3)
        # Zeroing coefficients corresponding to non-valid paths
        # [num_targets, num_sources, max_num_paths, 2]
        mat_t = torch.where(mask_, mat_t, torch.zeros_like(mat_t))
        paths_tmp.mat_t = mat_t

        return paths

    ##################################################################
    # Methods used for computing the scattered paths
    ##################################################################

    def _scat_test_rx_blockage(self, targets, sources, candidates, hit_points):
        r"""
        Test if the LoS between the hit points and the target is blocked.
        Blocked paths are masked out.

        Input
        -----
        targets : [num_targets, 3], torch.float
            Coordinates of the targets.

        sources : [num_sources, 3], torch.float
            Coordinates of the sources.

        candidates : [max_depth, num_sources, num_paths_per_source], int
            Sequence of primitives hit at `hit_points`.

        hit_points : [max_depth, num_sources, num_paths_per_source, 3], torch.float
            Intersection points.

        Output
        -------
        paths : :class:`~torchrf.rt.Paths`
            Structure storing the scattered paths.

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Addtional quantities required for paths computation
        """

        num_sources = candidates.shape[1]
        num_targets = targets.shape[0]
        max_depth = candidates.shape[0]

        # Expand for broadcasting with max_depth, num_sources, and num_paths
        # [1, num_targets, 1, 1, 3]
        targets_ = torch.unsqueeze(insert_dims(targets, 2, 1), dim=0)

        # Build the rays for shooting
        # Origins
        # [max_depth, num_targets, num_targets, num_paths, 3]
        hit_points = torch.unsqueeze(hit_points, dim=1).repeat(1, num_targets, 1, 1, 1)
        # [max_depth * num_targets * num_sources * num_paths, 3]
        ray_origins = reshape(hit_points, [-1, 3])
        # Directions
        # [max_depth, num_targets, num_sources, num_paths, 3]
        ray_directions, rays_lengths = normalize(targets_ - hit_points)
        # [max_depth * num_targets * num_sources * num_paths, 3]
        ray_directions = reshape(ray_directions, [-1, 3])
        # [max_depth * num_targets * num_sources * num_paths]
        rays_lengths = reshape(rays_lengths, [-1])

        # Test for blockage
        # [max_depth * num_targets * num_sources * num_paths]
        blocked = self._test_obstruction(ray_origins, ray_directions,
                                         rays_lengths)
        # [max_depth, num_targets, num_sources, num_paths]
        blocked = reshape(blocked,
                          [max_depth, num_targets, num_sources, -1])

        # Mask blocked paths
        # [max_depth, num_targets, num_sources, num_paths]
        candidates = torch.unsqueeze(candidates, dim=1).repeat(1, num_targets, 1, 1)
        # [max_depth, num_targets, num_sources, num_paths]
        prefix_mask = torch.logical_and(~blocked, torch.ne(candidates, -1))

        # Optimize tensor size by ensuring that the length of the paths
        # dimension correspond to the maximum number of paths over all links

        # Keep a path if at least one of its prefix is valid
        # [num_targets, num_sources, num_paths]
        prefix_mask_ = torch.any(prefix_mask, dim=0)
        prefix_mask_int_ = cast(prefix_mask_, torch.int32)

        # Maximum number of valid paths over all links
        # [num_targets, num_sources]
        num_paths = torch.sum(prefix_mask_int_, dim=-1)
        # Maximum number of paths
        # ()
        max_num_paths = torch.max(num_paths)

        # [num_valid_paths, 3]
        gather_indices = torch.nonzero(prefix_mask_, as_tuple=False)
        # To build the indices of the paths in the tensor with optimized size,
        # the path dimension is indexed by counting the valid path in the order
        # in which they appear
        # [num_targets, num_sources, num_paths]
        path_indices = torch.cumsum(prefix_mask_int_, dim=-1)
        # [num_valid_paths, 3]
        path_indices = gather_nd(path_indices, gather_indices) - 1
        # The indices used to scatter the valid paths in the tensors with
        # optimized size are built by replacing the index of the paths by
        # the previous ones, which leads to skipping the invalid paths
        # [3, num_valid_paths]
        scatter_indices = torch.permute(gather_indices, [1, 0])
        if not size(scatter_indices) == 0:
            scatter_indices[2] = path_indices
        # [num_valid_paths, 3]
        scatter_indices = torch.permute(scatter_indices, [1, 0])
        gather_indices = torch.nonzero(prefix_mask_, as_tuple=False)

        # Mask of valid paths
        # [num_targets, num_sources, max_num_paths]
        opt_prefix_mask = torch.full([max_depth, num_targets, num_sources,
                                      max_num_paths], False)
        # Locations of the interactions
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        opt_hit_points = torch.zeros([max_depth, num_targets, num_sources,
                                      max_num_paths, 3], dtype=self._rdtype)
        # [max_depth, num_targets, num_sources, max_num_paths]
        opt_candidates = torch.full([max_depth, num_targets, num_sources,
                                     max_num_paths], -1)

        if max_depth > 0:

            for depth in torch.arange(max_depth, dtype=torch.int64):
                # Indices for storing the valid items for this depth
                scatter_indices_ = F.pad(scatter_indices, (1, 0, 0, 0), mode='constant', value=depth)

                # Prefix mask
                # [num_targets, num_sources, num_samples]
                prefix_mask_ = prefix_mask[depth]
                # [num_valid_paths, 3]
                prefix_mask_ = gather_nd(prefix_mask_, gather_indices)
                # Store the valid intersection points
                # [max_depth, num_targets, num_sources, max_num_paths]
                opt_prefix_mask = scatter_nd_update(opt_prefix_mask,
                                                    scatter_indices_, prefix_mask_)

                # Location of the interactions
                # [num_targets, num_sources, num_samples, 3]
                hit_points_ = hit_points[depth]
                # [num_valid_paths, 3]
                hit_points_ = gather_nd(hit_points_, gather_indices)
                # Store the valid intersection points
                # [max_depth, num_targets, num_sources, max_num_paths, 3]
                opt_hit_points = scatter_nd_update(opt_hit_points,
                                                   scatter_indices_, hit_points_)

                # Intersected primitives
                # [num_targets, num_sources, num_samples]
                candidates_ = candidates[depth]
                # [num_valid_paths, 3]
                candidates_ = gather_nd(candidates_, gather_indices)
                # Store the valid intersection points
                # [max_depth, num_targets, num_sources, max_num_paths]
                opt_candidates = scatter_nd_update(opt_candidates,
                                                   scatter_indices_, candidates_)

        # Gather normals to the intersected primitives
        # Note: They are not oriented in the direction of the incoming wave.
        # This is done later.
        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flagged as active.
        # [max_depth, num_targets, num_sources, max_num_paths]
        opt_candidates_ = torch.where(opt_candidates == -1, 0, opt_candidates)
        # [max_depth, num_targets, num_sources, num_paths, 3]
        normals = self._normals[opt_candidates_]

        # Map primitives to the corresponding objects
        # Add a dummy entry to primitives_2_objects with value -1.
        # [num_samples + 1]
        primitives_2_objects = F.pad(self._primitives_2_objects, (0, 1), value=-1)
        # Replace all -1 by num_samples
        num_primitives = self._primitives_2_objects.shape[0]
        # [max_depth, num_targets, num_sources, max_num_paths]
        opt_candidates_ = torch.where(opt_candidates == -1, num_primitives, opt_candidates)
        # [max_depth, num_targets, num_sources, max_num_paths]
        objects = primitives_2_objects[opt_candidates_]

        # Create and return the the objects storing the scattered paths
        paths = Paths(sources=sources,
                      targets=targets,
                      scene=self._scene,
                      types=Paths.SCATTERED)
        paths.vertices = opt_hit_points
        paths.objects = objects

        paths_tmp = PathsTmpData(sources, targets, self._dtype)
        paths_tmp.normals = normals
        paths_tmp.scat_prefix_mask = opt_prefix_mask
        paths_tmp.scat_prefix_k_s, _ = normalize(targets_ - opt_hit_points)

        return paths, paths_tmp

    def _scat_discard_crossing_paths(self, paths, paths_tmp, scat_keep_prob):
        r"""
        Discards paths:

        - for which the scattered ray is crossing the intersected
        primitive, and

        - randomly with probability `` 1 - scat_keep_prob``.

        Input
        ------
        paths : :class:`~torchrf.rt.Paths`
            Structure storing the scattered paths.

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Addtional quantities required for paths computation

        scat_keep_prob : torch.float
            Probablity of keeping a valid scattered paths.
            Must be in )0,1).

        Output
        -------
        paths : :class:`~torchrf.rt.Paths`
            Updates paths.

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Updated addtional quantities required for paths computation
        """

        theta_t = paths.theta_t
        phi_t = paths.phi_t
        objects = paths.objects
        vertices = paths.vertices

        normals = paths_tmp.normals
        mask = paths_tmp.scat_prefix_mask
        k_i = paths_tmp.k_i
        k_r = paths_tmp.k_r
        k_s = paths_tmp.scat_prefix_k_s
        total_distance = paths_tmp.total_distance

        max_depth = vertices.shape[0]

        # Ensure the normals point in the same direction as -k_i
        # [max_depth, num_targets, num_sources, max_num_paths, 1]
        s = -torch.sign(dot(k_i[:max_depth], normals, keepdim=True))
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        normals = normals * s

        # Mask paths for which k_s does not point in the same direction as the
        # normal
        # [max_depth, num_targets, num_sources, max_num_paths]
        same_side = dot(normals, k_s) > SolverBase.EPSILON
        # [max_depth, num_targets, num_sources, max_num_paths]
        mask = torch.logical_and(mask, same_side)

        # Keep valid path with probability `scat_keep_prob`
        # [max_depth, num_targets, num_sources, max_num_paths]
        random_mask = random_uniform(mask.shape, 0., 1., self._rdtype)
        # [max_depth, num_targets, num_sources, max_num_paths]
        random_mask = torch.lt(random_mask, scat_keep_prob)
        # [max_depth, num_targets, num_sources, max_num_paths]
        mask = torch.logical_and(mask, random_mask)

        # Discard paths invalid for all links
        valid_indices = torch.nonzero(mask.any(0).any(0).any(0), as_tuple=False)[:, 0]
        # [num_targets, num_sources, max_num_paths]]
        theta_t = theta_t[:, :, valid_indices]
        phi_t = phi_t[:, :, valid_indices]
        # [max_depth, num_targets, num_sources, max_num_paths]
        objects = objects[:, :, :, valid_indices]
        mask = mask[:, :, :, valid_indices]
        total_distance = total_distance[:, :, :, valid_indices]
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        normals = normals[:, :, :, valid_indices]
        k_r = k_r[:, :, :, valid_indices]
        k_i = k_i[:, :, :, valid_indices]
        vertices = vertices[:, :, :, valid_indices]

        paths.theta_t = theta_t
        paths.phi_t = phi_t
        paths.vertices = vertices
        paths.objects = objects

        paths_tmp.scat_prefix_mask = mask
        paths_tmp.k_i = k_i
        paths_tmp.k_r = k_r
        paths_tmp.k_tx = k_i[0]
        paths_tmp.total_distance = total_distance
        paths_tmp.normals = normals

        return paths, paths_tmp

    def _scat_prefixes_2_paths(self, paths, paths_tmp):
        """
        Extracts valid prefixes as invidual paths.

        Input
        ------
        paths : :class:`~torchrf.rt.Paths`
            Structure storing the scattered paths.

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Addtional quantities required for paths computation

        Output
        -------
        paths : :class:`~torchrf.rt.Paths`
            Updates paths.

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Updated addtional quantities required for paths computation
        """

        # [max_depth, num_targets, num_sources, max_num_paths]
        prefix_mask = paths_tmp.scat_prefix_mask
        prefix_mask_int = cast(prefix_mask, torch.int32)
        # [max_depth, num_targets, num_sources, max_num_paths, 2, 2]
        prefix_mat_t = paths_tmp.scat_prefix_mat_t
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        prefix_vertices = paths.vertices
        # [max_depth, num_targets, num_sources, max_num_paths]
        prefix_objects = paths.objects
        # [num_targets, num_sources, max_num_paths]
        prefix_theta_t = paths.theta_t
        # [num_targets, num_sources, max_num_paths]
        prefix_phi_t = paths.phi_t
        # [max_depth, num_targets, num_sources, num_paths, 3]
        prefix_normals = paths_tmp.normals
        # [max_depth, num_targets, num_sources, num_paths, 3]
        prefix_k_i = paths_tmp.k_i[:-1]
        # [max_depth, num_targets, num_sources, num_paths]
        prefix_distances = paths_tmp.total_distance
        # [num_targets, num_sources, max_num_paths, 3]
        prefix_ktx = paths_tmp.k_tx

        max_depth = prefix_mask.shape[0]
        max_depth64 = cast(max_depth, torch.int64)
        num_targets = prefix_mask.shape[1]
        num_sources = prefix_mask.shape[2]

        # Number of paths for each link and depth
        # [max_depth, num_targets, num_sources]
        paths_count = torch.sum(prefix_mask_int, dim=3)
        # Maximum number of paths for each depth over all the links
        # [max_depth]
        path_count_depth = torch.amax(paths_count, dim=(1, 2))
        # Upper bound on the total number of paths
        # ()
        max_num_paths = torch.sum(path_count_depth).item()

        # [num_valid_paths, 4]
        gather_indices = torch.nonzero(prefix_mask, as_tuple=False)
        # To build the indices of the paths in the tensor with optimized size,
        # the path dimension is indexed by counting the valid path in the order
        # in which they appear
        # [max_depth, num_targets, num_sources, num_paths]
        path_indices = torch.cumsum(prefix_mask_int, dim=-1)
        # [num_valid_paths, 4]
        path_indices = gather_nd(path_indices, gather_indices) - 1
        scatter_indices = torch.permute(gather_indices, [1, 0])
        if not size(scatter_indices) == 0:
            scatter_indices[3] = path_indices
        # [num_valid_paths, 3]
        scatter_indices = torch.permute(scatter_indices, [1, 0])
        gather_indices = torch.nonzero(prefix_mask, as_tuple=False)

        # Create the final tensors to update
        # [num_targets, num_sources, max_num_paths]
        mask = torch.full([num_targets, num_sources, max_num_paths], False)
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = torch.zeros(num_targets, num_sources, max_num_paths, 2, 2, dtype=self._dtype)
        # This tensor is created transposed as paths
        # are added with all the objects hit along the paths.
        # [num_targets, num_sources, max_num_paths, max_depth, 3]
        vertices = torch.zeros(num_targets, num_sources, max_num_paths, max_depth, 3, dtype=self._rdtype)
        # Last vertices that were hit
        # [num_targets, num_sources, max_num_paths, 3]
        last_vertices = torch.zeros(num_targets, num_sources, max_num_paths, 3, dtype=self._rdtype)
        # Objects that were hit. This tensor is created transposed as paths
        # are added with all the objects hit along the paths.
        # [num_targets, num_sources, max_num_paths, max_depth]
        objects = torch.full([num_targets, num_sources, max_num_paths, max_depth],
                             -1)
        # Last objects that were hit
        # [num_targets, num_sources, max_num_paths]
        last_objects = torch.full([num_targets, num_sources, max_num_paths], -1)
        # [num_targets, num_sources, max_num_paths]
        theta_t = torch.zeros(num_targets, num_sources, max_num_paths,
                              dtype=self._rdtype)
        # [num_targets, num_sources, max_num_paths]
        phi_t = torch.zeros(num_targets, num_sources, max_num_paths,
                            dtype=self._rdtype)
        # Normal to the last intersected objects
        # [num_targets, num_sources, max_num_paths, 3]
        last_normals = torch.zeros(num_targets, num_sources, max_num_paths, 3,
                                   dtype=self._rdtype)
        # Direction of incidence at the last interaction point
        # [num_targets, num_sources, max_num_paths, 3]
        last_k_i = torch.zeros(num_targets, num_sources, max_num_paths, 3,
                               dtype=self._rdtype)
        # Distance from the sources to the last interaction point
        # [num_targets, num_sources, max_num_paths]
        last_distance = torch.zeros(num_targets, num_sources, max_num_paths,
                                    dtype=self._rdtype)
        # [num_targets, num_sources, max_num_paths, 3]
        k_tx = torch.zeros(num_targets, num_sources, max_num_paths, 3,
                           dtype=self._rdtype)

        # Need to transpose these tensors in order to gather paths from them
        # with all the interactions, i.e., extract the entire "max_depth"
        # dimension.
        # [max_depth, num_targets, num_sources, max_num_paths]
        prefix_objects_tp = torch.permute(prefix_objects, [1, 2, 3, 0])
        # [num_targets, num_sources, max_num_paths, max_depth, 3]
        prefix_vertices_tp = torch.permute(prefix_vertices, [1, 2, 3, 0, 4])

        # We sequentially add the prefixes for each depth value.
        # To avoid overwriting the paths scattered at the previous
        # iterations, we incremdent the path index by the maximum number
        # of paths over all links, cumulated over the iterations.
        path_ind_incr = 0
        for depth in torch.arange(max_depth, dtype=torch.int64):
            # Indices of valid paths with depth d
            # [num_valid_paths with depth=depth, 4]
            gather_indices_ = gather_indices[torch.where(gather_indices[:, 0] == depth)[0]]
            # Depth is not needed for some tensors
            # [num_valid_paths with depth=depth, 3]
            gather_indices_nd_ = gather_indices_[:, 1:]

            # Indices for scattering the results in the target tensor
            # [num_valid_paths with depth=depth, 4]
            scatter_indices_ = scatter_indices[torch.where(scatter_indices[:, 0] == depth)[0]]

            # [1, 4]
            path_ind_incr_ = cast([0, 0, 0, path_ind_incr], torch.int64)
            # [num_valid_paths with depth=depth, 4]
            scatter_indices_ = scatter_indices_ + path_ind_incr_
            # Depth is not needed for some tensors
            # [num_valid_paths with depth=depth, 3]
            scatter_indices_nd_ = scatter_indices_[:, 1:]
            # Prepare for next iteration
            path_ind_incr = path_ind_incr + path_count_depth[depth]

            # Update the tensors

            # Mask
            # [num_valid_paths with depth=depth]
            prefix_mask_ = torch.full([scatter_indices_nd_.shape[0]], True)
            # [num_targets, num_sources, max_num_paths]
            mask = scatter_nd_update(mask, scatter_indices_nd_,
                                     prefix_mask_)

            # Transition matrices
            prefix_mat_t_ = gather_nd(prefix_mat_t, gather_indices_)
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = scatter_nd_update(mat_t, scatter_indices_nd_,
                                      prefix_mat_t_)

            # Vertices
            prefix_vertices_ = gather_nd(prefix_vertices_tp,
                                         gather_indices_nd_)
            # [num_targets, num_sources, max_num_paths, max_depth, 3]
            vertices = scatter_nd_update(vertices,
                                         scatter_indices_nd_,
                                         prefix_vertices_)

            # Last vertex
            prefix_vertex = gather_nd(prefix_vertices, gather_indices_)
            # [num_targets, num_sources, max_num_paths, 3]
            last_vertices = scatter_nd_update(last_vertices,
                                              scatter_indices_nd_,
                                              prefix_vertex)

            # Objects
            # [num_paths, max_depth]
            objects_ = gather_nd(prefix_objects_tp, gather_indices_nd_)
            # Only keep the prefix of length depth
            # [num_paths, depth]
            objects_ = objects_[:, :depth + 1]
            # [num_paths, max_depth]
            objects_ = F.pad(objects_, (0, max_depth64 - depth - 1), value=-1)
            # [num_targets, num_sources, max_num_paths, max_depth]
            objects = scatter_nd_update(objects, scatter_indices_nd_,
                                        objects_)

            # Last hit objects
            objects_ = gather_nd(prefix_objects, gather_indices_)
            # [num_targets, num_sources, max_num_paths]
            last_objects = scatter_nd_update(last_objects,
                                             scatter_indices_nd_,
                                             objects_)

            # Azimuth of departure
            phi_t_ = gather_nd(prefix_phi_t, gather_indices_nd_)
            # [max_depth, num_targets, num_sources, max_num_paths]
            phi_t = scatter_nd_update(phi_t, scatter_indices_nd_,
                                      phi_t_)

            # Elevation of departure
            theta_t_ = gather_nd(prefix_theta_t, gather_indices_nd_)
            # [max_depth, num_targets, num_sources, max_num_paths]
            theta_t = scatter_nd_update(theta_t, scatter_indices_nd_,
                                        theta_t_)

            # Normals at the last object
            normals_ = gather_nd(prefix_normals, gather_indices_)
            # [num_targets, num_sources, max_num_paths, 3]
            last_normals = scatter_nd_update(last_normals,
                                             scatter_indices_nd_,
                                             normals_)

            # Direction of incidence at the last interaction point
            k_i_ = gather_nd(prefix_k_i, gather_indices_)
            # [num_targets, num_sources, max_num_paths, 3]
            last_k_i = scatter_nd_update(last_k_i,
                                         scatter_indices_nd_,
                                         k_i_)

            # Distance from the sources to the last interaction point
            last_dist_ = gather_nd(prefix_distances, gather_indices_)
            # [num_targets, num_sources, max_num_paths, 3]
            last_distance = scatter_nd_update(last_distance,
                                              scatter_indices_nd_,
                                              last_dist_)

            # Direction of tx
            k_tx_ = gather_nd(prefix_ktx, gather_indices_nd_)
            # [num_targets, num_sources, max_num_paths, 3]
            k_tx = scatter_nd_update(k_tx, scatter_indices_nd_, k_tx_)

        # Computes the angles of arrivals, direction of the scattered field,
        # and distance from the scattering point to the targets
        # [num_targets, 3]
        targets = paths.targets
        # [num_targets, 1, 1, 3]
        targets = insert_dims(targets, 2, 1)
        # k_s : [num_targets, num_sources, max_num_paths, 3]
        # scat_2_target_dist : [num_targets, num_sources, max_num_paths]
        k_s, scat_2_target_dist = normalize(targets - last_vertices)
        # Angles of arrivales
        # theta_r, phi_r : [num_targets, num_sources, max_num_paths]
        theta_r, phi_r = theta_phi_from_unit_vec(-k_s)
        # Compute the delays
        # [num_targets, num_sources, max_num_paths]
        tau = (last_distance + scat_2_target_dist) / SPEED_OF_LIGHT
        # [num_targets, num_sources, max_num_paths]
        tau = torch.where(mask, tau, -torch.ones_like(tau))
        # [max_depth, num_targets, num_sources, max_num_paths]
        objects = torch.permute(objects, [3, 0, 1, 2])
        vertices = torch.permute(vertices, [3, 0, 1, 2, 4])

        paths.mask = mask
        paths.vertices = vertices
        paths.objects = objects
        paths.tau = tau
        paths.phi_t = phi_t
        paths.theta_t = theta_t
        paths.phi_r = phi_r
        paths.theta_r = theta_r

        paths_tmp.mat_t = mat_t
        paths_tmp.scat_last_objects = last_objects
        paths_tmp.scat_last_normals = last_normals
        paths_tmp.scat_last_k_i = last_k_i
        paths_tmp.scat_src_2_last_int_dist = last_distance
        paths_tmp.scat_k_s = k_s
        paths_tmp.scat_2_target_dist = scat_2_target_dist
        paths_tmp.k_tx = k_tx
        paths_tmp.k_rx = -k_s

        return paths, paths_tmp

    ##################################################################
    # Utilities
    ##################################################################

    def _compute_directions_distances_delays_angles(self, paths, paths_tmp,
                                                    scattering):
        # pylint: disable=line-too-long
        r"""
        Computes:
        - The direction of incidence and departure at every interaction points
        ``k_i`` and ``k_r``
        - The length of each path segment ``distances``
        - The delays of each path
        - The angles of departure (``theta_t``, ``phi_t``) and arrival
        (``theta_r``, ``phi_r``)

        Input
        ------
        paths : :class:`~torchrf.rt.Paths`
            Paths to update

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Addtional quantities required for paths computation

        scattering : bool
            Set to `True` computing the scattered paths.

        Output
        -------
        paths : :class:`~torchrf.rt.Paths`
            Updated paths

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Updated addtional quantities required for paths computation
        """

        objects = paths.objects
        vertices = paths.vertices
        sources = paths.sources
        targets = paths.targets
        if scattering:
            mask = paths_tmp.scat_prefix_mask
        else:
            mask = paths.mask

        # Maximum depth
        max_depth = vertices.shape[0]

        # Flag that indicates if a ray is valid
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_ray = torch.ne(objects, -1)

        # Vertices updated with the sources and targets
        # [1, num_sources, 1, 3]
        sources = torch.unsqueeze(torch.unsqueeze(sources, dim=0), dim=2)
        # [num_targets, num_sources, max_num_paths, 3]
        sources = torch.broadcast_to(sources, vertices.shape[1:])
        # [1, num_targets, num_sources, max_num_paths, 3]
        sources = torch.unsqueeze(sources, dim=0)
        # [1 + max_depth, num_targets, num_sources, max_num_paths, 3]
        vertices = torch.cat([sources, vertices], dim=0)
        # For the targets, we need to account for the paths having different
        # depths.
        # Pad vertices with dummy values to create the required extra depth
        # [1 + max_depth + 1, num_targets, num_sources, max_num_paths, 3]
        # vertices = torch.nn.functional.pad(vertices, (0, 0, 0, 0, 0, 0, 0, 0, 0, 1), value=1)
        vertices = torch.cat((vertices, torch.ones_like(vertices)[:1]), dim=0)
        # [num_targets, 1, 1, 3]
        targets = torch.unsqueeze(torch.unsqueeze(targets, dim=1), dim=2)
        # [num_targets, num_sources, max_num_paths, 3]
        targets = torch.broadcast_to(targets, vertices.shape[1:])

        #  [max_depth, num_targets, num_sources, max_num_paths]
        target_indices = cast(valid_ray, torch.int64)
        #  [num_targets, num_sources, max_num_paths]
        target_indices = torch.sum(target_indices, dim=0) + 1
        # [num_targets*num_sources*max_num_paths]
        target_indices = reshape(target_indices, [-1, 1])
        # Indices of all (target, source,paths) entries
        # [num_targets*num_sources*max_num_paths, 3]
        target_indices_ = torch.nonzero(torch.full(vertices.shape[1:4], True), as_tuple=False)
        # Indices of all entries in vertices
        # [num_targets*num_sources*max_num_paths, 4]
        target_indices = torch.cat([target_indices, target_indices_], dim=1)
        # Reshape targets
        # vertices : [max_depth + 1, num_targets, num_sources, max_num_paths, 3]
        targets = reshape(targets, [-1, 3])
        vertices = scatter_nd_update(vertices, target_indices, targets)
        # Direction of arrivals (k_i)
        # The last item (k_i[max_depth]) correspond to the direction of arrival
        # at the target. Therefore, k_i is a tensor of length `max_depth + 1`,
        # where `max_depth` is the number of maximum interaction (which could be
        # zero if only LoS is requested).
        # k_i : [max_depth + 1, num_targets, num_sources, max_num_paths, 3]
        # ray_lengths : [max_depth + 1, num_targets, num_sources, max_num_paths]
        k_i = torch.roll(vertices, -1, dims=0) - vertices
        k_i, ray_lengths = normalize(k_i)
        k_i = k_i[:max_depth + 1]
        ray_lengths = ray_lengths[:max_depth + 1]

        # Direction of departures (k_r) at interaction points.
        # We do not need the direction of departure at the source, as it
        # is the same as k_i[0]. Therefore `k_r` only stores the directions of
        # departures at the `max_depth` interaction points.
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        k_r = torch.roll(vertices, -2, dims=0) - torch.roll(vertices, -1, dims=0)
        k_r, _ = normalize(k_r)
        k_r = k_r[:max_depth]

        # Compute the distances
        # [max_depth, num_targets, num_sources, max_num_paths]
        lengths_mask = cast(valid_ray, self._rdtype)
        # First ray is always valid (LoS)
        # [1 + max_depth, num_targets, num_sources, max_num_paths]
        # lengths_mask = F.pad(lengths_mask, (0, 0, 0, 0, 0, 0, 1, 0), value=1.0)
        lengths_mask = torch.cat((lengths_mask, torch.ones_like(lengths_mask)[:1]), dim=0)
        # Compute path distance
        # [1 + max_depth, num_targets, num_sources, max_num_paths]
        distances = lengths_mask * ray_lengths

        # Propagation delay [s]
        # Total length of the paths
        if scattering:
            # Distances of every path prefix, not including the one connecting
            # to the target
            # [max_depth, num_targets, num_sources, max_num_paths]
            total_distance = torch.cumsum(distances[:max_depth], dim=0)
        else:
            # [num_targets, num_sources, max_num_paths]
            total_distance = torch.sum(distances, dim=0)
            # [num_targets, num_sources, max_num_paths]
            tau = total_distance / SPEED_OF_LIGHT
            # Setting -1 for delays corresponding to non-valid paths
            # [num_targets, num_sources, max_num_paths]
            tau = torch.where(mask, tau, -torch.ones_like(tau))

        # Compute angles of departures and arrival
        # theta_t, phi_t: [num_targets, num_sources, max_num_paths]
        theta_t, phi_t = theta_phi_from_unit_vec(k_i[0])
        # In the case of scattering, the angles of arrival are not computed
        # by this function
        if not scattering:
            # Depth of the rays
            # [num_targets, num_sources, max_num_paths]
            ray_depth = torch.sum(cast(valid_ray, torch.int32), dim=0)
            k_rx = -torch.cat([torch.gather(torch.permute(k_i[..., i], [1, 2, 3, 0]),
                                            3, ray_depth[..., None]) for i in range(k_i.shape[-1])], dim=-1)
            # theta_r, phi_r: [num_targets, num_sources, max_num_paths]
            theta_r, phi_r = theta_phi_from_unit_vec(k_rx)

            # Remove duplicated paths.
            # Paths intersecting an edge belonging to two different triangles
            # can be considered twice.
            # Note that this is rare, as intersections rarely occur on edges.
            # The similarity measure used to distinguish paths if the distance
            # between the angles of arrivals and departures.
            # [num_targets, num_sources, max_num_paths, 4]
            sim = torch.stack([theta_t, phi_t, theta_r, phi_r], dim=3)
            # [num_targets, num_sources, max_num_paths, max_num_paths, 4]
            sim = torch.unsqueeze(sim, dim=2) - torch.unsqueeze(sim, dim=3)
            # [num_targets, num_sources, max_num_paths, max_num_paths]
            sim = torch.sum(torch.square(sim), dim=4)
            sim = torch.eq(sim, torch.zeros_like(sim))
            # Keep only the paths with no duplicates.
            # If many paths are identical, keep the one with the highest index
            # [num_targets, num_sources, max_num_paths, max_num_paths]
            sim = torch.triu(sim, diagonal=1)
            sim = torch.logical_and(sim, torch.unsqueeze(mask, dim=-2))
            # [num_targets, num_sources, max_num_paths]
            uniques = torch.all(~sim, dim=3)
            # Keep only the unique paths
            # [num_targets, num_sources, max_num_paths]
            mask = torch.logical_and(uniques, mask)

        # Updates the object storing the paths
        if not scattering:
            paths.mask = mask
            paths.tau = tau
            # In the case of scattering, the angles of arrival are not computed
            # by this function
            paths.theta_r = theta_r
            paths.phi_r = phi_r
            paths_tmp.k_rx = k_rx
        else:
            paths_tmp.scat_prefix_mask = mask
        paths.theta_t = theta_t
        paths.phi_t = phi_t
        paths_tmp.k_i = k_i
        paths_tmp.k_r = k_r
        paths_tmp.k_tx = k_i[0]
        paths_tmp.total_distance = total_distance

        return paths, paths_tmp

    def _get_tx_rx_rotation_matrices(self):
        r"""
        Computes and returns the rotation matrices for rotating according to
        the orientations of the transmitters and receivers rotation matrices,

        Output
        -------
        rx_rot_mat : [num_rx, 3, 3], torch.float
            Matrices for rotating according to the receivers orientations

        tx_rot_mat : [num_tx, 3, 3], torch.float
            Matrices for rotating according to the receivers orientations
        """

        transmitters = self._scene.transmitters.values()
        receivers = self._scene.receivers.values()

        # Rotation matrices for transmitters
        # [num_tx, 3]
        tx_orientations = [tx.orientation for tx in transmitters]
        tx_orientations = torch.stack(tx_orientations, dim=0)
        # [num_tx, 3, 3]
        tx_rot_mat = rotation_matrix(tx_orientations)

        # Rotation matrices for receivers
        # [num_rx, 3]
        rx_orientations = [rx.orientation for rx in receivers]
        rx_orientations = torch.stack(rx_orientations, dim=0)
        # [num_rx, 3, 3]
        rx_rot_mat = rotation_matrix(rx_orientations)

        return rx_rot_mat, tx_rot_mat

    def _get_antennas_relative_positions(self, rx_rot_mat, tx_rot_mat):
        r"""
        Returns the positions of the antennas of the transmitters and receivers.
        The positions are relative to the center of the radio devices, but
        rotated to the GCS.

        Input
        ------
        rx_rot_mat : [num_rx, 3, 3], torch.float
            Matrices for rotating according to the receivers orientations

        tx_rot_mat : [num_tx, 3, 3], torch.float
            Matrices for rotating according to the receivers orientations

        Output
        -------
        rx_rel_ant_pos: [num_rx, rx_array_size, 3], torch.float
            Relative positions of the receivers antennas

        tx_rel_ant_pos: [num_tx, rx_array_size, 3], torch.float
            Relative positions of the transmitters antennas
        """

        # Rotated position of the TX and RX antenna elements
        # [1, tx_array_size, 3]
        tx_rel_ant_pos = torch.unsqueeze(self._scene.tx_array.positions, dim=0)
        # [num_tx, 1, 3, 3]
        tx_rot_mat = torch.unsqueeze(tx_rot_mat, dim=1)
        # [num_tx, tx_array_size, 3]
        tx_rel_ant_pos = matvec(tx_rot_mat, tx_rel_ant_pos)

        # [1, rx_array_size, 3]
        rx_rel_ant_pos = torch.unsqueeze(self._scene.rx_array.positions, dim=0)
        # [num_rx, 1, 3, 3]
        rx_rot_mat = torch.unsqueeze(rx_rot_mat, dim=1)
        # [num_tx, tx_array_size, 3]
        rx_rel_ant_pos = matvec(rx_rot_mat, rx_rel_ant_pos)

        return rx_rel_ant_pos, tx_rel_ant_pos

    def _apply_synthetic_array(self, rx_rot_mat, tx_rot_mat, paths, paths_tmp):
        # pylint: disable=line-too-long
        r"""
        Applies the phase shifts to simulate the effect of a synthetic array
        on a planar wave

        Input
        ------
        rx_rot_mat : [num_rx, 3, 3], torch.float
            Matrices for rotating according to the receivers orientations

        tx_rot_mat : [num_tx, 3, 3], torch.float
            Matrices for rotating according to the receivers orientations

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Addtional quantities required for paths computation

        Output
        -------
        paths : :class:`~torchrf.rt.PathsTmpData`
            Updated paths
        """

        # [num_rx, num_rx_patterns, 1, num_tx, num_tx_patterns, 1,
        #   max_num_paths]
        a = paths.a
        # [num_rx, num_tx, samples_per_tx, 3]
        k_tx = paths_tmp.k_tx
        # [num_tx, num_tx, samples_per_tx, 3]
        k_rx = paths_tmp.k_rx

        two_pi = cast(2. * PI, self._rdtype)

        # Relative positions of the antennas of the transmitters and receivers
        # rx_rel_ant_pos: [num_rx, rx_array_size, 3], torch.float
        #     Relative positions of the receivers antennas
        # tx_rel_ant_pos: [num_tx, rx_array_size, 3], torch.float
        #     Relative positions of the transmitters antennas
        rx_rel_ant_pos, tx_rel_ant_pos = \
            self._get_antennas_relative_positions(rx_rot_mat, tx_rot_mat)

        # Expand dims for broadcasting with antennas
        # The receive vector is flipped as we need vectors that point away
        # from the arrays.
        # [num_rx, 1, 1, num_tx, 1, 1, max_num_paths, 3]
        k_rx = insert_dims(insert_dims(k_rx, 2, 1), 2, 4)
        k_tx = insert_dims(insert_dims(k_tx, 2, 1), 2, 4)
        # Compute the synthetic phase shifts due to the antenna array
        # Transmitter side
        # Expand for broadcasting with receiver, receive antennas,
        # paths
        # [1, 1, 1, num_tx, tx_array_size, 3]
        tx_rel_ant_pos = insert_dims(tx_rel_ant_pos, 3, axis=0)
        # [1, 1, 1, num_tx, 1, tx_array_size, 1, 3]
        tx_rel_ant_pos = torch.unsqueeze(torch.unsqueeze(tx_rel_ant_pos, dim=4),
                                         dim=6)
        # [num_rx, 1, 1, num_tx, 1, tx_array_size, max_num_paths]
        tx_phase_shifts = dot(tx_rel_ant_pos, k_tx)
        # Receiver side
        # Expand for broadcasting with transmitter, transmit antennas,
        # paths
        # [num_rx, 1, rx_array_size, 1, 1, 1, 1, 3]
        rx_rel_ant_pos = insert_dims(torch.unsqueeze(rx_rel_ant_pos, dim=1),
                                     4, axis=3)
        # [num_rx, 1, rx_array_size, num_tx, 1, 1, 1, max_num_paths]
        rx_phase_shifts = dot(rx_rel_ant_pos, k_rx)
        # Total phase shift
        # [num_rx, 1, rx_array_size, num_tx, 1, tx_array_size, max_num_paths]
        phase_shifts = rx_phase_shifts + tx_phase_shifts
        phase_shifts = two_pi * phase_shifts / self._scene.wavelength
        # Apply the phase shifts
        # Broadcast is not supported by TF for such high rank tensors.
        # We therefore do it manually
        # [num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns,
        #   tx_array_size, max_num_paths]
        a = torch.tile(a, [1, 1, phase_shifts.shape[2], 1, 1,
                        phase_shifts.shape[5], 1])
        # [num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns,
        #   tx_array_size, max_num_paths]
        a = a * torch.exp(1j*phase_shifts)
        # a = flatten_dims(flatten_dims(a, 2, 1), 2, 3)
        a = a.flatten(1, 2).flatten(3, 4)
        paths.a = a
        return paths

    def _compute_paths_coefficients(self, rx_rot_mat, tx_rot_mat, paths,
                                    paths_tmp, num_samples,
                                    scattering_coefficient, xpd_coefficient,
                                    etas, alpha_r, alpha_i, lambda_,
                                    scat_keep_prob):
        # pylint: disable=line-too-long
        r"""
        Computes the paths coefficients.

        Input
        ------
        rx_rot_mat : [num_rx, 3, 3], torch.float
            Matrices for rotating according to the receivers orientations

        tx_rot_mat : [num_tx, 3, 3], torch.float
            Matrices for rotating according to the receivers orientations

        paths : :class:`~torchrf.rt.Paths`
            Paths to update

        paths_tmp : :class:`~torchrf.rt.PathsTmpData`
            Updated addtional quantities required for paths computation

        num_samples : int
            Number of random rays to trace in order to generate candidates.
            A large sample count may exhaust GPU memory.

        scattering_coefficient : [num_shapes], torch.float
            Scattering coefficient :math:`S\in[0,1]` as defined in
            :eq:`scattering_coefficient`.

        xpd_coefficient: [num_shapes], torch.float
            Cross-polarization discrimination coefficient :math:`K_x\in[0,1]` as
            defined in :eq:`xpd`.

        etas : [num_shapes], torch.complex
            Complex relative permittivity :math:`\eta` :eq:`eta`

        alpha_r : [num_shapes], torch.int32
            Parameter related to the width of the scattering lobe in the
            direction of the specular reflection.

        alpha_i : [num_shapes], torch.int32
            Parameter related to the width of the scattering lobe in the
            incoming direction.

        lambda_ : [num_shapes], torch.float
            Parameter determining the percentage of the diffusely
            reflected energy in the lobe around the specular reflection.

        scat_keep_prob : float
            Probability with which to keep scattered paths.
            This is helpful to reduce the number of scattered paths computed,
            which might be prohibitively high in some setup.
            Must be in the range (0,1).

        Output
        ------
        paths : :class:`~torchrf.rt.Paths`
            Updated paths
        """

        # [num_rx, num_tx, max_num_paths, 2, 2]
        theta_t = paths.theta_t
        phi_t = paths.phi_t
        theta_r = paths.theta_r
        phi_r = paths.phi_r
        types = paths.types

        mat_t = paths_tmp.mat_t
        k_tx = paths_tmp.k_tx
        k_rx = paths_tmp.k_rx

        # Apply multiplication by wavelength/4pi
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths,2, 2]
        cst = cast(self._scene.wavelength / (4. * PI), self._dtype)
        a = cst * mat_t

        # Get dimensions that are needed later on
        num_rx = a.shape[0]
        rx_array_size = a.shape[1]
        num_tx = a.shape[2]
        tx_array_size = a.shape[3]

        # Expand dimension for broadcasting with receivers/transmitters,
        # antenna dimensions, and paths dimensions
        # [1, 1, num_tx, 1, 1, 3, 3]
        tx_rot_mat = insert_dims(insert_dims(tx_rot_mat, 2, 0), 2, 3)
        # [num_rx, 1, 1, 1, 1, 3, 3]
        rx_rot_mat = insert_dims(rx_rot_mat, 4, 1)

        if self._scene.synthetic_array:
            # Expand for broadcasting with antenna dimensions
            # [num_rx, 1, num_tx, 1, max_num_paths, 3]
            k_rx = torch.unsqueeze(torch.unsqueeze(k_rx, dim=1), dim=3)
            k_tx = torch.unsqueeze(torch.unsqueeze(k_tx, dim=1), dim=3)
            # [num_rx, 1, num_tx, 1, max_num_paths]
            theta_t = torch.unsqueeze(torch.unsqueeze(theta_t, dim=1), dim=3)
            phi_t = torch.unsqueeze(torch.unsqueeze(phi_t, dim=1), dim=3)
            theta_r = torch.unsqueeze(torch.unsqueeze(theta_r, dim=1), dim=3)
            phi_r = torch.unsqueeze(torch.unsqueeze(phi_r, dim=1), dim=3)

        # Normalized wave transmit vector in the local coordinate system of
        # the transmitters
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        k_prime_t = matvec(tx_rot_mat, k_tx, transpose_a=True)

        # Normalized wave receiver vector in the local coordinate system of
        # the receivers
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        k_prime_r = matvec(rx_rot_mat, k_rx, transpose_a=True)

        # Angles of departure in the local coordinate system of the
        # transmitter
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_prime_t, phi_prime_t = theta_phi_from_unit_vec(k_prime_t)

        # Angles of arrival in the local coordinate system of the
        # receivers
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_prime_r, phi_prime_r = theta_phi_from_unit_vec(k_prime_r)

        # Spherical global frame vectors for tx and rx
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_hat_t = theta_hat(theta_t, phi_t)
        phi_hat_t = phi_hat(phi_t)
        theta_hat_r = theta_hat(theta_r, phi_r)
        phi_hat_r = phi_hat(phi_r)

        # Spherical local frame vectors for tx and rx
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
        theta_hat_prime_t = theta_hat(theta_prime_t, phi_prime_t)
        phi_hat_prime_t = phi_hat(phi_prime_t)
        theta_hat_prime_r = theta_hat(theta_prime_r, phi_prime_r)
        phi_hat_prime_r = phi_hat(phi_prime_r)

        # Rotation matrix for going from the spherical LCS to the spherical GCS
        # For transmitters
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths]
        tx_lcs2gcs_11 = dot(theta_hat_t,
                            matvec(tx_rot_mat, theta_hat_prime_t))
        tx_lcs2gcs_12 = dot(theta_hat_t,
                            matvec(tx_rot_mat, phi_hat_prime_t))
        tx_lcs2gcs_21 = dot(phi_hat_t,
                            matvec(tx_rot_mat, theta_hat_prime_t))
        tx_lcs2gcs_22 = dot(phi_hat_t,
                            matvec(tx_rot_mat, phi_hat_prime_t))
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths,2, 2]
        tx_lcs2gcs = torch.stack(
            [torch.stack([tx_lcs2gcs_11, tx_lcs2gcs_12], dim=-1),
             torch.stack([tx_lcs2gcs_21, tx_lcs2gcs_22], dim=-1)],
            dim=-2)
        tx_lcs2gcs = torch.complex(tx_lcs2gcs, torch.zeros_like(tx_lcs2gcs))
        # For receivers
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths]
        rx_lcs2gcs_11 = dot(theta_hat_r,
                            matvec(rx_rot_mat, theta_hat_prime_r))
        rx_lcs2gcs_12 = dot(theta_hat_r,
                            matvec(rx_rot_mat, phi_hat_prime_r))
        rx_lcs2gcs_21 = dot(phi_hat_r,
                            matvec(rx_rot_mat, theta_hat_prime_r))
        rx_lcs2gcs_22 = dot(phi_hat_r,
                            matvec(rx_rot_mat, phi_hat_prime_r))
        # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths,2, 2]
        rx_lcs2gcs = torch.stack(
            [torch.stack([rx_lcs2gcs_11, rx_lcs2gcs_12], dim=-1),
             torch.stack([rx_lcs2gcs_21, rx_lcs2gcs_22], dim=-1)],
            dim=-2)
        rx_lcs2gcs = torch.complex(rx_lcs2gcs, torch.zeros_like(rx_lcs2gcs))

        # List of antenna patterns (callables)
        tx_patterns = self._scene.tx_array.antenna.patterns
        rx_patterns = self._scene.rx_array.antenna.patterns

        tx_ant_fields_hat = []
        for pattern in tx_patterns:
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size,
            #   max_num_paths, 2]
            tx_ant_f = torch.stack(pattern(theta_prime_t, phi_prime_t), dim=-1)
            tx_ant_fields_hat.append(tx_ant_f)

        rx_ant_fields_hat = []
        for pattern in rx_patterns:
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size,
            #   max_num_paths, 2]
            rx_ant_f = torch.stack(pattern(theta_prime_r, phi_prime_r), dim=-1)
            rx_ant_fields_hat.append(rx_ant_f)

        # Stacking the patterns, corresponding to different polarization
        # directions, as an additional dimension
        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, 1/tx_array_size,
        #   max_num_paths, 2]
        rx_ant_fields_hat = torch.stack(rx_ant_fields_hat, dim=1)
        # Expand for broadcasting with tx polarization
        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, 1, 1,
        #   1/tx_array_size, max_num_paths, 2]
        rx_ant_fields_hat = torch.unsqueeze(rx_ant_fields_hat, dim=4)

        # Stacking the patterns, corresponding to different polarization
        # [num_rx, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size,
        #   max_num_paths, 2]
        tx_ant_fields_hat = torch.stack(tx_ant_fields_hat, dim=3)
        # Expand for broadcasting with rx polarization
        # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size,
        #   max_num_paths, 2]
        tx_ant_fields_hat = torch.unsqueeze(tx_ant_fields_hat, dim=1)

        # Antenna patterns to spherical global coordinate system
        # Expand to broadcast with antenna patterns
        # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size,
        #   max_num_paths, 2, 2]
        rx_lcs2gcs = torch.unsqueeze(torch.unsqueeze(rx_lcs2gcs, dim=1), dim=4)
        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, 1, 1/tx_array_size,
        #   max_num_paths, 2]
        rx_ant_fields = matvec(rx_lcs2gcs, rx_ant_fields_hat)
        # Expand to broadcast with antenna patterns
        # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size,
        #   max_num_paths, 2, 2]
        tx_lcs2gcs = torch.unsqueeze(torch.unsqueeze(tx_lcs2gcs, dim=1), dim=4)
        # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size,
        #   max_num_paths, 2, 2]
        tx_ant_fields = matvec(tx_lcs2gcs, tx_ant_fields_hat)

        # Expand the field to broadcast with the antenna patterns
        # [num_rx, 1, rx_array_size, num_tx, 1, tx_array_size, max_num_paths,
        #   2, 2]
        a = torch.unsqueeze(torch.unsqueeze(a, dim=1), dim=4)

        # Compute transmitted field
        # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns, 1/tx_array_size,
        #   max_num_paths, 2]
        a = matvec(a, tx_ant_fields)

        ## Scattering: For scattering, a is the field specularly reflected by
        # the last interaction point. We need to compute the scattered field.
        # [num_scat_paths]
        scat_ind = torch.where(types == Paths.SCATTERED)[0]
        n_scat = size(scat_ind)
        if n_scat > 0:
            n_other = a.shape[-2] - n_scat

            # Get k_x, generate random phase shifts, and compute field vector
            # [num_targets, num_sources, max_num_paths]
            k_x = xpd_coefficient[paths_tmp.scat_last_objects]
            phase_shape = torch.Size(torch.cat([torch._shape_as_tensor(k_x), torch.tensor([2])], dim=0))
            # [num_targets, num_sources, max_num_paths, 2]
            phases = random_uniform(phase_shape, high=2 * PI,  dtype=self._rdtype)
            # [num_targets, num_sources, max_num_paths, 2]
            field_vec = torch.exp(torch.complex(cast(0, self._rdtype), phases))
            # [num_targets, num_sources, max_num_paths, 2]
            k_x_ = torch.stack([torch.sqrt(1 - k_x), torch.sqrt(k_x)], dim=-1)
            k_x_ = torch.complex(k_x_, torch.zeros_like(k_x_))
            field_vec *= k_x_

            # Evalute scattering pattern
            # Get all material properties related to scattering for each path
            # [num_targets, num_sources, max_num_paths]
            etas = etas[paths_tmp.scat_last_objects]
            alpha_r = alpha_r[paths_tmp.scat_last_objects]
            alpha_i = alpha_i[paths_tmp.scat_last_objects]
            lambda_ = lambda_[paths_tmp.scat_last_objects]
            s = scattering_coefficient[paths_tmp.scat_last_objects]

            # Evaluate scattering pattern for all paths. Flattening is needed
            # here as the pattern cannot handle it otherwise.
            f_s = ScatteringPattern.pattern(
                reshape(paths_tmp.scat_last_k_i, [-1, 3]),
                reshape(paths_tmp.scat_k_s, [-1, 3]),
                reshape(paths_tmp.scat_last_normals, [-1, 3]),
                reshape(alpha_r, [-1]),
                reshape(alpha_i, [-1]),
                reshape(lambda_, [-1]))

            # Reshape f_s to original dimensions
            # [num_targets, num_sources, max_num_paths]
            f_s = reshape(f_s, alpha_r.shape)

            # Complete the computation of the field
            # [num_targets, num_sources, max_num_paths]
            scaling = torch.sqrt(f_s) * s

            # The term cos(theta_i)*dA is equal to 4*PI/N*r^2
            # [num_targets, num_sources, max_num_paths]
            scaling *= torch.sqrt(4 * cast(PI, self._rdtype) \
                                  / (scat_keep_prob * num_samples))
            scaling *= paths_tmp.scat_src_2_last_int_dist

            # Apply path loss due to propagation from scattering point
            # to target
            # [num_targets, num_sources, max_num_paths]
            scaling /= paths_tmp.scat_2_target_dist

            # Compute scaled field vector
            # [num_targets, num_sources, max_num_paths, 2]
            field_vec *= torch.unsqueeze(torch.complex(scaling,
                                                       torch.zeros_like(scaling)), -1)

            # Compute Fresnel reflection coefficients at hit point
            # These will be scaled by the reflection reduction factor
            # [num_targets, num_sources, max_num_paths]
            cos_theta = -dot(paths_tmp.scat_last_k_i,
                             paths_tmp.scat_last_normals)

            # [num_targets, num_sources, max_num_paths]
            r_s, r_p = reflection_coefficient(etas, cos_theta)

            # [num_targets, num_sources, max_num_paths, 3]
            e_i_s, e_i_p = compute_field_unit_vectors(
                paths_tmp.scat_last_k_i,
                paths_tmp.scat_k_s,
                paths_tmp.scat_last_normals,
                SolverBase.EPSILON,
                return_e_r=False)

            # a_scat : [num_rx, 1, rx_array_size, num_tx, num_tx_patterns,
            #   tx_array_size, n_scat, 2]
            # a_other : [num_rx, 1, rx_array_size, num_tx, num_tx_patterns,
            #   tx_array_size, max_num_paths - n_scat, 2]
            a_other, a_scat = torch.split(a, [n_other, n_scat], dim=-2)
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size,
            #   max_num_paths, 3]
            _, scat_theta_hat_r = torch.split(theta_hat_r, [n_other, n_scat],
                                              dim=-2)
            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size,
            #   max_num_paths, 3]
            _, scat_phi_hat_r = torch.split(phi_hat_r, [n_other, n_scat],
                                            dim=-2)

            # Compute incoming field
            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size, n_scat,
            #   (3)]
            scat_k_i = paths_tmp.scat_last_k_i
            if self._scene.synthetic_array:
                r_s = insert_dims(r_s, 2, axis=1)
                r_s = insert_dims(r_s, 2, axis=4)
                r_p = insert_dims(r_p, 2, axis=1)
                r_p = insert_dims(r_p, 2, axis=4)
                e_i_s = insert_dims(e_i_s, 2, axis=1)
                e_i_s = insert_dims(e_i_s, 2, axis=4)
                e_i_p = insert_dims(e_i_p, 2, axis=1)
                e_i_p = insert_dims(e_i_p, 2, axis=4)
                scat_k_i = insert_dims(scat_k_i, 2, axis=1)
                scat_k_i = insert_dims(scat_k_i, 2, axis=4)
                field_vec = insert_dims(field_vec, 2, axis=1)
                field_vec = insert_dims(field_vec, 2, axis=4)
            else:
                num_rx = len(self._scene.receivers)
                num_tx = len(self._scene.transmitters)
                r_s = split_dim(r_s, [num_rx, -1], 0)
                r_s = torch.unsqueeze(r_s, dim=1)
                r_s = split_dim(r_s, [num_tx, -1], 3)
                r_s = torch.unsqueeze(r_s, dim=4)
                r_p = split_dim(r_p, [num_rx, -1], 0)
                r_p = torch.unsqueeze(r_p, dim=1)
                r_p = split_dim(r_p, [num_tx, -1], 3)
                r_p = torch.unsqueeze(r_p, dim=4)
                e_i_s = split_dim(e_i_s, [num_rx, -1], 0)
                e_i_s = torch.unsqueeze(e_i_s, dim=1)
                e_i_s = split_dim(e_i_s, [num_tx, -1], 3)
                e_i_s = torch.unsqueeze(e_i_s, dim=4)
                e_i_p = split_dim(e_i_p, [num_rx, -1], 0)
                e_i_p = torch.unsqueeze(e_i_p, dim=1)
                e_i_p = split_dim(e_i_p, [num_tx, -1], 3)
                e_i_p = torch.unsqueeze(e_i_p, dim=4)
                scat_k_i = split_dim(scat_k_i, [num_rx, -1], 0)
                scat_k_i = torch.unsqueeze(scat_k_i, dim=1)
                scat_k_i = split_dim(scat_k_i, [num_tx, -1], 3)
                scat_k_i = torch.unsqueeze(scat_k_i, dim=4)
                field_vec = split_dim(field_vec, [num_rx, -1], 0)
                field_vec = torch.unsqueeze(field_vec, dim=1)
                field_vec = split_dim(field_vec, [num_tx, -1], 3)
                field_vec = torch.unsqueeze(field_vec, dim=4)

            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size, n_scat,2]
            scat_r = torch.stack([r_s, r_p], dim=-1)

            # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns,
            #   1/tx_array_size, n_scat, 2]
            a_in = divide_no_nan(a_scat, scat_r)

            # Compute polarization field vector
            a_in_s, a_in_p = a_in[..., 0:1], a_in[..., 1:2]
            e_i_s = torch.complex(e_i_s, torch.zeros_like(e_i_s))
            e_i_p = torch.complex(e_i_p, torch.zeros_like(e_i_p))
            e_in_pol = a_in_s * e_i_s + a_in_p * e_i_p
            e_pol_hat, _ = normalize(torch.real(e_in_pol))
            e_xpol_hat = cross(e_pol_hat, scat_k_i)

            # Compute incoming spherical unit vectors in GCS
            scat_theta_i, scat_phi_i = theta_phi_from_unit_vec(-scat_k_i)
            scat_theta_hat_i = theta_hat(scat_theta_i, scat_phi_i)
            scat_phi_hat_i = phi_hat(scat_phi_i)

            # Transformation to theta_hat_i, phi_hat_i
            trans_mat = component_transform(e_pol_hat, e_xpol_hat,
                                            scat_theta_hat_i, scat_phi_hat_i)

            # Transformation from theta_hat_s, phi_hat_s to theta_hat_r, phi_hat_r
            # [num_targets, num_sources, max_num_paths, 3]
            # = [num_rx*1/rx_array_size, num_tx*1/tx_array_size, max_num_paths, 3]
            scat_theta_s, scat_phi_s = theta_phi_from_unit_vec(paths_tmp.scat_k_s)
            scat_theta_hat_s = theta_hat(scat_theta_s, scat_phi_s)
            scat_phi_hat_s = phi_hat(scat_phi_s)

            # [num_rx, 1/rx_array_size, num_sources, max_num_paths, 3]
            scat_theta_hat_s = split_dim(scat_theta_hat_s,
                                         [num_rx, rx_array_size], 0)
            scat_phi_hat_s = split_dim(scat_phi_hat_s,
                                       [num_rx, rx_array_size], 0)

            # [num_rx, 1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
            scat_theta_hat_s = split_dim(scat_theta_hat_s,
                                         [num_tx, tx_array_size], 2)
            scat_phi_hat_s = split_dim(scat_phi_hat_s,
                                       [num_tx, tx_array_size], 2)

            # [num_rx, 1,  1/rx_array_size, num_tx, 1/tx_array_size, max_num_paths, 3]
            scat_theta_hat_s = torch.unsqueeze(scat_theta_hat_s, 1)
            scat_phi_hat_s = torch.unsqueeze(scat_phi_hat_s, 1)

            # [num_rx, 1,  1/rx_array_size, num_tx, 1, 1/tx_array_size, max_num_paths, 3]
            scat_theta_hat_s = torch.unsqueeze(scat_theta_hat_s, 4)
            scat_phi_hat_s = torch.unsqueeze(scat_phi_hat_s, 4)

            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size,
            #   max_num_scat_paths, 3]
            scat_theta_hat_r = torch.unsqueeze(scat_theta_hat_r, dim=1)
            scat_theta_hat_r = torch.unsqueeze(scat_theta_hat_r, dim=4)
            # [num_rx, 1, 1/rx_array_size, num_tx, 1, 1/tx_array_size,
            #   max_num_scat_paths, 3]
            scat_phi_hat_r = torch.unsqueeze(scat_phi_hat_r, dim=1)
            scat_phi_hat_r = torch.unsqueeze(scat_phi_hat_r, dim=4)

            trans_mat2 = component_transform(scat_theta_hat_s, scat_phi_hat_s,
                                             scat_theta_hat_r, scat_phi_hat_r)

            trans_mat = torch.matmul(trans_mat2, trans_mat)

            # Compute basis transform matrix for GCS
            # [num_rx, 1, rx_array_size, num_tx, num_tx_patterns, tx_array_size,
            #   max_num_scat_paths, 2, 2]
            trans_mat = torch.complex(trans_mat, torch.zeros_like(trans_mat))

            # Multiply a_scat by sqrt of reflected energy
            # The splitting along the last dim is done because
            # TF cannot handle reduce_sum for such high-dimensional
            # tensors
            #
            # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns,
            #   1/tx_array_size, max_num_paths-n_scat, 1]
            e_spec = torch.sum(torch.square(torch.abs(a_scat)), dim=-1,
                               keepdims=True)
            e_spec = torch.sqrt(e_spec)

            # [num_rx, 1, 1/rx_array_size, num_tx, num_tx_patterns,
            #   1/tx_array_size, max_num_paths-n_scat, 2]
            e_spec = torch.complex(e_spec, torch.zeros_like(e_spec))
            a_scat = field_vec * e_spec

            # Basis transform
            a_scat = matvec(trans_mat, a_scat)

            # Concat with other paths
            a = torch.cat([a_other, a_scat], dim=-2)

        # [num_rx, num_rx_patterns, 1/rx_array_size, num_tx, num_tx_patterns,
        #   1/tx_array_size, max_num_paths]
        a = dot(rx_ant_fields, a)

        if not self._scene.synthetic_array:
            # Reshape as expected to merge antenna and antenna patterns into one
            # dimension, as expected by torchrf
            # [ num_rx, num_rx_ant = num_rx_patterns*rx_array_size,
            #   num_tx, num_tx_ant = num_tx_patterns*tx_array_size,
            #   max_num_paths]
            # a = flatten_dims(flatten_dims(a, 2, 1), 2, 3)
            a = a.flatten(2, 1).flatten(2, 3)

        paths.a = a
        return paths
