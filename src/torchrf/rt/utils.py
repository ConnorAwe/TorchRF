#
# SPDX-FileCopyrightText: Copyright (c) 2023 SRI International. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Ray tracer utilities
"""

import torch
from scipy.special import fresnel
import drjit as dr
import numpy as np

from torchrf.utils.tensors import expand_to_rank
from torchrf.constants import PI


def rotation_matrix(angles):
    r"""
    Computes rotation matrices as defined in :eq:`rotation`

    The closed-form expression in (7.1-4) [TR38901]_ is used.

    Input
    ------
    angles : [...,3], torch.float
        Angles for the rotations [rad].
        The last dimension corresponds to the angles
        :math:`(\alpha,\beta,\gamma)` that define
        rotations about the axes :math:`(z, y, x)`,
        respectively.

    Output
    -------
    : [...,3,3], torch.float
        Rotation matrices
    """

    a = angles[..., 0]
    b = angles[..., 1]
    c = angles[..., 2]
    cos_a = torch.cos(a)
    cos_b = torch.cos(b)
    cos_c = torch.cos(c)
    sin_a = torch.sin(a)
    sin_b = torch.sin(b)
    sin_c = torch.sin(c)

    r_11 = cos_a * cos_b
    r_12 = cos_a * sin_b * sin_c - sin_a * cos_c
    r_13 = cos_a * sin_b * cos_c + sin_a * sin_c
    r_1 = torch.stack([r_11, r_12, r_13], dim=-1)

    r_21 = sin_a * cos_b
    r_22 = sin_a * sin_b * sin_c + cos_a * cos_c
    r_23 = sin_a * sin_b * cos_c - cos_a * sin_c
    r_2 = torch.stack([r_21, r_22, r_23], dim=-1)

    r_31 = -sin_b
    r_32 = cos_b * sin_c
    r_33 = cos_b * cos_c
    r_3 = torch.stack([r_31, r_32, r_33], dim=-1)

    rot_mat = torch.stack([r_1, r_2, r_3], dim=-2)
    return rot_mat


def rotate(p, angles):
    r"""
    Rotates points ``p`` by the ``angles`` according
    to the 3D rotation defined in :eq:`rotation`

    Input
    -----
    p : [...,3], torch.float
        Points to rotate

    angles : [..., 3]
        Angles for the rotations [rad].
        The last dimension corresponds to the angles
        :math:`(\alpha,\beta,\gamma)` that define
        rotations about the axes :math:`(z, y, x)`,
        respectively.

    Output
    ------
    : [...,3]
        Rotated points ``p``
    """

    # Rotation matrix
    # [..., 3, 3]
    rot_mat = rotation_matrix(angles)
    rot_mat = expand_to_rank(rot_mat, torch.linalg.matrix_rank(p) + 1, 0)

    # Rotation around ``center``
    # [..., 3]
    rot_p = torch.mv(rot_mat, p)

    return rot_p


def theta_phi_from_unit_vec(v):
    r"""
    Computes zenith and azimuth angles (:math:`\theta,\varphi`)
    from unit-norm vectors as described in :eq:`theta_phi`

    Input
    ------
    v : [...,3], torch.float
        Tensor with unit-norm vectors in the last dimension

    Output
    -------
    theta : [...], torch.float
        Zenith angles :math:`\theta`

    phi : [...], torch.float
        Azimuth angles :math:`\varphi`
    """
    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]

    # Clip to ensure numerical stability
    theta = acos_diff(z)
    phi = torch.atan2(y, x)
    return theta, phi


def r_hat(theta, phi):
    r"""
    Computes the spherical unit vetor :math:`\hat{\mathbf{r}}(\theta, \phi)`
    as defined in :eq:`spherical_vecs`

    Input
    -------
    theta : arbitrary shape, torch.float
        Zenith angles :math:`\theta` [rad]

    phi : same shape as ``theta``, torch.float
        Azimuth angles :math:`\varphi` [rad]

    Output
    --------
    rho_hat : ``phi.shape`` + [3], torch.float
        Vector :math:`\hat{\mathbf{r}}(\theta, \phi)`  on unit sphere
    """
    rho_hat = torch.stack([torch.sin(theta) * torch.cos(phi),
                           torch.sin(theta) * torch.sin(phi),
                           torch.cos(theta)], dim=-1)
    return rho_hat


def theta_hat(theta, phi):
    r"""
    Computes the spherical unit vector
    :math:`\hat{\boldsymbol{\theta}}(\theta, \varphi)`
    as defined in :eq:`spherical_vecs`

    Input
    -------
    theta : arbitrary shape, torch.float
        Zenith angles :math:`\theta` [rad]

    phi : same shape as ``theta``, torch.float
        Azimuth angles :math:`\varphi` [rad]

    Output
    --------
    theta_hat : ``phi.shape`` + [3], torch.float
        Vector :math:`\hat{\boldsymbol{\theta}}(\theta, \varphi)`
    """
    x = torch.cos(theta) * torch.cos(phi)
    y = torch.cos(theta) * torch.sin(phi)
    z = -torch.sin(theta)
    return torch.stack([x, y, z], -1)


def phi_hat(phi):
    r"""
    Computes the spherical unit vector
    :math:`\hat{\boldsymbol{\varphi}}(\theta, \varphi)`
    as defined in :eq:`spherical_vecs`

    Input
    -------
    phi : same shape as ``theta``, torch.float
        Azimuth angles :math:`\varphi` [rad]

    Output
    --------
    theta_hat : ``phi.shape`` + [3], torch.float
        Vector :math:`\hat{\boldsymbol{\varphi}}(\theta, \varphi)`
    """
    x = -torch.sin(phi)
    y = torch.cos(phi)
    z = torch.zeros_like(x)
    return torch.stack([x, y, z], -1)


def cross(u, v):
    r"""
    Computes the cross (or vector) product between u and v

    Input
    ------
    u : [...,3]
        First vector

    v : [...,3]
        Second vector

    Output
    -------
    : [...,3]
        Cross product between ``u`` and ``v``
    """
    u_x = u[..., 0]
    u_y = u[..., 1]
    u_z = u[..., 2]

    v_x = v[..., 0]
    v_y = v[..., 1]
    v_z = v[..., 2]

    w = torch.stack([u_y * v_z - u_z * v_y,
                     u_z * v_x - u_x * v_z,
                     u_x * v_y - u_y * v_x], dim=-1)

    return w


def dot(u, v, keepdim=False, clip=False):
    """
    Computes the dot product between u and v.

    Parameters:
    - u (torch.Tensor): First vector of shape [..., 3].
    - v (torch.Tensor): Second vector of shape [..., 3].
    - keepdim (bool): If True, keep the last dimension. Defaults to False.
    - clip (bool): If True, clip output to [-1, 1]. Defaults to False.

    Returns:
    - torch.Tensor: Dot product between u and v. The last dimension is removed if keepdim is set to False.
    """
    # Reshape u to [..., 1, 3] and perform matrix-vector multiplication with v
    res = torch.sum(u*v, dim=-1, keepdim=keepdim)

    if clip:
        one = torch.ones((), dtype=u.dtype)
        res = torch.clamp(res, -one, one)

    return res


def normalize(v, dim=-1):
    r"""
    Normalizes ``v`` to unit norm

    Input
    ------
    v : [...,3], torch.float
        Vector

    Output
    -------
    : [...,3], torch.float
        Normalized vector

    : [...], torch.float
        Norm of the unnormalized vector
    """
    norm = torch.norm(v, dim=dim, keepdim=True)
    n_v = torch.nan_to_num(torch.div(v, norm))
    norm = torch.squeeze(norm, dim=dim)
    return n_v, norm


def moller_trumbore(o, d, p0, p1, p2, epsilon):
    r"""
    Computes the intersection between a ray ``ray`` and a triangle defined
    by its vertices ``p0``, ``p1``, and ``p2`` using the Moller–Trumbore
    intersection algorithm.

    Input
    -----
    o, d: [..., 3], torch.float
        Ray origin and direction.
        The direction `d` must be a unit vector.

    p0, p1, p2: [..., 3], torch.float
        Vertices defining the triangle

    epsilon : (), torch.float
        Small value used to avoid errors due to numerical precision

    Output
    -------
    t : [...], torch.float
        Position along the ray from the origin at which the intersection
        occurs (if any)

    hit : [...], bool
        `True` if the ray intersects the triangle. `False` otherwise.
    """

    rdtype = o.dtype
    zero = 0.0
    one = 1.0

    # [..., 3]
    e1 = p1 - p0
    e2 = p2 - p0

    # [...,3]
    pvec = cross(d, e2)
    # [...,1]
    det = dot(e1, pvec, keepdim=True)

    # If the ray is parallel to the triangle, then det = 0.
    hit = torch.greater(torch.abs(det), zero)

    # [...,3]
    tvec = o - p0
    # [...,1]
    u = torch.nan_to_num(torch.div(dot(tvec, pvec, keepdim=True), det))
    # [...,1]
    hit = torch.logical_and(hit,
                            torch.logical_and(torch.greater_equal(u, -epsilon),
                                              torch.less_equal(u, one + epsilon)))

    # [..., 3]
    qvec = cross(tvec, e1)
    # [...,1]
    v = torch.nan_to_num(torch.div(dot(d, qvec, keepdim=True), det))
    # [..., 1]
    hit = torch.logical_and(hit,
                            torch.logical_and(torch.greater_equal(v, -epsilon),
                                              torch.less_equal(u + v, one + epsilon)))
    # [..., 1]
    t = torch.nan_to_num(torch.div(dot(e2, qvec, keepdim=True), det))
    # [..., 1]
    hit = torch.logical_and(hit, torch.greater_equal(t, epsilon))

    # [...]
    t = torch.squeeze(t, dim=-1)
    hit = torch.squeeze(hit, dim=-1)

    return t, hit


def component_transform(e_s, e_p, e_i_s, e_i_p):
    """
    Compute basis change matrix for reflections

    Input
    -----
    e_s : [..., 3], torch.float
        Source unit vector for S polarization

    e_p : [..., 3], torch.float
        Source unit vector for P polarization

    e_i_s : [..., 3], torch.float
        Target unit vector for S polarization

    e_i_p : [..., 3], torch.float
        Target unit vector for P polarization

    Output
    -------
    r : [..., 2, 2], torch.float
        Change of basis matrix for going from (e_s, e_p) to (e_i_s, e_i_p)
    """
    r_11 = dot(e_i_s, e_s)
    r_12 = dot(e_i_s, e_p)
    r_21 = dot(e_i_p, e_s)
    r_22 = dot(e_i_p, e_p)
    r1 = torch.stack([r_11, r_12], dim=-1)
    r2 = torch.stack([r_21, r_22], dim=-1)
    r = torch.stack([r1, r2], dim=-2)
    return r


max32 = 2**31 - 1


def mi_to_torch_tensor(mi_tensor, dtype):
    """
    Get a TensorFlow eager tensor from a Mitsuba/DrJIT tensor
    """
    # When there is only one input, the .tf() methods crashes.
    # The following hack takes care of this corner case
    # TODO: is this necessary when we aren't using tensorflow?
    # TODO: does this case even work in pytorch?
    dr.eval(mi_tensor)
    dr.sync_thread()
    if dr.shape(mi_tensor)[-1] == 1:
        mi_tensor = dr.repeat(mi_tensor, 2)
        tf_tensor = torch.tensor(mi_tensor, dtype=dtype)[:1]
    else:
        try:
            tf_tensor = mi_tensor.torch().to(dtype)
        except RuntimeError:
            # TODO: verify that this is safe
            # TODO: this might not be the fastest way to do this
            if dtype is torch.int32:
                x = mi_tensor.numpy()
                x = np.where(x < max32, x, max32)
                tf_tensor = torch.tensor(x.astype(np.int32, casting='unsafe'))
    return tf_tensor


def gen_orthogonal_vector(k, epsilon):
    """
    Generate an arbitrary vector that is orthogonal to ``k``.

    Input
    ------
    k : [..., 3], torch.float
        Vector

    epsilon : (), torch.float
        Small value used to avoid errors due to numerical precision

    Output
    -------
    : [..., 3], torch.float
        Vector orthogonal to ``k``
    """
    rdtype = k.dtype
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=rdtype)
    ex = expand_to_rank(ex, len(k.shape), 0)

    ey = torch.tensor([0.0, 1.0, 0.0], dtype=rdtype)
    ey = expand_to_rank(ey, len(k.shape), 0)

    n1 = cross(k, ex)
    n1_norm = torch.norm(n1, dim=-1, keepdim=True)
    n2 = cross(k, ey)
    return torch.where(torch.greater(n1_norm, epsilon), n1, n2)


def compute_field_unit_vectors(k_i, k_r, n, epsilon, return_e_r=True):
    """
    Compute unit vector parallel and orthogonal to incident plane

    Input
    ------
    k_i : [..., 3], torch.float
        Direction of arrival

    k_r : [..., 3], torch.float
        Direction of reflection

    n : [..., 3], torch.float
        Surface normal

    epsilon : (), torch.float
        Small value used to avoid errors due to numerical precision

    return_e_r : bool
        If `False`, only ``e_i_s`` and ``e_i_p`` are returned.

    Output
    ------
    e_i_s : [..., 3], torch.float
        Incident unit field vector for S polarization

    e_i_p : [..., 3], torch.float
        Incident unit field vector for P polarization

    e_r_s : [..., 3], torch.float
        Reflection unit field vector for S polarization.
        Only returned if ``return_e_r`` is `True`.

    e_r_p : [..., 3], torch.float
        Reflection unit field vector for P polarization
        Only returned if ``return_e_r`` is `True`.
    """
    e_i_s = cross(k_i, n)
    e_i_s_norm = torch.norm(e_i_s, dim=-1, keepdim=True)
    # In case of normal incidence, the incidence plan is not uniquely
    # define and the Fresnel coefficent is the same for both polarization
    # (up to a sign flip for the parallel component due to the definition of
    # polarization).
    # It is required to detect such scenarios and define an arbitrary valid
    # e_i_s to fix an incidence plane, as the result from previous
    # computation leads to e_i_s = 0.
    e_i_s = torch.where(torch.greater(e_i_s_norm, epsilon), e_i_s,
                        gen_orthogonal_vector(n, epsilon))

    e_i_s, _ = normalize(e_i_s)
    e_i_p, _ = normalize(cross(e_i_s, k_i))
    if not return_e_r:
        return e_i_s, e_i_p
    else:
        e_r_s = e_i_s
        e_r_p, _ = normalize(cross(e_r_s, k_r))
        return e_i_s, e_i_p, e_r_s, e_r_p


def reflection_coefficient(eta, cos_theta):
    """
    Compute simplified reflection coefficients

    Input
    ------
    eta : Any shape, torch.complex
        Real part of the relative permittivity

    cos_thehta : Same as ``eta``, torch.float
        Cosine of the incident angle

    Output
    -------
    r_te : Same as input, torch.complex
        Fresnel reflection coefficient for S direction

    r_tm : Same as input, torch.complex
        Fresnel reflection coefficient for P direction
    """
    cos_theta = torch.complex(cos_theta, torch.zeros_like(cos_theta))

    # Fresnel equations
    a = cos_theta
    b = torch.sqrt(eta - 1. + cos_theta ** 2)
    r_te = torch.nan_to_num(torch.div(a - b, a + b))

    c = eta * a
    d = b
    r_tm = torch.nan_to_num(torch.div(c - d, c + d))
    return r_te, r_tm


def paths_to_segments(paths):
    """
    Extract the segments corresponding to a set of ``paths``

    Input
    -----
    paths : :class:`~torchrf.rt.Paths`
        A set of paths

    Output
    -------
    starts, ends : [n,3], float
        Endpoints of the segments making the paths.
    """

    mask = paths.mask.numpy()
    vertices = paths.vertices.numpy()
    objects = paths.objects.numpy()
    sources, targets = paths.sources.numpy(), paths.targets.numpy()

    # Emit directly two lists of the beginnings and endings of line segments
    starts = []
    ends = []
    for rx in range(vertices.shape[1]):  # For each receiver
        for tx in range(vertices.shape[2]):  # For each transmitter
            for p in range(vertices.shape[3]):  # For each path depth
                if not mask[rx, tx, p]:
                    continue

                start = sources[tx]
                i = 0
                while ((i < objects.shape[0])
                       and (objects[i, rx, tx, p] != -1)):
                    end = vertices[i, rx, tx, p]
                    starts.append(start)
                    ends.append(end)
                    start = end
                    i += 1
                # Explicitly add the path endpoint
                starts.append(start)
                ends.append(targets[rx])
    return starts, ends


def scene_scale(scene):
    bbox = scene.mi_scene.bbox()
    tx_positions, rx_positions = {}, {}
    devices = ((scene.transmitters, tx_positions),
               (scene.receivers, rx_positions))
    for source, destination in devices:
        for k, rd in source.items():
            p = rd.position.numpy()
            bbox.expand(p)
            destination[k] = p

    sc = 2. * bbox.bounding_sphere().radius
    return sc, tx_positions, rx_positions, bbox


def fibonacci_lattice(num_points, dtype=torch.float32):
    """
    Generates a Fibonacci lattice for the unit 3D sphere

    Input
    -----
    num_points : int
        Number of points

    Output
    -------
    points : [num_points, 3]
        Generated rectangular coordinates of the lattice points
    """

    golden_ratio = (1. + np.sqrt(5.)) / 2.

    if (num_points % 2) == 0:
        min_n = -num_points // 2
        max_n = num_points // 2 - 1
    else:
        min_n = -(num_points - 1) // 2
        max_n = (num_points - 1) // 2

    ns = torch.arange(min_n, max_n + 1, dtype=dtype)

    # Spherical coordinate
    phis = 2. * PI * ns / golden_ratio
    thetas = torch.acos(2. * ns / num_points)

    # Rectangular coordinates
    x = torch.sin(thetas) * torch.cos(phis)
    y = torch.sin(thetas) * torch.sin(phis)
    z = torch.cos(thetas)
    points = torch.stack([x, y, z], dim=1)

    return points


def cot(x):
    """
    Cotangens function

    Input
    ------
    x : [...], torch.float

    Output
    -------
    : [...], torch.float
        Cotengent of x
    """
    return torch.nan_to_num(1 / torch.tan(x))


def rot_mat_from_unit_vecs(a, b):
    r"""
    Computes Rodrigues` rotation formular :eq:`rodrigues_matrix`

    Input
    ------
    a : [...,3], torch.float
        First unit vector

    b : [...,3], torch.float
        Second unit vector

    Output
    -------
    : [...,3,3], torch.float
        Rodrigues' rotation matrix
    """
    # Compute rotation axis vector
    k, _ = normalize(cross(a, b))

    # Deal with special case where a and b are parallel
    o = gen_orthogonal_vector(a, 1e-6)
    k = torch.where(torch.sum(torch.abs(k), dim=-1, keepdims=True) == 0, o, k)

    # Compute K matrix
    shape = torch.concat([k.shape[:-1], [1]], dim=-1)
    zeros = torch.zeros(shape, a.dtype)
    kx, ky, kz = torch.split(k, 3, dim=-1)
    l1 = torch.concat([zeros, -kz, ky], dim=-1)
    l2 = torch.concat([kz, zeros, -kx], dim=-1)
    l3 = torch.concat([-ky, kx, zeros], dim=-1)
    k_mat = torch.stack([l1, l2, l3], dim=-2)

    # Assemble full rotation matrix
    tmp_eye = torch.eye(3, dtype=a.dtype)
    eye = torch.ones(k.shape[:-1] + (3, 3), dtype=a.dtype) * expand_to_rank(tmp_eye, len(tmp_eye.shape) + len(k.shape[:-1]), 0)

    cos_theta = dot(a, b)
    sin_theta = torch.sin(acos_diff(cos_theta))
    cos_theta = expand_to_rank(cos_theta, len(eye.shape), axis=-1)
    sin_theta = expand_to_rank(sin_theta, len(eye.shape), axis=-1)
    rot_mat = eye + k_mat * sin_theta + \
              torch.linalg.matmul(k_mat, k_mat) * (1 - cos_theta)
    return rot_mat


def sample_points_on_hemisphere(normals, num_samples=1):
    # pylint: disable=line-too-long
    r"""
    Randomly sample points on hemispheres defined by their normal vectors

    Input
    -----
    normals : [batch_size, 3], torch.float
        Normal vectors defining hemispheres

    num_samples : int
        Number of random samples to draw for each hemisphere
        defined by its normal vector.
        Defaults to 1.

    Output
    ------
    points : [batch_size, num_samples, 3], torch.float or [batch_size, 3], torch.float if num_samples=1.
        Random points on the hemispheres
    """
    dtype = normals.dtype
    batch_size = normals.shape[0]
    shape = [batch_size, num_samples]

    # Sample phi uniformly distributed on [0,2*PI]
    phi = torch.rand(shape, dtype=dtype) * 2 * PI

    # Generate samples of theta for uniform distribution on the hemisphere
    u = torch.rand(shape, dtype=dtype)
    theta = torch.acos(u)

    # Transform spherical to Cartesian coordinates
    points = r_hat(theta, phi)

    # Compute rotation matrices
    z_hat = torch.tensor([[0, 0, 1]], dtype=dtype)
    z_hat = torch.broadcast_to(z_hat, normals.shape)
    rot_mat = rot_mat_from_unit_vecs(z_hat, normals)
    rot_mat = torch.unsqueeze(rot_mat, dim=1)

    # Compute rotated points
    points = torch.matmul(rot_mat, points)

    if num_samples == 1:
        points = torch.squeeze(points, dim=1)

    return points


def acos_diff(x, epsilon=1e-7):
    r"""
    Implementation of arccos(x) that avoids evaluating the gradient at x
    -1 or 1 by using straight through estimation, i.e., in the
    forward pass, x is clipped to (-1, 1), but in the backward pass, x is
    clipped to (-1 + epsilon, 1 - epsilon).

    Input
    ------
    x : any shape, torch.float
        Value at which to evaluate arccos

    epsilon : torch.float
        Small backoff to avoid evaluating the gradient at -1 or 1.
        Defaults to 1e-7.

    Output
    -------
     : same shape as x, torch.float
        arccos(x)
    """

    x_clip_1 = torch.clip(x, -1., 1.)
    x_clip_2 = torch.clip(x, -1. + epsilon, 1. - epsilon)
    eps = torch.detach(x - x_clip_2)
    x_1 = x - eps
    acos_x_1 = torch.acos(x_1)
    y = acos_x_1 + torch.detach(torch.acos(x_clip_1) - acos_x_1)
    return y


def fresnel_sin(x):
    # Ensure the input tensor is a PyTorch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)

    # Compute the Fresnel Sine integral using SciPy
    s, _ = fresnel(x)

    return s


def fresnel_cos(x):
    # Ensure the input tensor is a PyTorch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)

    # Compute the Fresnel Cosine integral using SciPy
    _, c = fresnel(x)

    return c


def divide_no_nan(x, y):
    # Compute the division
    division_result = torch.div(x, y)
    
    # Replace NaN and inf values with 0.0
    result = torch.nan_to_num(division_result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result


def random_uniform(shape, low=0, high=1, dtype=torch.float):
    # Specify the range for the uniform distribution

    # Generate a random uniform tensor
    uniform_tensor = (high - low) * torch.rand(shape, dtype=dtype) + low
    return uniform_tensor


def scatter_nd_update(tensor, indices, updates):
    outp = tensor.clone()
    outp[tuple(indices.t())] = updates.to(outp.dtype)
    return outp


# TODO: this is only written for the cases in test_scatter_nd_add and might not be fully general
# TODO: extend to handle repeated indices
def scatter_nd_add(tensor, indices, updates):
    """
    Tries to follow the specification for tensorflow.scatter_nd_add.  Note that tf allows indices to be repeated,
    in which case they are summed.  This is different from torch.scatter, which is non-deterministic in case of
    repeated indices.
    """
    # Flatten the indexed dimensions of the tensor
    tself = tensor.flatten(0, indices.shape[-1]-1)  # `self` in torch.scatter documentation

    # Convert indices to flattened indices: ind[i,j,k] = i*shape1*shape2 + j*shape1 + k
    shape = torch.tensor(tensor.shape[:indices.shape[-1]], dtype=torch.int64)
    reindex = torch.ones_like(shape)
    reindex[1:] = torch.cumprod(shape.__reversed__()[:-1], 0)
    reindex = reindex.__reversed__()
    reindex = reindex[:indices.shape[-1]]
    reindex = reindex.expand(indices.shape)
    tindex = (indices * reindex).sum(-1)
    while len(tindex.shape) > 1:
        tindex = tindex.sum(-1)
    while len(tindex.shape) < len(tself.shape):
        tindex = tindex.unsqueeze(-1)
    tindex = tindex.expand(-1, *tself.shape[1:])

    # Flatten the indexed dimensions of updates
    tsrc = updates.flatten(0, len(indices.shape)-2)  # `src` in torch.scatter documentation

    # Backprop in torch is only implemented when the following assertion holds:
    assert tsrc.shape == tindex.shape

    """
    Strategy: For multiple copies of the same index, first scatter all 0th instances, then all 1th instances, etc.
    That way we never scatter multiple copies of the same index in the same call, which is nondeterministic in torch.
    """
    _, inverse, counts = torch.unique(tindex, dim=0, return_inverse=True, return_counts=True)
    # Assign instance[k] = n-1 to the nth instance of each index in inverse
    instance = torch.zeros_like(inverse)
    for ind in torch.where(counts > 1)[0]:
        instance[inverse == ind] = torch.arange(counts[ind])
    # scatter indices / src based on iteration
    for it in range(counts.max()):
        zeros = torch.zeros_like(tself)
        add = zeros.scatter_(0, tindex[instance == it, ...], tsrc[instance == it, ...])
        tself = add + tself
    tself = tself.reshape(tensor.shape)
    return tself



# TODO: replace scatter_nd_update


def cast(var, dtype):
    if isinstance(var, torch.Tensor):
        return var.to(dtype)
    else:
        return torch.tensor(var, dtype=dtype)


def reshape(tensor, shape):
    return tensor.reshape(shape)


def shape(tensor):
    return tensor.shape


def size(tensor):
    return tensor.numel()


def gather_batch(params, dim, indices, batch_dim=0):
    result = torch.stack([torch.gather(p, dim - batch_dim, i)
                          for p, i in zip(params, indices)])
    return result


def gather_nd_torch(params, indices, batch_dims=0):
    """
    Gather elements from 'params' using 'indices' with support for 'batch_dims'.
    
    Args:
    params (Tensor): The input tensor from which elements will be gathered.
    indices (Tensor): The tensor containing indices for gathering elements.
    batch_dims (int): The number of batch dimensions to skip.
    
    Returns:
    Tensor: A tensor containing the gathered elements.
    """
    # Ensure that the input tensors have the same data type and device.
    if params.dtype != indices.dtype:
        indices = indices.to(params.dtype)
    
    # Get the shape of 'params' and 'indices'.
    params_shape = params.shape
    indices_shape = indices.shape
    
    # Calculate the number of dimensions and gather dimensions.
    num_dims = indices_shape[-1]
    gather_dims = indices_shape[batch_dims:-1]
    
    # Flatten the dimensions before the batch dimensions.
    flat_dims = params_shape[:batch_dims]
    
    # Flatten the dimensions after the batch dimensions.
    batch_params = params.view(*flat_dims, -1)
    
    # Calculate the number of batch elements.
    num_batch_elements = batch_params.size(batch_dims)
    
    # Calculate the flat indices by multiplying indices with strides.
    strides = torch.tensor([params_shape[-1] ** i for i in range(num_dims)], device=params.device)
    flat_indices = (indices.view(-1, num_dims) * strides).sum(dim=-1)
    
    # Compute the gather indices, adjusting for batch dimensions.
    gather_indices = (flat_indices // num_batch_elements).view(-1)
    
    # Gather the elements from 'params'.
    gathered_elements = batch_params.gather(batch_dims, gather_indices.view(*gather_dims).to(torch.int64))
    
    return gathered_elements


def gather_nd(params, indices):
    """ A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.

    Args:
      params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
      indices: a tensor of dimension [b1, ..., bn, x, m]

    Returns:
      gathered: a tensor of dimension [b1, ..., bn, x, c].

    """
    gathered = params[tuple(indices.t())]
    return gathered


def band_part(input_matrix, num_lower, num_upper):
    """
    Extracts the lower or upper triangular part of a matrix.

    Parameters:
        input_matrix (Tensor): Input matrix.
        num_lower (int): Number of lower diagonals to keep (negative values count from the upper diagonal).
        num_upper (int): Number of upper diagonals to keep (negative values count from the lower diagonal).

    Returns:
        Tensor: Matrix with only the specified diagonals.
    """
    n = input_matrix.size(0)  # Assuming the input is a square matrix
    indices = torch.triu_indices(n, n, offset=num_upper).t() + torch.tril_indices(n, n, offset=num_lower).t()
    return input_matrix[indices[:, 0], indices[:, 1]].view(n, n)


def rank(tensor):
    """
    Gets rank of tensor
    """
    return len(tensor.shape)


def matvec(a, b, transpose_a=False):
    """
    Gets rank of tensor
    """
    if transpose_a:
        outp = torch.matmul(a.transpose(-2, -1), b.unsqueeze(-1)).squeeze(-1)
    else:
        outp = torch.matmul(a, b.unsqueeze(-1)).squeeze(-1)
    return outp


def tf_gather(params, indices, axis=0):
    if isinstance(indices, int):
        slicer = [None] * len(params.shape)
        slicer[axis] = indices
        return params[slicer]
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices)

    # Flatten indices and then expand to match params
    ind = indices.flatten()
    for _ in range(axis):
        ind = ind.unsqueeze(0)
    for _ in range(axis+1, len(params.shape)):
        ind = ind.unsqueeze(-1)
    ind_shape = list(params.shape)
    ind_shape[axis] = ind.shape[axis]
    ind = ind.expand(ind_shape)

    # Torch gather
    out = torch.gather(params, axis, cast(ind, torch.int64))

    # Reshape output
    out_shape = list(params.shape[:axis]) + list(indices.shape) + list(params.shape[axis+1:])
    out = out.reshape(out_shape)
    return out


def tf_matvec(a, b, transpose_a=False):
    if not isinstance(a, torch.Tensor):
        raise ValueError("expected a Tensor")
    if not isinstance(b, torch.Tensor):
        raise ValueError("expected a Tensor")
    if transpose_a:
        a = a.transpose(-2, -1)
    if not b.shape[-1] == a.shape[-1]:
        raise ValueError("Tensors must have same final dimension")
    # Also, a[:-2] and b[:-1] must be able to broadcast together
    b = b.unsqueeze(-2)
    p = a * b
    s = p.sum(-1)
    return s
