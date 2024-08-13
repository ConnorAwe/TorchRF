#
# SPDX-FileCopyrightText: Copyright (c) 2023 SRI International. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Ray tracing module of Torch RF.
"""

###########################################
# Configuring Mitsuba variant
###########################################

import torch
import mitsuba as mi

# If at least one GPU is detected, the CUDA variant is used.
# Otherwise, the LLVM variant is used.
# Note: LLVM is required for execution on CPU.
# Note: If multiple GPUs are visible, the first one is used.
gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
if len(gpus) > 0:
    mi.set_variant('cuda_ad_rgb')
else:
    mi.set_variant('llvm_ad_rgb')

############################################
# Objects available to the user
############################################
# pylint: disable=wrong-import-position
from .scene import load_scene, Scene
from .camera import Camera
from .antenna import Antenna, compute_gain, visualize, iso_pattern,\
                     dipole_pattern, hw_dipole_pattern, tr38901_pattern,\
                     polarization_model_1, polarization_model_2
from .antenna_array import AntennaArray, PlanarArray
from .radio_material import RadioMaterial
from .scene_object import SceneObject
from .scattering_pattern import ScatteringPattern, LambertianPattern,\
    DirectivePattern, BackscatteringPattern
from .transmitter import Transmitter
from .receiver import Receiver
from .paths import Paths
from .coverage_map import CoverageMap
from .utils import rotation_matrix, rotate, theta_phi_from_unit_vec,\
                   r_hat, theta_hat, phi_hat, cross, dot,\
                   normalize, moller_trumbore, component_transform,\
                   reflection_coefficient, compute_field_unit_vectors,\
                   gen_orthogonal_vector, mi_to_torch_tensor, fibonacci_lattice,\
                   cot, rot_mat_from_unit_vecs,\
                       sample_points_on_hemisphere