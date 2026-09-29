#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""RCS Module of Sionna RT"""

from .rcs import register_rcs, unregister_rcs, get_rcs, ConstantRCS
from .cpm import register_cpm, unregister_cpm, get_cpm, ConstantCPM
from .scattering_points import ScatteringPoints
from .scattering_model import ScatteringModel
from .scattering_model_viewer import ScatteringModelViewer
from .sensing_target import SensingTarget
from .constant_rcs_sensing_target import ConstantRCSSensingTarget
from .solver import RCSSolver
from .tr38901 import (TR38901RCS, TR38901CPM, TR38901ScatteringModel,
                      TR38901SensingTarget)
