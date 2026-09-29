#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Module implementing radio materials for the Sionna RT"""

from .radio_material_base import RadioMaterialBase
from .radio_material import RadioMaterial
from .itu_material import ITURadioMaterial
from .absorber_material import AbsorberRadioMaterial
from .scattering_pattern import register_scattering_pattern, \
                                scattering_pattern_registry, \
                                ScatteringPattern, \
                                LambertianPattern, \
                                BackscatteringPattern, \
                                DirectivePattern
