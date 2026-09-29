#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class implementing a fully absorbing radio material"""

import drjit as dr
import mitsuba as mi
from typing import Tuple

from sionna.rt.constants import InteractionType
from .radio_material_base import RadioMaterialBase


class AbsorberRadioMaterial(RadioMaterialBase):
    # pylint: disable=line-too-long
    r"""
    Class implementing a radio material that absorbs all the incident energy

    This material scatters no energy: no specular reflection, no diffuse
    reflection, no refraction, and no diffraction can occur on a surface made
    of this material. Paths that intersect such a surface are therefore
    terminated by the path solvers, i.e., objects made of this material are
    perfect absorbers that cast radio shadows.

    This material has no electromagnetic parameter. Its only parameter is the
    ``color`` used to display the objects made of it.

    :param name: Unique name of the material. Required if ``props`` is not provided.
        Ignored if ``props`` is provided.
    :param color: RGB (red, green, blue) color for the radio material as displayed in the previewer and renderer. Each RGB component must have a value within the range :math:`[0,1]`. If set to :py:class:`None`, then a random color is used.
    :param props: Mitsuba container storing the material properties, and used
        when loading a scene to initialize the radio material.
    """

    def __init__(self,
                 name: str | None = None,
                 color: Tuple[float, float, float] | None = None,
                 props: mi.Properties | None = None):

        if props is None:
            if name is None:
                raise ValueError("`name` is required when `props` is not"
                                 " provided")
            props = mi.Properties("absorber-radio-material")
            props.set_id(name)
            if color is not None:
                props["color"] = mi.ScalarColor3f(color)

        super().__init__(props)

    # pylint: disable=unused-argument
    def sample(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: mi.Float,
        sample2: mi.Point2f,
        active: bool | mi.Bool = True
    ) -> Tuple[mi.BSDFSample3f, mi.Spectrum]:
        # pylint: disable=line-too-long
        r"""
        Samples the radio material

        As all the incident energy is absorbed, no interaction can be sampled:
        the returned sample always reports
        :data:`~sionna.rt.constants.InteractionType.NONE`, which terminates the
        path, together with a zero Jones matrix.

        :param ctx: A context data structure used to specify which interaction types are enabled
        :param si: Surface interaction data structure describing the underlying surface position
        :param sample1: A uniformly distributed sample on :math:`[0,1]` used to sample the type of interaction
        :param sample2: A uniformly distributed sample on :math:`[0,1]^2` used to sample the direction of the reflected wave in the case of diffuse reflection, or the direction of the diffracted ray on the Keller cone
        :param active: Mask to specify active rays

        :return: Radio material sample and Jones matrix as a :math:`4 \times 4` real-valued matrix
        """
        num_samples = dr.width(si.wi)

        bs = mi.BSDFSample3f()
        bs.sampled_component = dr.full(mi.UInt32, InteractionType.NONE,
                                       num_samples)
        # No energy leaves the surface. A direction must nonetheless be set.
        bs.wo = si.to_world(-si.wi)
        bs.pdf = dr.zeros(mi.Float, num_samples)

        # Not used but must be set
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.Null)
        bs.eta = 1.0

        return bs, mi.Spectrum(0.)

    # pylint: disable=unused-argument
    def eval(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool | mi.Bool = True
    ) -> mi.Spectrum:
        # pylint: disable=line-too-long
        r"""
        Evaluates the radio material

        As all the incident energy is absorbed, the Jones matrix is zero for
        every interaction type and direction of scattering.

        :param ctx: A context data structure used to specify which interaction types are enabled
        :param si: Surface interaction data structure describing the underlying surface position
        :param wo: Direction of propagation of the scattered wave in the world frame
        :param active: Mask to specify active rays

        :return: Jones matrix as a :math:`4 \times 4` real-valued matrix
        """
        return mi.Spectrum(0.)

    # pylint: disable=unused-argument
    def pdf(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool | mi.Bool = True
    ) -> mi.Float:
        # pylint: disable=line-too-long
        r"""
        Evaluates the probability of the sampled interaction type and direction of scattered ray

        As no interaction can be sampled, this probability is always zero.

        :param ctx: A context data structure used to specify which interaction types are enabled
        :param si: Surface interaction data structure describing the underlying surface position
        :param wo: Direction of propagation of the scattered wave in the world frame
        :param active: Mask to specify active rays

        :return: Probability density value
        """
        return dr.zeros(mi.Float, dr.width(si.wi))

    def traverse(self, callback: mi.TraversalCallback):
        # pylint: disable=line-too-long
        r"""
        Traverse the attributes and objects of this instance

        This material has no parameter, and therefore nothing to traverse.

        :param callback: Object used to traverse the scene graph
        """

    def to_string(self) -> str:
        r"""
        Returns a string describing the object
        """
        return "AbsorberRadioMaterial"


# Register this custom BSDF plugin
mi.register_bsdf("absorber-radio-material",
                 lambda props: AbsorberRadioMaterial(props=props))
