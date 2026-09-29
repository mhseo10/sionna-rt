#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Parameters of the RCS models of 3GPP TR 38.901, clause 7.9.2"""

from dataclasses import dataclass
from typing import Tuple

# Models of clause 7.9.2.1. Model 1 characterizes a sensing target as a single
# scattering point whose monostatic RCS does not depend on the aspect, i.e.,
# with the angular component `sigma_D` fixed to 1. Model 2 adds that angular
# component.
MODEL_1 = 1
MODEL_2 = 2


@dataclass(frozen=True)
class LobeParameters:
    r"""
    One set of parameters of a scattering point, i.e., one row of Tables
    7.9.2.1-2 to 7.9.2.1-7

    A sensing target modelled with a single scattering point holds all the
    sets of parameters of its table, and the bisector angle indexes one of
    them. A sensing target split into several scattering points gives one set
    to every point.

    :param name: Name of the lobe, i.e., the side of the target it describes
    :param phi_center: Azimuth angle of the lobe center [deg], or
        :py:class:`None` for a lobe with no azimuth dependence
    :param phi_3db: Azimuth 3dB beamwidth [deg], or :py:class:`None` for a
        lobe with no azimuth dependence
    :param theta_center: Zenith angle of the lobe center [deg]
    :param theta_3db: Zenith 3dB beamwidth [deg]
    :param g_max: Peak value of :math:`10\lg(\sigma_M\sigma_D)` [dBsm]
    :param sigma_max: Dynamic range of :math:`10\lg(\sigma_M\sigma_D)` [dB]
    :param theta_range: Range of zenith angles of the bisector for which this
        set of parameters applies [deg]
    :param phi_range: Range of azimuth angles of the bisector for which this
        set of parameters applies [deg]
    """

    name: str
    phi_center: float | None
    phi_3db: float | None
    theta_center: float
    theta_3db: float
    g_max: float
    sigma_max: float
    theta_range: Tuple[float, float]
    phi_range: Tuple[float, float]

    @property
    def azimuth_dependent(self) -> bool:
        r"""Whether :math:`\sigma^H_{dB}(\varphi)` is evaluated for this lobe

        The roof and bottom lobes have no azimuth center, and the notes of
        Tables 7.9.2.1-2 to 7.9.2.1-7 set
        :math:`\sigma^H_{dB}(\varphi) = 0` for them.

        :type: :py:class:`bool`
        """
        return self.phi_center is not None

    @property
    def floor(self) -> float:
        r""":math:`G_{max} - \sigma_{max}` [dBsm], the lower bound of
        :math:`10\lg(\sigma_M\sigma_D)`

        :type: :py:class:`float`
        """
        return self.g_max - self.sigma_max


@dataclass(frozen=True)
class TargetParameters:
    r"""
    Parameters of the RCS of one type of sensing target

    :param lobes: Sets of parameters of the target. A target modelled with a
        single scattering point holds all of them and selects one from the
        bisector angle, whereas a target split into several scattering points
        gives one to every point.
    :param num_scattering_points: Number of scattering points of the target
    :param k1: :math:`k_1` of eq. 7.9.2-3
    :param k2: :math:`k_2` of eq. 7.9.2-3
    :param sigma_m_db: :math:`10\lg(\sigma_M)` [dBsm], the mean of the linear
        monostatic RCS values of a scattering point. It parameterizes eq.
        7.9.2-2 for model 1, and is reference data for model 2, where it is
        already accounted for in ``g_max``.
    :param sigma_s_db: Standard deviation of :math:`10\lg(\sigma_S)` [dB], the
        third RCS component of clause 7.9.2.1. It parameterizes the draw of
        that component, see :class:`~sionna.rt.rcs.TR38901RCS`, which only
        draws it if its ``random_sigma_s`` is set and otherwise fixes
        :math:`\sigma_S` to 1, exactly its linear mean per eq. 7.9.2-1.
    :param xpr_mean_db: Mean of the XPR [dB], from Table 7.9.2.2-1. It sets the
        cross-polarized entries of the cross-polarization matrix of clause
        7.9.2.2, see :class:`~sionna.rt.rcs.TR38901CPM`.
    :param xpr_std_db: Standard deviation of the XPR [dB], from Table
        7.9.2.2-1. It parameterizes the draw of the XPR, which
        :class:`~sionna.rt.rcs.TR38901CPM` only makes if its ``random_xpr``
        is set and otherwise fixes the XPR to ``xpr_mean_db``.
    :param dimensions: Default size of the target, as length, width and height
        [m]
    """

    lobes: Tuple[LobeParameters, ...]
    num_scattering_points: int
    k1: float
    k2: float
    sigma_m_db: float
    sigma_s_db: float
    xpr_mean_db: float
    xpr_std_db: float
    dimensions: Tuple[float, float, float]


# XPR of Table 7.9.2.2-1, as (mean, standard deviation) [dB]
_XPR_UAV = (13.75, 7.07)
_XPR_HUMAN = (19.81, 4.25)
_XPR_VEHICLE = (21.12, 6.88)
_XPR_AGV = (9.60, 6.85)

# Default sizes of the sensing targets, as length, width and height [m]. The
# ordering follows the spec, which gives "Size (Length x Width x Height)" in
# the human table of clause 7.9.1.
#
# Option 2 and Option 1 of the UAV table of clause 7.9.1, the latter also
# being the "Target type" row of Tables 7.9.6.1-1 and 7.9.6.2-1.
_SIZE_UAV_SMALL = (0.3, 0.4, 0.2)
_SIZE_UAV_LARGE = (1.6, 1.5, 0.7)
# Adult pedestrian of the human table of clause 7.9.1 and of Table 7.9.6.1-2,
# which also lists a child at (0.2, 0.3, 1.0)
_SIZE_HUMAN = (0.5, 0.5, 1.75)
# Option 1 of the AGV table of clause 7.9.1 and of Table 7.9.6.1-4, which also
# lists Option 2 at (1.5, 3.0, 1.5). The AGV is wider than it is long, which
# is what clause 7.9.2.1 means by "The front of the AGV is the short edge of
# AGV in horizontal direction", the front facing the +x axis.
_SIZE_AGV = (0.5, 1.0, 0.5)
# Table 7.9.6.1-3 specifies the sensing target as "Vehicle type 2 [TR37.885]"
# without giving its size. TR 37.885 defines type 2 as a passenger vehicle 5m
# long, 2m wide and 1.6m high, with a rooftop antenna at 1.6m, the latter
# matching the "1.6m for vehicle type UE" of the same table.
_SIZE_VEHICLE = (5.0, 2.0, 1.6)


# Table 7.9.2.1-1: parameters on RCS for the STs with angular independent
# monostatic RCS values, as (10lg(sigma_M), sigma_sigmaS_dB)
_TABLE_1_UAV_SMALL = (-12.81, 3.74)
_TABLE_1_HUMAN = (-1.37, 3.94)

# Table 7.9.2.1-2: parameters on RCS for UAV with large size
_TABLE_2 = (
    LobeParameters("left", 90., 7.13, 90., 8.68, 7.43, 14.30,
                   (45., 135.), (45., 135.)),
    LobeParameters("back", 180., 10.09, 90., 11.43, 3.99, 10.86,
                   (45., 135.), (135., 225.)),
    LobeParameters("right", 270., 7.13, 90., 8.68, 7.43, 14.30,
                   (45., 135.), (225., 315.)),
    LobeParameters("front", 0., 14.19, 90., 16.53, 1.02, 7.89,
                   (45., 135.), (-45., 45.)),
    LobeParameters("bottom", None, None, 180., 4.93, 13.55, 20.42,
                   (135., 180.), (0., 360.)),
    LobeParameters("roof", None, None, 0., 4.93, 13.55, 20.42,
                   (0., 45.), (0., 360.)),
)

# Table 7.9.2.1-3: parameters on RCS for human with RCS model 2
_TABLE_3 = (
    LobeParameters("front", 0., 216.65, 90., 55.7, 2.14, 7.7,
                   (0., 180.), (-90., 90.)),
    LobeParameters("back", 180., 216.65, 90., 55.7, 2.14, 7.7,
                   (0., 180.), (90., 270.)),
)

# Table 7.9.2.1-4: parameters on RCS for vehicle with single scattering point
_TABLE_4 = (
    LobeParameters("left", 90., 26.90, 79.70, 44.42, 20.75, 13.68,
                   (30., 180.), (45., 135.)),
    LobeParameters("back", 180., 36.32, 79.65, 36.73, 14.56, 7.50,
                   (30., 180.), (135., 225.)),
    LobeParameters("right", 270., 26.90, 79.70, 44.42, 20.75, 13.68,
                   (30., 180.), (225., 315.)),
    LobeParameters("front", 0., 40.54, 71.75, 29.13, 15.52, 8.45,
                   (30., 180.), (-45., 45.)),
    LobeParameters("roof", None, None, 0.00, 18.13, 21.26, 14.19,
                   (0., 30.), (0., 360.)),
)

# Table 7.9.2.1-5: parameters on RCS for vehicle with multiple scattering
# points
_TABLE_5 = (
    LobeParameters("left", 90., 26.90, 79.70, 44.42, 20.60, 20.52,
                   (0., 180.), (0., 360.)),
    LobeParameters("back", 180., 36.32, 79.65, 36.73, 13.90, 13.82,
                   (0., 180.), (0., 360.)),
    LobeParameters("right", 270., 26.90, 79.70, 44.42, 20.60, 20.52,
                   (0., 180.), (0., 360.)),
    LobeParameters("front", 0., 40.54, 71.75, 29.13, 14.99, 14.91,
                   (0., 180.), (0., 360.)),
    LobeParameters("roof", None, None, 0.00, 18.13, 21.12, 21.05,
                   (0., 180.), (0., 360.)),
)

# Table 7.9.2.1-6: parameters on RCS for AGV with single scattering point
_TABLE_6 = (
    LobeParameters("front", 0., 13.68, 90., 13.68, 13.02, 23.29,
                   (30., 180.), (-45., 45.)),
    LobeParameters("left", 90., 15.53, 75., 20.03, 7.33, 17.60,
                   (30., 180.), (45., 135.)),
    LobeParameters("back", 180., 12.49, 90., 11.89, 11.01, 21.28,
                   (30., 180.), (135., 225.)),
    LobeParameters("right", 270., 15.53, 75., 20.03, 7.33, 17.60,
                   (30., 180.), (225., 315.)),
    LobeParameters("roof", None, None, 0., 11.44, 11.79, 22.06,
                   (0., 30.), (0., 360.)),
)

# Table 7.9.2.1-7: parameters on RCS for AGV with multiple scattering points
_TABLE_7 = (
    LobeParameters("front", 0., 13.68, 90., 13.68, 13.00, 30.26,
                   (0., 180.), (0., 360.)),
    LobeParameters("left", 90., 15.53, 75., 20.03, 7.27, 24.53,
                   (0., 180.), (0., 360.)),
    LobeParameters("back", 180., 12.49, 90., 11.89, 10.98, 28.24,
                   (0., 180.), (0., 360.)),
    LobeParameters("right", 270., 15.53, 75., 20.03, 7.27, 24.53,
                   (0., 180.), (0., 360.)),
    LobeParameters("roof", None, None, 0., 11.44, 11.77, 29.03,
                   (0., 180.), (0., 360.)),
)


# Parameters of every sensing target type and RCS model the spec defines,
# keyed by `(object_type, model_type)`.
#
# Neither the number of scattering points nor the number of lobes is a free
# parameter, as both follow from the object type. A table row is a set of
# parameters rather than a scattering point: in a single-scattering-point
# model the rows are mutually exclusive lobes of one point, indexed by the
# bisector angle, and only in a multiple-scattering-point model does every row
# become a point of its own contributing simultaneously. The large UAV has six
# rows and the single-point vehicle five, both still one point.
#
# The spec leaves a choice for the vehicle and the AGV only, "For UAV with
# large size and human, single SPST is modelled. While for vehicle and AGV,
# both models with single and multiple SPSTs are provided", hence the
# `-single-sp` and `-multi-sp` object types. Those two variants are different
# parameterizations rather than one model expressed twice, e.g., the floor
# `G_max - sigma_max` of Table 7.9.2.1-5 is that of Table 7.9.2.1-4 minus
# 10lg(5), the single-point value split across the five points.
TARGET_PARAMETERS = {
    ("uav-small-size", MODEL_1): TargetParameters(
        lobes=(), num_scattering_points=1, k1=0., k2=0.,
        sigma_m_db=_TABLE_1_UAV_SMALL[0], sigma_s_db=_TABLE_1_UAV_SMALL[1],
        xpr_mean_db=_XPR_UAV[0], xpr_std_db=_XPR_UAV[1],
        dimensions=_SIZE_UAV_SMALL),
    ("human", MODEL_1): TargetParameters(
        lobes=(), num_scattering_points=1, k1=0., k2=0.,
        sigma_m_db=_TABLE_1_HUMAN[0], sigma_s_db=_TABLE_1_HUMAN[1],
        xpr_mean_db=_XPR_HUMAN[0], xpr_std_db=_XPR_HUMAN[1],
        dimensions=_SIZE_HUMAN),
    ("uav-large-size", MODEL_2): TargetParameters(
        lobes=_TABLE_2, num_scattering_points=1, k1=6.05, k2=1.33,
        sigma_m_db=-5.85, sigma_s_db=2.50,
        xpr_mean_db=_XPR_UAV[0], xpr_std_db=_XPR_UAV[1],
        dimensions=_SIZE_UAV_LARGE),
    ("human", MODEL_2): TargetParameters(
        lobes=_TABLE_3, num_scattering_points=1, k1=0.5714, k2=0.1,
        sigma_m_db=-1.37, sigma_s_db=3.94,
        xpr_mean_db=_XPR_HUMAN[0], xpr_std_db=_XPR_HUMAN[1],
        dimensions=_SIZE_HUMAN),
    ("vehicle-single-sp", MODEL_2): TargetParameters(
        lobes=_TABLE_4, num_scattering_points=1, k1=6., k2=1.65,
        sigma_m_db=11.25, sigma_s_db=3.41,
        xpr_mean_db=_XPR_VEHICLE[0], xpr_std_db=_XPR_VEHICLE[1],
        dimensions=_SIZE_VEHICLE),
    ("vehicle-multi-sp", MODEL_2): TargetParameters(
        lobes=_TABLE_5, num_scattering_points=5, k1=6., k2=1.65,
        sigma_m_db=11.25, sigma_s_db=3.41,
        xpr_mean_db=_XPR_VEHICLE[0], xpr_std_db=_XPR_VEHICLE[1],
        dimensions=_SIZE_VEHICLE),
    ("agv-single-sp", MODEL_2): TargetParameters(
        lobes=_TABLE_6, num_scattering_points=1, k1=12., k2=1.45,
        sigma_m_db=-4.25, sigma_s_db=2.51,
        xpr_mean_db=_XPR_AGV[0], xpr_std_db=_XPR_AGV[1],
        dimensions=_SIZE_AGV),
    ("agv-multi-sp", MODEL_2): TargetParameters(
        lobes=_TABLE_7, num_scattering_points=5, k1=12., k2=1.45,
        sigma_m_db=-4.25, sigma_s_db=2.51,
        xpr_mean_db=_XPR_AGV[0], xpr_std_db=_XPR_AGV[1],
        dimensions=_SIZE_AGV),
}

# Object types of the lookup, in the order in which they are reported to the
# user
OBJECT_TYPES = ("uav-small-size", "uav-large-size", "human",
                "vehicle-single-sp", "vehicle-multi-sp",
                "agv-single-sp", "agv-multi-sp")

# Object types that only name one of the two variants the spec provides, and
# are therefore ambiguous
_AMBIGUOUS_OBJECT_TYPES = ("vehicle", "agv")


def get_target_parameters(object_type: str,
                          model_type: int) -> TargetParameters:
    """Returns the parameters of a type of sensing target

    :param object_type: Type of sensing target, one of
        :data:`~sionna.rt.rcs.tr38901.parameters.OBJECT_TYPES`
    :param model_type: RCS model, either 1 or 2
    :return: Parameters of the sensing target
    """
    if object_type in _AMBIGUOUS_OBJECT_TYPES:
        raise ValueError(
            f"The specifications provide both a single and a multiple "
            f"scattering point model for '{object_type}', so one of "
            f"'{object_type}-single-sp' and '{object_type}-multi-sp' must be "
            f"selected. They are different parameterizations, not one model "
            f"expressed twice.")
    if object_type not in OBJECT_TYPES:
        raise ValueError(f"Unknown object type '{object_type}'. Must be one "
                         f"of {list(OBJECT_TYPES)}")
    if model_type not in (MODEL_1, MODEL_2):
        raise ValueError(f"Invalid model type '{model_type}'. Must be either "
                         f"{MODEL_1} or {MODEL_2}")

    parameters = TARGET_PARAMETERS.get((object_type, model_type))
    if parameters is None:
        available = sorted(model for object_type_, model in TARGET_PARAMETERS
                           if object_type_ == object_type)
        raise ValueError(f"The specifications do not define RCS model "
                         f"{model_type} for '{object_type}'. Available model "
                         f"types for this object type: {available}")
    return parameters
