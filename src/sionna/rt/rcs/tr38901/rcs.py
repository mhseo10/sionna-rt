#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""RCS callables of the models of 3GPP TR 38.901, clause 7.9.2"""

import math
from typing import Sequence, Tuple
import drjit as dr
import mitsuba as mi

from sionna.rt.utils import theta_phi_from_unit_vec

from .parameters import LobeParameters
from .random_draws import STREAM_SIGMA_S, direction_keys, gaussian

# Lower bound applied to cos(beta/2) before taking its logarithm in eq.
# 7.9.2-3. The term diverges at beta = 180 degrees, i.e., for forward
# scattering, where the `G_max - sigma_max` floor governs anyway. Flooring the
# argument keeps the arithmetic finite while staying far below any floor of
# the tables, as 5*log10(1e-12) = -60 dB.
_MIN_COS_HALF_BETA = 1e-12

# Dr.Jit provides no base-10 logarithm or exponential
_INV_LN_10 = 1./math.log(10.)
_LN_10_OVER_10 = 0.1*math.log(10.)
_LN_10_OVER_20 = 0.05*math.log(10.)

# Number of standard deviations above its mean at which eq. 7.9.4-1 truncates
# the draw of `sigma_S`
_SIGMA_S_NUM_STD = 3.


class TR38901RCS:
    r"""
    Callable evaluating the RCS of a scattering point of a sensing target, as
    specified in 3GPP TR 38.901, clause 7.9.2.1

    The callable returns the bistatic radar cross-section
    :math:`\sigma_M\sigma_D\sigma_S` [:math:`\text{m}^2`], where
    :math:`10\lg(\sigma_M\sigma_D)` is given by eq. 7.9.2-2 for RCS model 1
    and by eq. 7.9.2-3 for RCS model 2, and :math:`\sigma_S` is the third RCS
    component of clause 7.9.2.1.

    How the scattered power is distributed over the polarization components is
    set by the cross-polarization matrix of clause 7.9.2.2, which
    :class:`~sionna.rt.rcs.TR38901CPM` implements, and not by this callable.

    The component :math:`\sigma_S` is random, and is only drawn if
    ``random_sigma_s`` is set. It is otherwise fixed to 1, which eq. 7.9.2-1
    makes exactly its linear mean. When drawn, :math:`10\lg(\sigma_S)` is
    Gaussian with the standard deviation ``sigma_s_std_db`` of Tables
    7.9.2.1-1 to 7.9.2.1-7 and the mean of eq. 7.9.2-1, which is what makes
    the linear mean 1, and is truncated three standard deviations above its
    mean as in eq. 7.9.4-1.

    The draw is keyed by a hash of the pair of incident and scattered
    directions and of the ``seed`` the callable is evaluated with, so it is a
    function of its inputs alone. Two evaluations of the same pair of
    directions with the same seed therefore give the same cross-section, and
    two runs of :class:`~sionna.rt.rcs.RCSSolver` on the same scene with the
    same seed give every path the same cross-section. Exchanging the two
    directions also leaves it unchanged, as clause 7.9.4 requires for
    monostatic sensing.

    With several lobes, the bisector angle indexes one of them as specified by
    the ``theta_range`` and ``phi_range`` of every lobe, which tile the sphere.
    The selection is branch-free, as every lobe is evaluated.

    :param lobes: Sets of parameters of the scattering point. An empty
        sequence selects RCS model 1, which has no lobe.
    :param k1: :math:`k_1` of eq. 7.9.2-3
    :param k2: :math:`k_2` of eq. 7.9.2-3
    :param sigma_m_db: :math:`10\lg(\sigma_M)` [dBsm], used by RCS model 1
        only
    :param sigma_s_std_db: Standard deviation of :math:`10\lg(\sigma_S)` [dB],
        from the tables of clause 7.9.2.1. Only used if ``random_sigma_s`` is
        set.
    :param random_sigma_s: If set to `True`, the component :math:`\sigma_S` is
        drawn for every pair of directions. It is otherwise fixed to 1.
    """

    def __init__(self,
                 lobes: Sequence[LobeParameters],
                 k1: float = 0.,
                 k2: float = 0.,
                 sigma_m_db: float = 0.,
                 sigma_s_std_db: float = 0.,
                 random_sigma_s: bool = False):

        self._lobes = tuple(lobes)
        self._k1 = float(k1)
        self._k2 = float(k2)
        self._sigma_m_db = float(sigma_m_db)
        self._sigma_s_std_db = float(sigma_s_std_db)
        if self._sigma_s_std_db < 0.:
            raise ValueError("`sigma_s_std_db` must be non-negative")

        # Mean of eq. 7.9.2-1, which sets the linear mean of `sigma_S` to 1,
        # and the truncation of eq. 7.9.4-1 which follows from it
        self._sigma_s_mean_db = -_LN_10_OVER_20*self._sigma_s_std_db**2
        self._sigma_s_max_db = self._sigma_s_mean_db \
                               + _SIGMA_S_NUM_STD*self._sigma_s_std_db

        self.random_sigma_s = random_sigma_s

    @property
    def lobes(self) -> Tuple[LobeParameters, ...]:
        """Sets of parameters of the scattering point

        :type: :py:class:`tuple` [
            :class:`~sionna.rt.rcs.tr38901.LobeParameters` ]
        """
        return self._lobes

    @property
    def sigma_s_std_db(self) -> float:
        r"""Standard deviation of :math:`10\lg(\sigma_S)` [dB], from the
        tables of clause 7.9.2.1

        :type: :py:class:`float`
        """
        return self._sigma_s_std_db

    @property
    def random_sigma_s(self) -> bool:
        r"""Get/set whether the component :math:`\sigma_S` is drawn for every
        pair of directions, rather than fixed to its linear mean of 1

        :type: :py:class:`bool`
        """
        return self._random_sigma_s

    @random_sigma_s.setter
    def random_sigma_s(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("`random_sigma_s` must be a bool")
        self._random_sigma_s = value

    def __call__(self,
                 k_i: mi.Vector3f,
                 k_s: mi.Vector3f,
                 seed: int = 0) -> mi.Float:
        r"""Evaluates the RCS for the given directions

        :param k_i: Incident directions of propagation, in the local
            coordinate system of the scattering point
        :param k_s: Scattered directions of propagation, in the local
            coordinate system of the scattering point
        :param seed: Seed of the draw of :math:`\sigma_S`, unused if
            ``random_sigma_s`` is not set

        :return: Bistatic radar cross-section
            :math:`\sigma_M\sigma_D\sigma_S` [:math:`\text{m}^2`]
        """
        sigma_db = self.sigma_md_db(k_i, k_s) + self.sigma_s_db(k_i, k_s, seed)
        return dr.exp(_LN_10_OVER_10*sigma_db)

    def sigma_s_db(self,
                   k_i: mi.Vector3f,
                   k_s: mi.Vector3f,
                   seed: int = 0) -> mi.Float:
        r"""Draws :math:`10\lg(\sigma_S)` [dB] for the given directions

        The draw is the truncated Gaussian of clause 7.9.2.1, and is 0, i.e.,
        :math:`\sigma_S = 1`, if ``random_sigma_s`` is not set.

        :param k_i: Incident directions of propagation, in the local
            coordinate system of the scattering point
        :param k_s: Scattered directions of propagation, in the local
            coordinate system of the scattering point
        :param seed: Seed of the draw

        :return: :math:`10\lg(\sigma_S)` [dB]
        """
        if not self._random_sigma_s:
            return dr.zeros(mi.Float, dr.width(k_i))

        # Only the key which is symmetric in the two directions is needed, as
        # clause 7.9.4 requires the same `sigma_S` in both directions of
        # propagation
        key, _, _ = direction_keys(k_i, k_s, seed)
        sigma_s_db = gaussian(key, STREAM_SIGMA_S, self._sigma_s_mean_db,
                              self._sigma_s_std_db)
        return dr.minimum(sigma_s_db, self._sigma_s_max_db)

    def sigma_md_db(self,
                    k_i: mi.Vector3f,
                    k_s: mi.Vector3f) -> mi.Float:
        r"""Evaluates :math:`10\lg(\sigma_M\sigma_D)` [dBsm] for the given
        directions

        :param k_i: Incident directions of propagation, in the local
            coordinate system of the scattering point
        :param k_s: Scattered directions of propagation, in the local
            coordinate system of the scattering point

        :return: :math:`10\lg(\sigma_M\sigma_D)` [dBsm]
        """
        # Per eqs. 7.9.4-11 and 7.9.4-12, the incident and scattered
        # directions of the specifications point away from the scattering
        # point, whereas the solver gives directions of propagation
        d_i = -mi.Vector3f(k_i)
        d_s = mi.Vector3f(k_s)

        # Bistatic angle, which is 0 for monostatic backscatter. It is
        # computed from the tangent rather than from the cosine, as the RCS
        # peaks at `beta = 0` where an arc cosine loses precision.
        beta = dr.atan2(dr.norm(dr.cross(d_i, d_s)), dr.dot(d_i, d_s))

        if not self._lobes:
            return self._sigma_md_db_model_1(beta)

        # Bisector between the incident and the scattered ray, which indexes
        # the lobe and gives the aspect the scattering point is seen from.
        # Its norm is `2*cos(beta/2)`, so it vanishes for forward scattering,
        # where the floor of eq. 7.9.2-3 governs and any direction will do.
        bisector = d_i + d_s
        bisector_norm = dr.norm(bisector)
        bisector = dr.select(bisector_norm > _MIN_COS_HALF_BETA,
                             bisector*dr.rcp(bisector_norm),
                             mi.Vector3f(0., 0., 1.))
        theta, phi = theta_phi_from_unit_vec(bisector)

        return self._sigma_md_db_model_2(beta, theta, phi)

    def _sigma_md_db_model_1(self, beta: mi.Float) -> mi.Float:
        r"""Evaluates eq. 7.9.2-2

        :param beta: Bistatic angle [rad]
        :return: :math:`10\lg(\sigma_M\sigma_D)` [dBsm]
        """
        return self._sigma_m_db - 3.*dr.sin(0.5*beta)

    def _sigma_md_db_model_2(self,
                             beta: mi.Float,
                             theta: mi.Float,
                             phi: mi.Float) -> mi.Float:
        r"""Evaluates eq. 7.9.2-3, selecting one lobe from the bisector angle

        :param beta: Bistatic angle [rad]
        :param theta: Zenith angle of the bisector [rad]
        :param phi: Azimuth angle of the bisector [rad]
        :return: :math:`10\lg(\sigma_M\sigma_D)` [dBsm]
        """
        # Dependence on the bistatic angle, shared by every lobe. It applies
        # outside of the inner `min` of eq. 7.9.2-3, and is at most 0.
        cos_half_beta = dr.maximum(dr.cos(0.5*beta), _MIN_COS_HALF_BETA)
        bistatic_db = -self._k1*dr.sin(0.5*self._k2*beta) \
                      + 5.*_INV_LN_10*dr.log(cos_half_beta)

        sigma_md_db = None
        for lobe in self._lobes:
            value = self._lobe_sigma_md_db(lobe, theta, phi, bistatic_db)
            if sigma_md_db is None:
                # The ranges of the lobes tile the sphere, so the first lobe
                # is only used as a fallback that is never selected
                sigma_md_db = value
            else:
                sigma_md_db = dr.select(self._in_range(lobe, theta, phi),
                                        value, sigma_md_db)
        return sigma_md_db

    def _lobe_sigma_md_db(self,
                          lobe: LobeParameters,
                          theta: mi.Float,
                          phi: mi.Float,
                          bistatic_db: mi.Float) -> mi.Float:
        r"""Evaluates eq. 7.9.2-3 for a single lobe

        :param lobe: Set of parameters of the lobe
        :param theta: Zenith angle of the bisector [rad]
        :param phi: Azimuth angle of the bisector [rad]
        :param bistatic_db: Dependence on the bistatic angle [dB]
        :return: :math:`10\lg(\sigma_M\sigma_D)` [dBsm]
        """
        sigma_max = lobe.sigma_max

        theta_offset = theta - dr.deg2rad(lobe.theta_center)
        sigma_v_db = -dr.minimum(
            12.*dr.square(theta_offset*dr.rcp(dr.deg2rad(lobe.theta_3db))),
            sigma_max)

        if lobe.azimuth_dependent:
            # The offset is wrapped to [-pi, pi], as an azimuth is only
            # defined up to a full turn
            phi_offset = _wrap_to_pi(phi - dr.deg2rad(lobe.phi_center))
            sigma_h_db = -dr.minimum(
                12.*dr.square(phi_offset*dr.rcp(dr.deg2rad(lobe.phi_3db))),
                sigma_max)
        else:
            # The notes of Tables 7.9.2.1-2 to 7.9.2.1-7 set the azimuth
            # dependence to 0 for the roof and bottom lobes, which have no
            # azimuth center
            sigma_h_db = 0.

        pattern_db = lobe.g_max \
            - dr.minimum(-(sigma_v_db + sigma_h_db), sigma_max)
        return dr.maximum(pattern_db + bistatic_db, lobe.floor)

    @staticmethod
    def _in_range(lobe: LobeParameters,
                  theta: mi.Float,
                  phi: mi.Float) -> mi.Bool:
        """Whether a bisector direction selects a lobe

        :param lobe: Set of parameters of the lobe
        :param theta: Zenith angle of the bisector [rad]
        :param phi: Azimuth angle of the bisector [rad]
        :return: Mask set to `True` for the directions selecting ``lobe``
        """
        theta_lo, theta_hi = lobe.theta_range
        in_theta = theta >= dr.deg2rad(theta_lo)
        if theta_hi < 180.:
            in_theta &= theta < dr.deg2rad(theta_hi)
        else:
            # The upper end of the zenith range is inclusive, so that the
            # ranges of a table cover [0, 180] entirely
            in_theta &= theta <= dr.deg2rad(theta_hi)

        phi_lo, phi_hi = lobe.phi_range
        if (phi_hi - phi_lo) >= 360.:
            return in_theta

        # An azimuth range may wrap around, as with [-45, 45), so containment
        # is tested on the offset from its lower end, wrapped to [0, 2*pi)
        phi_offset = _wrap_to_two_pi(phi - dr.deg2rad(phi_lo))
        return in_theta & (phi_offset < dr.deg2rad(phi_hi - phi_lo))


def _wrap_to_pi(angle: mi.Float) -> mi.Float:
    """Wraps angles to [-pi, pi)

    :param angle: Angles [rad]
    :return: Wrapped angles [rad]
    """
    return _wrap_to_two_pi(angle + dr.pi) - dr.pi


def _wrap_to_two_pi(angle: mi.Float) -> mi.Float:
    """Wraps angles to [0, 2*pi)

    :param angle: Angles [rad]
    :return: Wrapped angles [rad]
    """
    wrapped = angle - dr.two_pi*dr.floor(angle*dr.rcp(dr.two_pi))
    # Guard against values landing on 2*pi through rounding
    return dr.select(wrapped < dr.two_pi, wrapped, 0.)
