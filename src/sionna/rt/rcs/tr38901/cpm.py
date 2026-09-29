#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""CPM callables of the models of 3GPP TR 38.901, clause 7.9.2"""

import math
from typing import Tuple
import drjit as dr
import mitsuba as mi

from .random_draws import (STREAM_PHASE_CROSS, STREAM_PHASE_PHI_PHI,
                           STREAM_PHASE_THETA_THETA, STREAM_XPR,
                           direction_keys, gaussian, phase)

# Dr.Jit provides no base-10 exponential
_LN_10_OVER_20 = 0.05*math.log(10.)


class TR38901CPM:
    r"""
    Callable evaluating the cross-polarization matrix (CPM) of a scattering
    point of a sensing target, as specified in 3GPP TR 38.901, clause 7.9.2.2

    The callable returns the matrix

    .. math::
        \mathbf{W} = \begin{bmatrix}
                e^{j\Phi^{\theta\theta}} & \beta e^{j\Phi^{\theta\varphi}}\\
                \beta e^{j\Phi^{\varphi\theta}} & e^{j\Phi^{\varphi\varphi}}
            \end{bmatrix},
        \qquad \beta = \sqrt{\kappa^{-1}} = 10^{-\text{XPR}_{dB}/20}

    which is the cross-polarization matrix of eq. 7.9.2-5, whose
    cross-polarized entries are set by the XPR :math:`\kappa` of
    Table 7.9.2.2-1 and whose four initial phases are the ones of that
    equation. The cross-section the scattering point scatters is given
    separately by :class:`~sionna.rt.rcs.TR38901RCS`.

    Both the XPR and the phases are random, and are only drawn if
    ``random_xpr`` and ``random_phases`` are set, respectively:

    - When drawn, :math:`10\lg(\kappa)` is Gaussian with the mean ``xpr_db``
      and the standard deviation ``xpr_std_db`` of Table 7.9.2.2-1, as in
      eq. 7.9.4-3. It is otherwise fixed to ``xpr_db``, i.e., to its mean in
      dB.
    - When drawn, the four phases are uniform over :math:`(-\pi, \pi)`. They
      are otherwise zero, so the matrix is real and a scattering point does
      not randomize the phase of a path.

    Two evaluations of the same pair of
    directions with the same seed give the same matrix, and two runs
    of :class:`~sionna.rt.rcs.RCSSolver` on the same scene with the same
    seed give every path the same matrix. Exchanging the two directions
    transposes it, as clause 7.9.4 requires for monostatic sensing.

    :param xpr_db: :math:`10\lg(\kappa)` [dB], the cross-polarization ratio of
        eq. 7.9.2-5, and its mean in dB if ``random_xpr`` is set. Set to
        :py:class:`None` for a scattering point which does not depolarize,
        i.e., an infinite XPR.
    :param xpr_std_db: Standard deviation of :math:`10\lg(\kappa)` [dB], from
        Table 7.9.2.2-1. Only used if ``random_xpr`` is set.
    :param random_phases: If set to `True`, the four initial phases of
        eq. 7.9.2-5 are drawn for every pair of directions. They are otherwise
        zero.
    :param random_xpr: If set to `True`, the XPR is drawn for every pair of
        directions. It is otherwise fixed to ``xpr_db``.
    """

    def __init__(self,
                 xpr_db: float | None = None,
                 xpr_std_db: float = 0.,
                 random_phases: bool = False,
                 random_xpr: bool = False):

        self._xpr_db = None if xpr_db is None else float(xpr_db)
        self._xpr_std_db = float(xpr_std_db)
        if self._xpr_std_db < 0.:
            raise ValueError("`xpr_std_db` must be non-negative")

        # Ratio of the cross-polarized to the co-polarized amplitude when the
        # XPR is not drawn, in which case it only depends on its mean and is
        # therefore evaluated once. A scattering point with an infinite XPR
        # does not depolarize.
        self._beta = 0. if self._xpr_db is None \
                     else 10.**(-0.05*self._xpr_db)

        self.random_phases = random_phases
        self.random_xpr = random_xpr

    @property
    def xpr_db(self) -> float | None:
        r""":math:`10\lg(\kappa)` [dB] of eq. 7.9.2-5, or :py:class:`None` for
        a scattering point which does not depolarize

        :type: :py:class:`float` | :py:class:`None`
        """
        return self._xpr_db

    @property
    def xpr_std_db(self) -> float:
        r"""Standard deviation of :math:`10\lg(\kappa)` [dB], from Table
        7.9.2.2-1

        :type: :py:class:`float`
        """
        return self._xpr_std_db

    @property
    def random_phases(self) -> bool:
        """Get/set whether the four initial phases of eq. 7.9.2-5 are drawn
        for every pair of directions, rather than zero

        :type: :py:class:`bool`
        """
        return self._random_phases

    @random_phases.setter
    def random_phases(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("`random_phases` must be a bool")
        self._random_phases = value

    @property
    def random_xpr(self) -> bool:
        """Get/set whether the XPR is drawn for every pair of directions,
        rather than fixed to its mean in dB

        :type: :py:class:`bool`
        """
        return self._random_xpr

    @random_xpr.setter
    def random_xpr(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("`random_xpr` must be a bool")
        self._random_xpr = value

    def __call__(self,
                 k_i: mi.Vector3f,
                 k_s: mi.Vector3f,
                 seed: int = 0) -> Tuple[mi.Matrix2f, mi.Matrix2f]:
        r"""Evaluates the CPM for the given directions

        :param k_i: Incident directions of propagation, in the local
            coordinate system of the scattering point
        :param k_s: Scattered directions of propagation, in the local
            coordinate system of the scattering point
        :param seed: Seed of the draws of the XPR and of the phases, unused if
            neither ``random_xpr`` nor ``random_phases`` is set

        :return: Real part of the cross-polarization matrix
        :return: Imaginary part of the cross-polarization matrix
        """
        num_samples = dr.width(k_i)

        # A scattering point with an infinite XPR does not depolarize, whether
        # or not the XPR is drawn
        draw_xpr = self._random_xpr and (self._xpr_db is not None)

        if not (draw_xpr or self._random_phases):
            return self._real_matrix(self._beta, num_samples)

        # The first key is symmetric in the two directions and the last
        # two are exchanged with them, which is what makes the matrix
        # transpose when the directions are exchanged
        key, key_cross, key_cross_rev = direction_keys(k_i, k_s, seed)

        if draw_xpr:
            # Ratio of the cross-polarized to the co-polarized amplitude
            xpr_db = gaussian(key, STREAM_XPR, self._xpr_db, self._xpr_std_db)
            beta = dr.exp(-_LN_10_OVER_20*xpr_db)
        else:
            beta = self._beta

        if not self._random_phases:
            return self._real_matrix(beta, num_samples)

        # Initial phases of eq. 7.9.2-5. The two co-polarized ones are drawn
        # from the symmetric key, and the two cross-polarized ones from the
        # keys of the ordered pairs, so that they swap with the directions.
        sin_co, cos_co = dr.sincos(phase(key, STREAM_PHASE_THETA_THETA))
        sin_co_2, cos_co_2 = dr.sincos(phase(key, STREAM_PHASE_PHI_PHI))
        sin_cross, cos_cross = dr.sincos(phase(key_cross, STREAM_PHASE_CROSS))
        sin_cross_2, cos_cross_2 = \
            dr.sincos(phase(key_cross_rev, STREAM_PHASE_CROSS))

        return (mi.Matrix2f(cos_co, beta*cos_cross,
                            beta*cos_cross_2, cos_co_2),
                mi.Matrix2f(sin_co, beta*sin_cross,
                            beta*sin_cross_2, sin_co_2))

    @staticmethod
    def _real_matrix(beta: float | mi.Float,
                     num_samples: int) -> Tuple[mi.Matrix2f, mi.Matrix2f]:
        r"""Builds the CPM with zero initial phases

        :param beta: Ratio of the cross-polarized to the co-polarized
            amplitude
        :param num_samples: Number of samples

        :return: Real part of the cross-polarization matrix
        :return: Imaginary part of the cross-polarization matrix, i.e., zero
        """
        # The entries are real, and are broadcast to the number of samples
        # as the caller expects one matrix per sample
        zero = dr.zeros(mi.Float, num_samples)
        co, cross = zero + 1., zero + beta
        # The matrix is symmetric, so the ordering of its entries does not
        # matter
        return (mi.Matrix2f(co, cross, cross, co),
                dr.zeros(mi.Matrix2f, num_samples))
