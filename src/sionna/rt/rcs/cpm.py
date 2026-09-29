#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Classes and functions related to cross-polarization matrices"""

import math
from typing import Callable, Tuple
import drjit as dr
import mitsuba as mi

from sionna.rt.registry import Registry

# A CPM of a scattering point is a matrix-valued function defined in the local
# coordinate system of the sensing target scattering point. It depends on both
# the incident direction and the scattered direction, as well as on a seed for
# the CPM models which have random components.
cpm_registry = Registry()

CPMCallable = Callable[[mi.Vector3f, mi.Vector3f, int],
                       Tuple[mi.Matrix2f, mi.Matrix2f]]

# Dr.Jit provides no base-10 exponential
_LN_10_OVER_20 = 0.05*math.log(10.)


def register_cpm(name: str, cpm_callable: CPMCallable):
    r"""Registers a new cross-polarization matrix (CPM) callable

    A CPM is defined in the local coordinate system of a sensing target and
    depends on the incident direction :math:`\mathbf{k}_i` and the scattered
    direction :math:`\mathbf{k}_s`. It evaluates to the complex
    :math:`2 \times 2` matrix :math:`\mathbf{W}`, returned as the real and
    imaginary parts of that matrix.

    The CPM describes how a scattering point transforms the polarization of
    the incident field, whereas the radar cross-section :math:`\sigma` of
    :func:`~sionna.rt.rcs.register_rcs` sets the power it scatters. Together,
    the two define the Jones matrix
    :math:`\mathbf{J} = \sqrt{\sigma}\,\mathbf{W}` of the transfer function of
    the scattering point, which
    :meth:`~sionna.rt.rcs.ScatteringPoints.eval_jones_matrix` assembles.

    A CPM is also given the seed of the evaluation, which the models with
    random components use to draw them.

    The following snippet registers a CPM which turns each linearly polarized
    component of the incident field into a circularly polarized one, and
    assigns it by name to a scattering point:

    .. code-block:: python

        import drjit as dr
        import mitsuba as mi
        from sionna.rt.rcs import ConstantRCS, ScatteringModel, register_cpm

        # W = [[1, j], [j, 1]]/sqrt(2), a unitary matrix which leaves the
        # scattered power set by the RCS unchanged. The seed is ignored, as
        # this matrix is deterministic.
        def circular_cpm(k_i, k_s, seed):
            zero = dr.zeros(mi.Float, dr.width(k_i))
            a = zero + 0.5**0.5
            return mi.Matrix2f(a, zero, zero, a), mi.Matrix2f(zero, a, a, zero)

        register_cpm("circular", circular_cpm)

        model = ScatteringModel([0, 0, 0], rcs=ConstantRCS(sigma=1.),
                                cpm="circular")

    :param name: Name of the CPM callable
    :param cpm_callable: Callable with signature
        ``(k_i, k_s, seed) -> (mi.Matrix2f, mi.Matrix2f)``, returning the real
        and imaginary parts of the cross-polarization matrix
    """
    if not isinstance(name, str):
        raise ValueError("`name` must be a string")
    if not callable(cpm_callable):
        raise ValueError("`cpm_callable` must be a callable")
    cpm_registry.register(cpm_callable, name)


def unregister_cpm(name: str):
    """Unregisters a CPM callable

    :param name: Name of the CPM callable
    """
    if not isinstance(name, str):
        raise ValueError("`name` must be a string")
    cpm_registry.unregister(name)


def get_cpm(name: str) -> CPMCallable:
    """Returns a registered CPM callable

    :param name: Name of the CPM callable
    :return: Registered CPM callable
    """
    if not isinstance(name, str):
        raise ValueError("`name` must be a string")
    try:
        return cpm_registry.get(name)
    except KeyError as err:
        raise ValueError(f"CPM callable {name} not found") from err


def _resolve_cpm(cpm: str | CPMCallable) -> CPMCallable:
    """Resolves a CPM name or callable to a callable

    :param cpm: Registered CPM name or CPM callable
    :return: CPM callable
    """
    if isinstance(cpm, str):
        return get_cpm(cpm)
    if callable(cpm):
        return cpm
    raise ValueError("CPM must be callable or string")


class ConstantCPM:
    r"""
    Callable evaluating a cross-polarization matrix that does not depend on
    the incident and scattered directions

    The callable returns the real matrix

    .. math::
        \mathbf{W} = \frac{1}{\sqrt{1+\beta^2}}
            \begin{bmatrix}
                1 & -\beta\\
                \beta & 1
            \end{bmatrix},
        \qquad \beta = 10^{-\text{XPR}_{dB}/20}

    where :math:`\text{XPR}_{dB}` is the cross-polarization ratio [dB] of the
    scattering point, i.e., the ratio of the co-polarized to the
    cross-polarized scattered power. The matrix is a rotation and hence
    unitary, so it satisfies the normalization
    :math:`\lVert\mathbf{W}\rVert_F^2 = 2`.

    Its entries are the ones of the diffuse scattering matrix of
    :eq:`scattered_field`, with the random phase shifts dropped and the
    cross-polarization discrimination coefficient

    .. math::
        K_x = \frac{1}{1 + 10^{\text{XPR}_{dB}/10}}\in[0,1]

    of :eq:`xpd` written in terms of the XPR, as :math:`\sqrt{1-K_x}` and
    :math:`\sqrt{K_x}` are the co-polarized and cross-polarized entries
    above. With an infinite XPR, the scattering point preserves the
    polarization of the incident field, with :math:`\text{XPR}_{dB} = 0` it
    splits the scattered power evenly over both polarization components, and
    as the XPR tends to :math:`-\infty` it scatters the power entirely on the
    cross-polarized component.

    :param xpr_db: Cross-polarization ratio :math:`\text{XPR}_{dB}` [dB]. Set
        to :py:class:`None` for a scattering point which does not depolarize,
        i.e., an infinite XPR.
    """

    def __init__(self, xpr_db: float | mi.Float | None = None):

        self.xpr_db = xpr_db

    @property
    def xpr_db(self):
        r"""Get/set the cross-polarization ratio
        :math:`\text{XPR}_{dB}` [dB], or :py:class:`None` for a scattering
        point which does not depolarize

        :type: :py:class:`mi.Float` | :py:class:`None`
        """
        return self._xpr_db

    @xpr_db.setter
    def xpr_db(self, xpr_db):
        self._xpr_db = None if xpr_db is None else mi.Float(xpr_db)

    def __call__(self,
                 k_i: mi.Vector3f,
                 k_s: mi.Vector3f,
                 seed: int = 0) -> Tuple[mi.Matrix2f, mi.Matrix2f]:
        r"""Evaluates the CPM for the given directions

        :param k_i: Incident directions of propagation, in the local
            coordinate system of the scattering point
        :param k_s: Scattered directions of propagation, in the local
            coordinate system of the scattering point
        :param seed: Ignored, as this matrix is deterministic

        :return: Real part of the cross-polarization matrix
        :return: Imaginary part of the cross-polarization matrix
        """
        # The entries are constant, and are broadcast to the number of
        # samples as the caller expects one matrix per sample
        zero = dr.zeros(mi.Float, dr.width(k_i))

        if self._xpr_db is None:
            co, cross = zero + 1., zero
        else:
            # Ratio of the cross-polarized to the co-polarized amplitude,
            # which the normalization brings back to a unitary matrix
            beta = dr.exp(-_LN_10_OVER_20*self._xpr_db)
            norm = dr.rcp(dr.sqrt(1. + dr.square(beta)))
            co, cross = zero + norm, zero + beta*norm

        return (mi.Matrix2f(co, -cross, cross, co),
                dr.zeros(mi.Matrix2f, dr.width(k_i)))
