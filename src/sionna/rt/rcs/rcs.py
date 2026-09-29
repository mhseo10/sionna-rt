#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Classes and functions related to RCS"""

from typing import Callable
import drjit as dr
import mitsuba as mi

from sionna.rt.registry import Registry

# An RCS of a scattering point is a real-valued function defined in the
# local coordinate system of the sensing target scattering point. It depends on
# both the incident direction and the scattered direction, as well as on a
# seed for the RCS models which have random components.
rcs_registry = Registry()

RCSCallable = Callable[[mi.Vector3f, mi.Vector3f, int], mi.Float]


def register_rcs(name: str, rcs_callable: RCSCallable):
    r"""Registers a new radar cross-section (RCS) callable

    An RCS is defined in the local coordinate system of a sensing target and
    depends on the incident direction :math:`\mathbf{k}_i` and the scattered
    direction :math:`\mathbf{k}_s`. It evaluates to the bistatic radar
    cross-section :math:`\sigma` [:math:`\text{m}^2`] of a scattering point
    for these directions.

    The cross-section sets the power a scattering point scatters, whereas the
    cross-polarization matrix (CPM) :math:`\mathbf{W}` of
    :func:`~sionna.rt.rcs.register_cpm` sets how that power is distributed
    over the polarization components. Together, the two characterize the
    scattering point completely through the Jones matrix

    .. math::
        \mathbf{J} = \sqrt{\sigma}\,\mathbf{W}

    of its transfer function, which
    :meth:`~sionna.rt.rcs.ScatteringPoints.eval_jones_matrix` assembles.

    Following :eq:`scattered_field`, a scattering point illuminated by an
    incident field :math:`\mathbf{E}_i` scatters the field

    .. math::
        \mathbf{E}_s = \frac{1}{r}\mathbf{J}\mathbf{E}_i

    at a distance :math:`r` from the point, the :math:`1/r` and the spreading
    of the scattered power over the sphere being supplied by the propagation
    of the scattered field. The entries of :math:`\mathbf{J}` therefore have
    units of meters, and those of :math:`\sigma` units of square meters.

    An RCS is also given the seed of the evaluation, which the models with
    random components use to draw them.

    The following snippet registers an RCS which is largest in the
    backscattering direction and vanishes in the forward direction, and
    assigns it by name to a scattering point:

    .. code-block:: python

        import drjit as dr
        from sionna.rt.rcs import ConstantCPM, ScatteringModel, register_rcs

        # 10 m^2 back towards the direction the wave comes from, i.e., for
        # k_s = -k_i, decreasing to 0 m^2 in the forward direction k_s = k_i.
        # The seed is ignored, as this cross-section is deterministic.
        def backscattering_rcs(k_i, k_s, seed):
            return 5.*(1. - dr.dot(k_i, k_s))

        register_rcs("backscattering", backscattering_rcs)

        model = ScatteringModel([0, 0, 0], rcs="backscattering",
                                cpm=ConstantCPM())

    :param name: Name of the RCS callable
    :param rcs_callable: Callable with signature
        ``(k_i, k_s, seed) -> mi.Float``, returning the bistatic radar
        cross-section :math:`\sigma` [:math:`\text{m}^2`]
    """
    if not isinstance(name, str):
        raise ValueError("`name` must be a string")
    if not callable(rcs_callable):
        raise ValueError("`rcs_callable` must be a callable")
    rcs_registry.register(rcs_callable, name)


def unregister_rcs(name: str):
    """Unregisters an RCS callable

    :param name: Name of the RCS callable
    """
    if not isinstance(name, str):
        raise ValueError("`name` must be a string")
    rcs_registry.unregister(name)


def get_rcs(name: str) -> RCSCallable:
    """Returns a registered RCS callable

    :param name: Name of the RCS callable
    :return: Registered RCS callable
    """
    if not isinstance(name, str):
        raise ValueError("`name` must be a string")
    try:
        return rcs_registry.get(name)
    except KeyError as err:
        raise ValueError(f"RCS callable {name} not found") from err


def _resolve_rcs(rcs: str | RCSCallable) -> RCSCallable:
    """Resolves an RCS name or callable to a callable

    :param rcs: Registered RCS name or RCS callable
    :return: RCS callable
    """
    if isinstance(rcs, str):
        return get_rcs(rcs)
    if callable(rcs):
        return rcs
    raise ValueError("RCS must be callable or string")


class ConstantRCS:
    r"""
    Callable evaluating a radar cross-section that does not depend on the
    incident and scattered directions

    The callable returns the bistatic radar cross-section :math:`\sigma`
    [:math:`\text{m}^2`] of the scattering point for every pair of incident
    and scattered directions.

    :param sigma: Bistatic radar cross-section
        :math:`\sigma` [:math:`\text{m}^2`]
    """

    def __init__(self, sigma: float | mi.Float = 1.):

        self.sigma = sigma

    @property
    def sigma(self):
        r"""Get/set the bistatic radar cross-section
        :math:`\sigma` [:math:`\text{m}^2`]

        :type: :py:class:`mi.Float`
        """
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if sigma < 0.:
            raise ValueError("Cross-section must be non-negative")
        self._sigma = mi.Float(sigma)

    def __call__(self,
                 k_i: mi.Vector3f,
                 k_s: mi.Vector3f,
                 seed: int = 0) -> mi.Float:
        r"""Evaluates the cross-section for the given directions

        :param k_i: Incident directions of propagation, in the local
            coordinate system of the scattering point
        :param k_s: Scattered directions of propagation, in the local
            coordinate system of the scattering point
        :param seed: Ignored, as this cross-section is deterministic

        :return: Bistatic radar cross-section [:math:`\text{m}^2`]
        """
        # The cross-section is constant, and is broadcast to the number of
        # samples as the caller expects one value per sample
        return dr.zeros(mi.Float, dr.width(k_i)) + self._sigma
