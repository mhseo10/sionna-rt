#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Classes and functions related to Scattering Points"""

from dataclasses import dataclass, field, InitVar
from typing import Callable, Self, Tuple
import drjit as dr
import mitsuba as mi

from sionna.rt.utils import concat_points, rotation_matrix

from .cpm import CPMCallable, _resolve_cpm
from .rcs import RCSCallable, _resolve_rcs


@dataclass
class ScatteringPoints:
    r"""
    Collection of scattering points of a sensing target

    Positions and orientations are stored in the local coordinate system (LCS)
    of the sensing target as Dr.Jit arrays.

    Every scattering point has exactly one RCS callable and one CPM callable,
    both of which are required. They are stored in ``rcs_callables`` and
    ``cpm_callables``, in the same order as the points, so the index of a
    point is also the index of its callables.

    :param lcs_positions: Positions of the scattering points in the LCS [m]
    :param lcs_orientations: Orientations of the scattering points in the LCS,
        specified through three angles :math:`(\alpha, \beta, \gamma)`
    :param rcs: RCS of each scattering point, given as a list with one RCS
        callable or registered RCS name per point. A single callable or name
        is broadcast to all points.
    :param cpm: CPM of each scattering point, given as a list with one CPM
        callable or registered CPM name per point. A single callable or name
        is broadcast to all points. Use
        :class:`~sionna.rt.rcs.ConstantCPM` for points which do not
        depolarize.
    """

    lcs_positions: mi.Point3f = field(default_factory=mi.Point3f)
    lcs_orientations: mi.Point3f = field(default_factory=mi.Point3f)
    rcs: InitVar[list[str | RCSCallable] | str | RCSCallable | None] = None
    cpm: InitVar[list[str | CPMCallable] | str | CPMCallable | None] = None

    # RCS callable of each scattering point, in point order
    rcs_callables: list[RCSCallable] = field(default_factory=list, init=False)

    # CPM callable of each scattering point, in point order
    cpm_callables: list[CPMCallable] = field(default_factory=list, init=False)

    def __post_init__(self, rcs, cpm):
        num_points = dr.width(self.lcs_positions)
        if dr.width(self.lcs_orientations) == 0:
            self.lcs_orientations = dr.zeros_like(self.lcs_positions)
        if num_points != dr.width(self.lcs_orientations):
            raise ValueError("`lcs_positions` and `lcs_orientations` must "
                             "have the same width")

        self.rcs_callables = _resolve_callables(rcs, num_points, "rcs", "RCS",
                                                _resolve_rcs)
        self.cpm_callables = _resolve_callables(cpm, num_points, "cpm", "CPM",
                                                _resolve_cpm)

    def concat(self, other: Self | list[Self]) -> Self:
        """Returns a new collection that combines this one with ``other``

        The points of ``other`` are appended to the ones of this collection,
        and so are their callables. If ``other`` is a list, its collections
        are appended in the order in which they are given.

        :param other: Scattering points to be concatenated
        :return: Concatenated scattering points
        """
        others = other if isinstance(other, list) else [other]
        for other_ in others:
            if not isinstance(other_, ScatteringPoints):
                raise ValueError("`other` must be a ScatteringPoints instance"
                                 " or a list of ScatteringPoints instances")

        collections = [self] + others
        concatenated = ScatteringPoints()
        concatenated.lcs_positions = \
            concat_points([c.lcs_positions for c in collections])
        concatenated.lcs_orientations = \
            concat_points([c.lcs_orientations for c in collections])
        concatenated.rcs_callables = [rcs for c in collections
                                      for rcs in c.rcs_callables]
        concatenated.cpm_callables = [cpm for c in collections
                                      for cpm in c.cpm_callables]
        return concatenated

    def scaled_lcs_positions(self,
                             st_scalings: mi.Vector3f,
                             st_indices: mi.UInt) -> mi.Point3f:
        r"""Returns the positions of the scattering points in the local
        coordinate system (LCS) of their sensing target, scaled by the scaling
        of that target

        The positions of this collection are given in the unscaled LCS of the
        sensing targets, i.e., in the frame in which the scattering model was
        defined, whereas scaling a target resizes its mesh. The positions are
        therefore scaled along with it, so that a scattering point keeps its
        place relative to the geometry of its target:

        .. math::
            \mathbf{p} = \mathbf{s}_{\text{st}} \odot \mathbf{p}_{\text{lcs}}

        where :math:`\odot` is the element-wise product.

        :param st_scalings: Scalings of the sensing targets
        :param st_indices: Index of the sensing target of each scattering point
            of this collection
        :return: Scaled positions of the scattering points in the LCS [m]
        """
        st_scale = dr.gather(mi.Vector3f, st_scalings, st_indices)
        return mi.Point3f(st_scale*self.lcs_positions)

    def gcs_positions(self,
                      st_positions: mi.Point3f,
                      st_orientations: mi.Point3f,
                      st_indices: mi.UInt,
                      st_scalings: mi.Vector3f | None = None) -> mi.Point3f:
        r"""Returns the positions of the scattering points in the global
        coordinate system (GCS)

        The positions of this collection are given in the local frame of the
        sensing targets, so they are scaled, rotated, and translated to the GCS
        using the to-world transformation of the sensing target of every point:

        .. math::
            \mathbf{p} = \mathbf{p}_{\text{st}}
                + \mathbf{R}(\text{st\_orientation})
                  \left( \mathbf{s}_{\text{st}}
                         \odot \mathbf{p}_{\text{lcs}} \right)

        where :math:`\mathbf{R}()` is defined in :eq:`rotation` and
        :math:`\odot` is the element-wise product.

        :param st_positions: Positions of the sensing targets in the GCS [m]
        :param st_orientations: Orientations of the sensing targets in the GCS,
            specified through three angles :math:`(\alpha, \beta, \gamma)`
        :param st_indices: Index of the sensing target of each scattering point
            of this collection
        :param st_scalings: Scalings of the sensing targets. If set to
            :py:class:`None`, the targets are assumed to be unscaled.
        :return: Positions of the scattering points in the GCS [m]
        """
        st_pos = dr.gather(mi.Point3f, st_positions, st_indices)
        st_ort = dr.gather(mi.Point3f, st_orientations, st_indices)
        lcs_positions = (self.lcs_positions if st_scalings is None
                         else self.scaled_lcs_positions(st_scalings,
                                                        st_indices))
        return mi.Point3f(st_pos + rotation_matrix(st_ort)@lcs_positions)

    def remove(self, indices: int | list[int]) -> Self:
        """Returns a new collection without the points at ``indices``

        The remaining points keep their relative order, and so do their
        callables, which are re-indexed accordingly.

        :param indices: Indices of the scattering points to remove
        :return: Scattering points without the removed ones
        """
        num_points = dr.width(self.lcs_positions)

        if isinstance(indices, int):
            indices = [indices]
        elif not isinstance(indices, list):
            raise ValueError("`indices` must be an integer or a list of "
                             "integers")
        if any(not isinstance(index, int) for index in indices):
            raise ValueError("`indices` must be an integer or a list of "
                             "integers")
        if any((index < 0) or (index >= num_points) for index in indices):
            raise ValueError("`indices` must be in the range "
                             f"[0, {num_points})")

        to_remove = set(indices)
        keep = [index for index in range(num_points)
                if index not in to_remove]

        reduced = ScatteringPoints()
        if keep:
            keep_indices = mi.UInt(keep)
            reduced.lcs_positions = dr.gather(mi.Point3f, self.lcs_positions,
                                              keep_indices)
            reduced.lcs_orientations = dr.gather(mi.Point3f,
                                                 self.lcs_orientations,
                                                 keep_indices)
        reduced.rcs_callables = [self.rcs_callables[index] for index in keep]
        reduced.cpm_callables = [self.cpm_callables[index] for index in keep]
        return reduced

    def eval_rcs(self,
                 k_i: mi.Vector3f,
                 k_s: mi.Vector3f,
                 st_orientations: mi.Point3f,
                 st_indices: mi.UInt,
                 spst_indices: mi.UInt,
                 seed: int = 0) -> mi.Float:
        r"""Evaluates, for every sample, the RCS of the scattering point it is
        associated with

        The directions ``k_i`` and ``k_s`` are given in the global coordinate
        system (GCS), whereas an RCS is defined in the local frame of its
        scattering point. The directions are therefore rotated to that frame
        before the RCS is evaluated, using the to-world transformation

        .. math::
            \mathbf{R}(\text{st\_orientation})
            \mathbf{R}(\text{lcs\_orientation})

        where :math:`\mathbf{R}()` is defined in :eq:`rotation`.

        :param k_i: Incident directions in the GCS
        :param k_s: Scattered directions in the GCS
        :param st_orientations: Orientations of the sensing targets in the GCS,
            specified through three angles :math:`(\alpha, \beta, \gamma)`
        :param st_indices: Index of the sensing target of each sample into
            ``st_orientations``
        :param spst_indices: Index of the scattering point of each sample into
            this collection
        :param seed: Seed given to the RCS callables, which the ones with
            random components use to draw them
        :return: Bistatic radar cross-section of each sample
            [:math:`\text{m}^2`]
        """
        if len(self.rcs_callables) == 0:
            return dr.zeros(mi.Float)

        k_i_local, k_s_local = self._local_directions(k_i, k_s,
                                                      st_orientations,
                                                      st_indices,
                                                      spst_indices)
        return _eval_callables(self.rcs_callables, spst_indices,
                               k_i_local, k_s_local, seed)

    def eval_cpm(self,
                 k_i: mi.Vector3f,
                 k_s: mi.Vector3f,
                 st_orientations: mi.Point3f,
                 st_indices: mi.UInt,
                 spst_indices: mi.UInt,
                 seed: int = 0) -> Tuple[mi.Matrix2f, mi.Matrix2f]:
        r"""Evaluates, for every sample, the CPM of the scattering point it is
        associated with

        As with :meth:`~sionna.rt.rcs.ScatteringPoints.eval_rcs`, the
        directions are given in the GCS and are rotated to the local frame of
        the scattering point before the CPM is evaluated.

        :param k_i: Incident directions in the GCS
        :param k_s: Scattered directions in the GCS
        :param st_orientations: Orientations of the sensing targets in the GCS,
            specified through three angles :math:`(\alpha, \beta, \gamma)`
        :param st_indices: Index of the sensing target of each sample into
            ``st_orientations``
        :param spst_indices: Index of the scattering point of each sample into
            this collection
        :param seed: Seed given to the CPM callables, which the ones with
            random components use to draw them
        :return: Real part of the cross-polarization matrix of each sample
        :return: Imaginary part of the cross-polarization matrix of each sample
        """
        if len(self.cpm_callables) == 0:
            return dr.zeros(mi.Matrix2f), dr.zeros(mi.Matrix2f)

        k_i_local, k_s_local = self._local_directions(k_i, k_s,
                                                      st_orientations,
                                                      st_indices,
                                                      spst_indices)
        return _eval_callables(self.cpm_callables, spst_indices,
                               k_i_local, k_s_local, seed)

    def eval_jones_matrix(
        self,
        k_i: mi.Vector3f,
        k_s: mi.Vector3f,
        st_orientations: mi.Point3f,
        st_indices: mi.UInt,
        spst_indices: mi.UInt,
        seed: int = 0
    ) -> Tuple[mi.Matrix2f, mi.Matrix2f]:
        r"""Evaluates, for every sample, the Jones matrix of the scattering
        point it is associated with

        The Jones matrix of the transfer function of a scattering point
        combines its RCS :math:`\sigma` and its cross-polarization matrix
        :math:`\mathbf{W}` as

        .. math::
            \mathbf{J} = \sqrt{\sigma}\,\mathbf{W}

        so this returns the real and imaginary parts of a complex
        :math:`2 \times 2` matrix, with one matrix per sample. As with
        :meth:`~sionna.rt.rcs.ScatteringPoints.eval_rcs`, the directions are
        given in the GCS and are rotated to the local frame of the scattering
        point before the RCS and the CPM are evaluated.

        :param k_i: Incident directions in the GCS
        :param k_s: Scattered directions in the GCS
        :param st_orientations: Orientations of the sensing targets in the GCS,
            specified through three angles :math:`(\alpha, \beta, \gamma)`
        :param st_indices: Index of the sensing target of each sample into
            ``st_orientations``
        :param spst_indices: Index of the scattering point of each sample into
            this collection
        :param seed: Seed given to the RCS and CPM callables, which the ones
            with random components use to draw them
        :return: Real part of the Jones matrix of each sample
        :return: Imaginary part of the Jones matrix of each sample
        """
        if len(self.rcs_callables) == 0:
            return dr.zeros(mi.Matrix2f), dr.zeros(mi.Matrix2f)

        k_i_local, k_s_local = self._local_directions(k_i, k_s,
                                                      st_orientations,
                                                      st_indices,
                                                      spst_indices)
        sigma = _eval_callables(self.rcs_callables, spst_indices,
                                k_i_local, k_s_local, seed)
        cpm_real, cpm_imag = _eval_callables(self.cpm_callables, spst_indices,
                                             k_i_local, k_s_local, seed)

        amplitude = dr.sqrt(sigma)
        return amplitude*cpm_real, amplitude*cpm_imag

    def _local_directions(
        self,
        k_i: mi.Vector3f,
        k_s: mi.Vector3f,
        st_orientations: mi.Point3f,
        st_indices: mi.UInt,
        spst_indices: mi.UInt
    ) -> Tuple[mi.Vector3f, mi.Vector3f]:
        r"""Rotates directions from the GCS to the local frames of the
        scattering points

        :param k_i: Incident directions in the GCS
        :param k_s: Scattered directions in the GCS
        :param st_orientations: Orientations of the sensing targets in the GCS,
            specified through three angles :math:`(\alpha, \beta, \gamma)`
        :param st_indices: Index of the sensing target of each sample into
            ``st_orientations``
        :param spst_indices: Index of the scattering point of each sample into
            this collection
        :return: Incident directions in the local frames
        :return: Scattered directions in the local frames
        """
        # World-to-local transformation of the scattering point of each sample
        st_ort = dr.gather(mi.Point3f, st_orientations, st_indices)
        spst_ort = dr.gather(mi.Point3f, self.lcs_orientations, spst_indices)
        to_local = (rotation_matrix(st_ort)@rotation_matrix(spst_ort)).T

        return to_local@mi.Vector3f(k_i), to_local@mi.Vector3f(k_s)


def _resolve_callables(callables: list[str | Callable] | str | Callable | None,
                       num_points: int,
                       name: str,
                       kind: str,
                       resolve: Callable) -> list[Callable]:
    """Resolves the callables given for a collection of scattering points

    A single callable or registered name is broadcast to every point, and one
    callable per point is required, so a collection which holds points rejects
    ``None``.

    :param callables: Callables of the points, given as a list with one
        callable or registered name per point, as a single callable or name,
        or as :py:class:`None`
    :param num_points: Number of scattering points of the collection
    :param name: Name of the parameter, as it appears in error messages
    :param kind: Kind of the callables, as it appears in error messages
    :param resolve: Function resolving a name or callable to a callable
    :return: Callable of every scattering point, in point order
    """
    if callables is None:
        callables = []
    elif isinstance(callables, str) or callable(callables):
        callables = [callables]*num_points
    elif not isinstance(callables, list):
        raise ValueError(f"`{name}` must be a list, a callable, or a "
                         f"registered {kind} name")
    if len(callables) != num_points:
        raise ValueError(f"`{name}` must provide one entry per scattering "
                         f"point (expected {num_points}, got "
                         f"{len(callables)})")

    return [resolve(callable_) for callable_ in callables]


def _eval_callables(callables: list[Callable],
                    spst_indices: mi.UInt,
                    k_i: mi.Vector3f,
                    k_s: mi.Vector3f,
                    seed: int):
    """Evaluates, for every sample, the callable of the scattering point it is
    associated with

    Every sample only evaluates the callable of its own scattering point. When
    all the points share the same callable, as with a single point or with a
    uniform CPM, that callable is evaluated directly rather than dispatched.

    :param callables: Callable of every scattering point, in point order
    :param spst_indices: Index of the scattering point of each sample
    :param k_i: Incident directions in the local frames
    :param k_s: Scattered directions in the local frames
    :param seed: Seed given to the callables, which the ones with random
        components use to draw them
    :return: Value returned by the callable of every sample
    """
    if all(callable_ is callables[0] for callable_ in callables[1:]):
        return callables[0](k_i, k_s, seed)

    return dr.switch(spst_indices, callables, k_i, k_s, seed)
