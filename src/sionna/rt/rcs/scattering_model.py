#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Classes and functions related to Scattering Model"""

import mitsuba as mi

from .cpm import CPMCallable
from .rcs import RCSCallable
from .scattering_points import ScatteringPoints


class ScatteringModel:
    r"""
    Class defining the scattering model of a sensing target

    A scattering model consists of a collection of scattering points, given in
    the local coordinate system (LCS) of the sensing target it is attached to.
    Scattering points can be added and removed at any time.

    Predefined scattering models are built by deriving from this class and
    adding the scattering points of the model in the constructor:

    .. code-block:: python

        class MyScatteringModel(ScatteringModel):
            def __init__(self):
                super().__init__()
                self.add_scattering_points(mi.Point3f([0, 1], [0, 0], [0, 0]),
                                           rcs="my_rcs", cpm="my_cpm")

    :param lcs_positions: Positions of the scattering points in the LCS [m].
        If set to :py:class:`None`, the model is created without scattering
        points.

    :param lcs_orientations: Orientations of the scattering points in the LCS,
        specified through three angles :math:`(\alpha, \beta, \gamma)`
        corresponding to a 3D rotation as defined in :eq:`rotation`.
        Defaults to :math:`(0,0,0)` for every point.

    :param rcs: RCS of each scattering point, given as a list with one RCS
        callable or registered RCS name per point. A single callable or name
        is broadcast to all points.

    :param cpm: CPM of each scattering point, given as a list with one CPM
        callable or registered CPM name per point. A single callable or name
        is broadcast to all points. Use
        :class:`~sionna.rt.rcs.ConstantCPM` for points which do not
        depolarize.
    """

    def __init__(self,
                 lcs_positions: mi.Point3f | None = None,
                 lcs_orientations: mi.Point3f | None = None,
                 rcs: list[str | RCSCallable] | str | RCSCallable | None
                        = None,
                 cpm: list[str | CPMCallable] | str | CPMCallable | None
                        = None):

        self._spst = ScatteringPoints()

        if lcs_positions is not None:
            self.add_scattering_points(lcs_positions, lcs_orientations, rcs,
                                       cpm)

    @property
    def spst(self):
        """
        Scattering points of the model

        :type: :class:`~sionna.rt.rcs.ScatteringPoints`
        """
        return self._spst

    def add_scattering_points(self,
                              lcs_positions: mi.Point3f,
                              lcs_orientations: mi.Point3f | None = None,
                              rcs: list[str | RCSCallable] | str
                                    | RCSCallable | None = None,
                              cpm: list[str | CPMCallable] | str
                                    | CPMCallable | None = None):
        r"""Adds scattering points to this model

        The new points are appended to the ones of this model.

        :param lcs_positions: Positions of the scattering points in the LCS [m]

        :param lcs_orientations: Orientations of the scattering points in the
            LCS, specified through three angles
            :math:`(\alpha, \beta, \gamma)`. Defaults to :math:`(0,0,0)` for
            every point.

        :param rcs: RCS of each scattering point, given as a list with one RCS
            callable or registered RCS name per point. A single callable or
            name is broadcast to all points.

        :param cpm: CPM of each scattering point, given as a list with one CPM
            callable or registered CPM name per point. A single callable or
            name is broadcast to all points. Use
            :class:`~sionna.rt.rcs.ConstantCPM` for points which do not
            depolarize.
        """
        if lcs_orientations is None:
            lcs_orientations = mi.Point3f()

        added = ScatteringPoints(mi.Point3f(lcs_positions),
                                 mi.Point3f(lcs_orientations),
                                 rcs,
                                 cpm)
        self._spst = self._spst.concat(added)

    def remove_scattering_points(self, indices: int | list[int]):
        """Removes scattering points from this model

        The remaining points keep their relative order, and so do their
        callables, which are re-indexed accordingly.

        :param indices: Indices of the scattering points to remove
        """
        self._spst = self._spst.remove(indices)
