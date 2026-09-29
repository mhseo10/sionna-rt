#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Scattering model of 3GPP TR 38.901, clause 7.9.2"""

from dataclasses import replace
from typing import List, Sequence, Tuple
import drjit as dr
import mitsuba as mi

from ..scattering_model import ScatteringModel
from .cpm import TR38901CPM
from .parameters import LobeParameters, MODEL_2, get_target_parameters
from .rcs import TR38901RCS


class TR38901ScatteringModel(ScatteringModel):
    # pylint: disable=line-too-long
    r"""
    Scattering model of a sensing target, as specified in 3GPP TR 38.901,
    clause 7.9.2

    The type of sensing target determines both the number of scattering points
    of the model and the sets of parameters of every point, which are taken
    from Tables 7.9.2.1-1 to 7.9.2.1-7. A table row is a set of parameters
    rather than a scattering point: the models with a single scattering point
    hold every row as a lobe of that point and select one of them from the
    bisector angle, whereas the models with multiple scattering points give
    one row to every point, all of them contributing simultaneously.

    The specifications only leave a choice between the two for the vehicle and
    the AGV, hence the ``-single-sp`` and ``-multi-sp`` object types:

    .. list-table::
        :header-rows: 1

        * - ``object_type``
          - ``model_type``
          - Scattering points
          - Table
        * - ``"uav-small-size"``
          - 1
          - 1
          - 7.9.2.1-1
        * - ``"human"``
          - 1 or 2
          - 1
          - 7.9.2.1-1 or 7.9.2.1-3
        * - ``"uav-large-size"``
          - 2
          - 1
          - 7.9.2.1-2
        * - ``"vehicle-single-sp"``
          - 2
          - 1
          - 7.9.2.1-4
        * - ``"vehicle-multi-sp"``
          - 2
          - 5
          - 7.9.2.1-5
        * - ``"agv-single-sp"``
          - 2
          - 1
          - 7.9.2.1-6
        * - ``"agv-multi-sp"``
          - 2
          - 5
          - 7.9.2.1-7

    Every scattering point depolarizes according to the cross-polarization
    matrix of clause 7.9.2.2, with the XPR of Table 7.9.2.2-1, as detailed in
    :class:`~sionna.rt.rcs.TR38901CPM`.

    The random components of the two models, i.e., the RCS component
    :math:`\sigma_S` of clause 7.9.2.1 and the XPR and initial phases of
    clause 7.9.2.2, are disabled by default and are enabled through
    ``random_sigma_s``, ``random_xpr`` and ``random_phases``. They are drawn
    per pair of incident and scattered directions from the seed given to
    :meth:`~sionna.rt.rcs.RCSSolver.__call__`, as detailed in
    :class:`~sionna.rt.rcs.TR38901RCS` and
    :class:`~sionna.rt.rcs.TR38901CPM`.

    This model is created by
    :class:`~sionna.rt.rcs.TR38901SensingTarget`, which determines the target
    dimensions and places the scattering points. Its flags can be set when
    constructing that target, which forwards them to the model, and changed
    at any time afterwards through the
    :attr:`~sionna.rt.rcs.SensingTarget.scattering_model` of the target.

    :param object_type: Type of sensing target, one of ``"uav-small-size"``,
        ``"uav-large-size"``, ``"human"``, ``"vehicle-single-sp"``,
        ``"vehicle-multi-sp"``, ``"agv-single-sp"``, ``"agv-multi-sp"``

    :param model_type: RCS model of clause 7.9.2.1, either 1 or 2. RCS model 1
        fixes the angular component :math:`\sigma_D` of the monostatic RCS to
        1, and is only defined for the small UAV and the human.

    :param lcs_positions: Positions of the scattering points, resolved from
        the geometry of the owning target [m]

    :param random_sigma_s: If set to `True`, the RCS component
        :math:`\sigma_S` of clause 7.9.2.1 is drawn for every pair of
        directions. Defaults to `False`.

    :param random_phases: If set to `True`, the four initial phases of
        eq. 7.9.2-5 are drawn for every pair of directions. Defaults to
        `False`.

    :param random_xpr: If set to `True`, the XPR of eq. 7.9.2-5 is drawn for
        every pair of directions. Defaults to `False`.

    Example
    -------
    .. code-block:: python

        from sionna.rt.rcs import TR38901SensingTarget

        # Enable the random initial phases when constructing the target
        target = TR38901SensingTarget(name="st",
                                      object_type="vehicle-multi-sp",
                                      random_phases=True)

        # Enable the random XPR afterwards, through the scattering model
        target.scattering_model.random_xpr = True
    """

    def __init__(self,
                 object_type: str,
                 model_type: int = MODEL_2,
                 lcs_positions: mi.Point3f | None = None,
                 random_sigma_s: bool = False,
                 random_phases: bool = False,
                 random_xpr: bool = False):

        parameters = get_target_parameters(object_type, model_type)
        if lcs_positions is None:
            raise ValueError("`lcs_positions` must be provided by a "
                             "TR38901SensingTarget")
        lcs_positions = mi.Point3f(lcs_positions)
        if dr.width(lcs_positions) != parameters.num_scattering_points:
            raise ValueError(
                f"`lcs_positions` must contain "
                f"{parameters.num_scattering_points} point(s) for "
                f"'{object_type}'")

        self._object_type = object_type
        self._model_type = model_type
        self._parameters = parameters
        self._random_sigma_s = random_sigma_s
        self._random_phases = random_phases
        self._random_xpr = random_xpr

        positions, orientations, rcs, cpm = \
            self._build_scattering_points(lcs_positions)

        # The callables are kept so that the flags can be written through to
        # them, the CPM being shared by every scattering point
        self._rcs = tuple(rcs)
        self._cpm = cpm

        super().__init__(positions, orientations, rcs, cpm)

    @property
    def object_type(self) -> str:
        """Type of sensing target

        :type: :py:class:`str`
        """
        return self._object_type

    @property
    def model_type(self) -> int:
        """RCS model of clause 7.9.2.1

        :type: :py:class:`int`
        """
        return self._model_type

    @property
    def random_sigma_s(self) -> bool:
        r"""Get/set whether the RCS component :math:`\sigma_S` of clause
        7.9.2.1 is drawn for every pair of directions

        :type: :py:class:`bool`
        """
        return self._random_sigma_s

    @random_sigma_s.setter
    def random_sigma_s(self, value: bool):
        for rcs in self._rcs:
            rcs.random_sigma_s = value
        self._random_sigma_s = value

    @property
    def random_phases(self) -> bool:
        """Get/set whether the four initial phases of eq. 7.9.2-5 are drawn
        for every pair of directions

        :type: :py:class:`bool`
        """
        return self._random_phases

    @random_phases.setter
    def random_phases(self, value: bool):
        self._cpm.random_phases = value
        self._random_phases = value

    @property
    def random_xpr(self) -> bool:
        """Get/set whether the XPR of eq. 7.9.2-5 is drawn for every pair of
        directions

        :type: :py:class:`bool`
        """
        return self._random_xpr

    @random_xpr.setter
    def random_xpr(self, value: bool):
        self._cpm.random_xpr = value
        self._random_xpr = value

    def _build_scattering_points(
        self,
        positions: mi.Point3f
    ) -> Tuple[mi.Point3f, mi.Point3f, List[TR38901RCS], TR38901CPM]:
        """Builds the scattering points of this model

        :return: Positions of the scattering points in the LCS [m]
        :return: Orientations of the scattering points in the LCS [rad]
        :return: RCS callable of every scattering point
        :return: CPM callable shared by every scattering point, as the
            specifications give one XPR per type of sensing target
        """
        if self._parameters.num_scattering_points == 1:
            orientations = [(0., 0., 0.)]
            # The single point carries every set of parameters of its table as
            # a lobe, and selects one of them from the bisector angle
            rcs = [self._make_rcs(self._parameters.lobes)]
        else:
            orientations = []
            rcs = []
            for lobe in self._parameters.lobes:
                # The azimuth of a point is carried by its orientation rather
                # than by its lobe, as a rotation about the z-axis leaves the
                # zenith angle of the bisector unchanged and therefore
                # reproduces the pattern of the table exactly. The azimuth
                # center of the lobe is zeroed so that it is not applied a
                # second time inside the callable.
                if lobe.azimuth_dependent:
                    orientations.append((dr.deg2rad(lobe.phi_center), 0., 0.))
                    lobe = replace(lobe, phi_center=0.)
                else:
                    orientations.append((0., 0., 0.))
                rcs.append(self._make_rcs([lobe]))

        cpm = TR38901CPM(self._parameters.xpr_mean_db,
                         xpr_std_db=self._parameters.xpr_std_db,
                         random_phases=self._random_phases,
                         random_xpr=self._random_xpr)

        alpha, beta, gamma = zip(*orientations)
        return (positions, mi.Point3f(list(alpha), list(beta), list(gamma)),
                rcs, cpm)

    def _make_rcs(self, lobes: Sequence[LobeParameters]) -> TR38901RCS:
        """Builds the RCS callable of a scattering point

        :param lobes: Sets of parameters of the scattering point
        :return: RCS callable
        """
        return TR38901RCS(lobes,
                          k1=self._parameters.k1,
                          k2=self._parameters.k2,
                          sigma_m_db=self._parameters.sigma_m_db,
                          sigma_s_std_db=self._parameters.sigma_s_db,
                          random_sigma_s=self._random_sigma_s)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(object_type='{self._object_type}', "
                f"model_type={self._model_type})")
