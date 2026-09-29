#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Sensing target with a single scattering point of constant RCS"""

from __future__ import annotations

from typing import Dict, Tuple

import mitsuba as mi

from sionna.rt.constants import DEFAULT_SENSING_TARGET_COLOR, \
                                DEFAULT_SENSING_TARGET_OPACITY
from sionna.rt.radio_devices import RadioDevice
from sionna.rt.scene_object import SceneObject
from sionna.rt.utils.meshes import clone_mesh, load_mesh
from .cpm import ConstantCPM
from .rcs import ConstantRCS
from .scattering_model import ScatteringModel
from .sensing_target import SensingTarget


#: Dimensions (length, width, height) [m] of the cuboid representing a
#: constant-RCS sensing target, unless specified otherwise
DEFAULT_DIMENSIONS = (0.3, 0.3, 0.3)


class ConstantRCSSensingTarget(SensingTarget):
    # pylint: disable=line-too-long
    r"""
    Sensing target with a single scattering point of constant radar
    cross-section

    The target automatically creates its
    :class:`~sionna.rt.rcs.ScatteringModel`; no scattering model is supplied
    by the user. The model consists of a single scattering point located at
    the center :math:`(0,0,0)` of the local coordinate system (LCS) of the
    target, whose RCS is a :class:`~sionna.rt.rcs.ConstantRCS` and whose
    cross-polarization matrix is a :class:`~sionna.rt.rcs.ConstantCPM`, so
    that neither depends on the incident and scattered directions.

    If no mesh is provided, the target is represented by a cuboid whose
    omitted dimensions default to a cube of 0.3m. If a mesh is provided,
    its axis-aligned bounding box (AABB) determines the reported dimensions.
    The AABB is read before the target is posed, so the resulting dimensions
    are the ones of its LCS regardless of the requested ``orientation``, and
    scaling the target afterwards scales them accordingly. As the scattering
    point sits at the center of the LCS, the dimensions only affect how the
    target is displayed and the geometry seen by
    :class:`~sionna.rt.PathSolver`, not its scattering response.

    To create a target of cross-section 0.1 :math:`\text{m}^2` shaped as the default cube:

    .. code-block:: python

        from sionna.rt.rcs import ConstantRCSSensingTarget

        target = ConstantRCSSensingTarget(name="st", sigma=0.1)

    The cross-section and cross-polarization ratio can also be changed after
    construction:

    .. code-block:: python

        target.sigma = 20.
        target.xpr_db = 3.

    :param name: Name of the sensing target

    :param sigma: Bistatic radar cross-section :math:`\sigma` [:math:`\text{m}^2`] of the
        scattering point

    :param xpr_db: Cross-polarization ratio :math:`\text{XPR}_{dB}` [dB] of
        the scattering point, as defined in
        :class:`~sionna.rt.rcs.ConstantCPM`. Set to :py:class:`None` for a
        target which does not depolarize.

    :param mi_mesh: Mitsuba shape. Mutually exclusive with ``fname`` and the
        cuboid dimensions.

    :param fname: Filename of a valid mesh ( "*.ply" | "*.obj"). Mutually
        exclusive with ``mi_mesh`` and the cuboid dimensions.

    :param length: Size along the x-axis of the LCS [m]. If omitted for a
        cuboid, defaults to 0.3m.

    :param width: Size along the y-axis of the LCS [m]. If omitted for a
        cuboid, defaults to 0.3m.

    :param height: Size along the z-axis of the LCS [m]. If omitted for a
        cuboid, defaults to 0.3m.

    :param color: RGB color used by the previewer and renderer

    :param display_opacity: Display opacity in the range :math:`[0,1]`

    :param position: Position :math:`(x,y,z)` [m] of the center of the target.
        If set to :py:class:`None`, the target keeps the position of its mesh.

    :param orientation: Orientation specified through three angles
        :math:`(\alpha, \beta, \gamma)`
        corresponding to a 3D rotation as defined in :eq:`rotation`.
        Mutually exclusive with ``look_at``; specifying both raises
        :py:class:`ValueError`. Defaults to :math:`(0,0,0)` if both
        ``orientation`` and ``look_at`` are :py:class:`None`.

    :param look_at: A position, or the instance of a
        :class:`~sionna.rt.SceneObject` or :class:`~sionna.rt.RadioDevice` to
        look at. Mutually exclusive with ``orientation``.

    :param velocity: Velocity vector of the target [m/s]
    """

    def __init__(
        self,
        name: str,
        sigma: float | mi.Float = 1.,
        xpr_db: float | mi.Float | None = None,
        mi_mesh: mi.Mesh | None = None,
        fname: str | None = None,
        length: float | None = None,
        width: float | None = None,
        height: float | None = None,
        color: Tuple[float, float, float] = DEFAULT_SENSING_TARGET_COLOR,
        display_opacity: float = DEFAULT_SENSING_TARGET_OPACITY,
        position: mi.Point3f | None = None,
        orientation: mi.Point3f | None = None,
        look_at: mi.Point3f | SceneObject | RadioDevice | None = None,
        velocity: mi.Vector3f | None = None
    ):
        given_dimensions = (length, width, height)
        has_dimensions = any(value is not None for value in given_dimensions)

        if (mi_mesh is not None) and (fname is not None):
            raise ValueError("Only one of `mi_mesh` and `fname` can be "
                             "provided")
        if ((mi_mesh is not None) or (fname is not None)) and has_dimensions:
            raise ValueError("A mesh (`mi_mesh` or `fname`) and cuboid "
                             "dimensions are mutually exclusive")

        if fname is not None:
            mi_mesh = load_mesh(fname)

        if mi_mesh is not None:
            if not isinstance(mi_mesh, mi.Mesh):
                raise ValueError("`mi_mesh` must be a Mitsuba Mesh object")
            bbox = mi_mesh.bbox()
            extents = bbox.max - bbox.min
            dimensions = tuple(float(value)
                               for value in (extents.x, extents.y, extents.z))
            if any(value <= 0. for value in dimensions):
                raise ValueError("The mesh AABB must have strictly positive "
                                 "dimensions")
            geometry = {"mi_mesh": mi_mesh}
        else:
            dimensions = tuple(
                default if given is None
                else _validate_dimension(dimension_name, given)
                for dimension_name, given, default
                in zip(("length", "width", "height"), given_dimensions,
                       DEFAULT_DIMENSIONS))
            geometry = dict(zip(("length", "width", "height"), dimensions))

        self._length, self._width, self._height = dimensions

        self._rcs = ConstantRCS(sigma=sigma)
        self._cpm = ConstantCPM(xpr_db=xpr_db)
        scattering_model = ScatteringModel(mi.Point3f(0., 0., 0.),
                                           rcs=self._rcs,
                                           cpm=self._cpm)

        # The dimensions above are read from the mesh before the target is
        # posed, and therefore are the ones of its local coordinate system
        super().__init__(name=name,
                         scattering_model=scattering_model,
                         color=color,
                         display_opacity=display_opacity,
                         position=position,
                         orientation=orientation,
                         look_at=look_at,
                         velocity=velocity,
                         **geometry)

    @property
    def rcs(self) -> ConstantRCS:
        r"""RCS callable of the scattering point of this target

        :type: :class:`~sionna.rt.rcs.ConstantRCS`
        """
        return self._rcs

    @property
    def cpm(self) -> ConstantCPM:
        r"""CPM callable of the scattering point of this target

        :type: :class:`~sionna.rt.rcs.ConstantCPM`
        """
        return self._cpm

    @property
    def sigma(self):
        r"""Get/set the bistatic radar cross-section :math:`\sigma` [:math:`\text{m}^2`] of
        the scattering point

        :type: :py:class:`mi.Float`
        """
        return self._rcs.sigma

    @sigma.setter
    def sigma(self, sigma):
        self._rcs.sigma = sigma

    @property
    def xpr_db(self):
        r"""Get/set the cross-polarization ratio
        :math:`\text{XPR}_{dB}` [dB] of the scattering point, or
        :py:class:`None` if it does not depolarize

        :type: :py:class:`mi.Float` | :py:class:`None`
        """
        return self._cpm.xpr_db

    @xpr_db.setter
    def xpr_db(self, xpr_db):
        self._cpm.xpr_db = xpr_db

    @property
    def length(self) -> float:
        """Size along the x-axis of the local coordinate system [m], scaled by
        the :attr:`~sionna.rt.SceneObject.scaling` of this target"""
        return self._length*self.scaling.x[0]

    @property
    def width(self) -> float:
        """Size along the y-axis of the local coordinate system [m], scaled by
        the :attr:`~sionna.rt.SceneObject.scaling` of this target"""
        return self._width*self.scaling.y[0]

    @property
    def height(self) -> float:
        """Size along the z-axis of the local coordinate system [m], scaled by
        the :attr:`~sionna.rt.SceneObject.scaling` of this target"""
        return self._height*self.scaling.z[0]

    @property
    def dimensions(self) -> Dict[str, float]:
        """Target dimensions as ``length``, ``width``, and ``height`` [m]"""
        return {"length": self.length, "width": self.width,
                "height": self.height}

    def clone(
        self,
        name: str | None = None,
        as_mesh: bool = False,
        props: mi.Properties | None = None
    ) -> ConstantRCSSensingTarget | mi.Mesh:
        """Creates a clone of the current constant-RCS sensing target

        The clone shares the scattering model of the original target, and has
        the same dimensions, geometry, pose, color, and display opacity.
        """
        cloned_mesh = clone_mesh(self.mi_mesh, name=name, props=props)

        if as_mesh:
            cloned_mesh.set_bsdf(self.radio_material)
            return cloned_mesh

        clone = ConstantRCSSensingTarget(
            name=cloned_mesh.id(),
            sigma=self.sigma,
            xpr_db=self.xpr_db,
            mi_mesh=cloned_mesh,
            color=self.radio_material.color,
            display_opacity=self.display_opacity)

        # The clone is built from the mesh of this target, whose AABB only
        # gives the dimensions in the local coordinate system as long as the
        # target is neither rotated nor scaled. The dimensions are therefore
        # carried over instead of being re-derived from the cloned mesh.
        # pylint: disable=protected-access
        clone._length = self._length
        clone._width = self._width
        clone._height = self._height
        clone._rcs = self._rcs
        clone._cpm = self._cpm
        clone._scattering_model = self._scattering_model
        self._carry_over_pose(clone)
        return clone

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name='{self.name}', "
                f"sigma={self.sigma}, "
                f"xpr_db={self.xpr_db}, "
                f"length={self.length}, width={self.width}, "
                f"height={self.height})")


def _validate_dimension(name: str, value: float) -> float:
    """Validates one user-provided target dimension"""
    value = float(value)
    if value <= 0.:
        raise ValueError(f"`{name}` must be strictly positive")
    return value
