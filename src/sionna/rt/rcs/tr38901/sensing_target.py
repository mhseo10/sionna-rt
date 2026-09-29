#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Sensing targets following 3GPP TR 38.901, clause 7.9.2"""

from __future__ import annotations

from typing import Dict, Tuple

import mitsuba as mi

from sionna.rt.constants import DEFAULT_SENSING_TARGET_COLOR, \
                                DEFAULT_SENSING_TARGET_OPACITY
from sionna.rt.radio_devices import RadioDevice
from sionna.rt.scene_object import SceneObject
from sionna.rt.utils.meshes import clone_mesh, load_mesh
from ..sensing_target import SensingTarget
from .parameters import LobeParameters, MODEL_2, get_target_parameters
from .scattering_model import TR38901ScatteringModel


# Height of the scattering point of a human, as a fraction of its height.
# Table 7.9.6.1-2 gives a height of 1.5m for an adult pedestrian of 1.75m.
_HUMAN_SCATTERING_POINT_HEIGHT = 1.5/1.75


class TR38901SensingTarget(SensingTarget):
    # pylint: disable=line-too-long
    r"""
    Sensing target following 3GPP TR 38.901, clause 7.9.2

    The target automatically creates its
    :class:`~sionna.rt.rcs.TR38901ScatteringModel`; no scattering model is
    supplied by the user. If no mesh is provided, the target is represented
    by a cuboid whose omitted dimensions default independently to the values
    specified for ``object_type``. If a mesh is provided, its axis-aligned
    bounding box (AABB) determines the dimensions used to place the
    scattering points. The AABB is read before the target is posed, so the
    resulting dimensions are the ones of its local coordinate system (LCS)
    regardless of the requested ``orientation``. Scaling the target afterwards
    resizes its geometry and its scattering points alike, and the reported
    dimensions follow.

    The random components of the model, i.e., the RCS component
    :math:`\sigma_S` of clause 7.9.2.1 and the XPR and initial phases of
    clause 7.9.2.2, are enabled through ``random_sigma_s``, ``random_xpr``
    and ``random_phases``, which the target forwards to its
    :class:`~sionna.rt.rcs.TR38901ScatteringModel`. They can also be changed
    afterwards through the properties of the same names of its
    :attr:`~sionna.rt.rcs.SensingTarget.scattering_model`:

    .. code-block:: python

        target = TR38901SensingTarget(name="st",
                                      object_type="vehicle-multi-sp",
                                      random_sigma_s=True)
        target.scattering_model.random_sigma_s = False

    :param name: Name of the sensing target

    :param object_type: Type of sensing target, one of ``"uav-small-size"``,
        ``"uav-large-size"``, ``"human"``, ``"vehicle-single-sp"``,
        ``"vehicle-multi-sp"``, ``"agv-single-sp"``, ``"agv-multi-sp"``

    :param model_type: RCS model of clause 7.9.2.1, either 1 or 2

    :param mi_mesh: Mitsuba shape. Mutually exclusive with ``fname`` and the
        cuboid dimensions.

    :param fname: Filename of a valid mesh ( "*.ply" | "*.obj"). Mutually
        exclusive with ``mi_mesh`` and the cuboid dimensions.

    :param length: Size along the x-axis of the LCS [m]. If omitted for a
        cuboid, defaults to the specification value for ``object_type``.

    :param width: Size along the y-axis of the LCS [m]. If omitted for a
        cuboid, defaults to the specification value for ``object_type``.

    :param height: Size along the z-axis of the LCS [m]. If omitted for a
        cuboid, defaults to the specification value for ``object_type``.

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

    :param random_sigma_s: If set to `True`, the RCS component
        :math:`\sigma_S` of clause 7.9.2.1 is drawn for every pair of
        directions. If set to :py:class:`None`, the default of
        :class:`~sionna.rt.rcs.TR38901ScatteringModel` is used.

    :param random_phases: If set to `True`, the four initial phases of
        eq. 7.9.2-5 are drawn for every pair of directions. If set to
        :py:class:`None`, the default of
        :class:`~sionna.rt.rcs.TR38901ScatteringModel` is used.

    :param random_xpr: If set to `True`, the XPR of eq. 7.9.2-5 is drawn for
        every pair of directions. If set to :py:class:`None`, the default of
        :class:`~sionna.rt.rcs.TR38901ScatteringModel` is used.
    """

    def __init__(
        self,
        name: str,
        object_type: str,
        model_type: int = MODEL_2,
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
        velocity: mi.Vector3f | None = None,
        random_sigma_s: bool | None = None,
        random_phases: bool | None = None,
        random_xpr: bool | None = None
    ):
        parameters = get_target_parameters(object_type, model_type)
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
                       parameters.dimensions))
            geometry = dict(zip(("length", "width", "height"), dimensions))

        self._object_type = object_type
        self._model_type = model_type
        self._length, self._width, self._height = dimensions

        positions = self._build_scattering_point_positions(parameters.lobes,
                                                            parameters.num_scattering_points)
        # Only the flags which are given are forwarded, so that the others
        # keep the defaults of the scattering model
        random_flags = {name: value for name, value in (
                            ("random_sigma_s", random_sigma_s),
                            ("random_phases", random_phases),
                            ("random_xpr", random_xpr))
                        if value is not None}
        scattering_model = TR38901ScatteringModel(
            object_type, model_type, positions, **random_flags)

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
    ) -> TR38901SensingTarget | mi.Mesh:
        """Creates a clone of the current TR 38.901 sensing target

        The clone shares the scattering model of the original target, and has
        the same dimensions, geometry, pose, color, and display opacity.
        """
        cloned_mesh = clone_mesh(self.mi_mesh, name=name, props=props)

        if as_mesh:
            cloned_mesh.set_bsdf(self.radio_material)
            return cloned_mesh

        clone = TR38901SensingTarget(
            name=cloned_mesh.id(),
            object_type=self.object_type,
            model_type=self.model_type,
            mi_mesh=cloned_mesh,
            color=self.radio_material.color,
            display_opacity=self.display_opacity)

        # The clone is built from the mesh of this target, whose AABB only
        # gives the dimensions in the local coordinate system as long as the
        # target is neither rotated nor scaled. The dimensions, and the
        # scattering model they determine, are therefore carried over instead
        # of being re-derived from the cloned mesh.
        # pylint: disable=protected-access
        clone._length = self._length
        clone._width = self._width
        clone._height = self._height
        clone._scattering_model = self._scattering_model
        self._carry_over_pose(clone)
        return clone

    def _build_scattering_point_positions(
        self,
        lobes: Tuple[LobeParameters, ...],
        num_scattering_points: int
    ) -> mi.Point3f:
        """Builds scattering-point positions in the target LCS [m]"""
        if num_scattering_points == 1:
            height = 0.5*self._height
            if self._object_type == "human":
                height = _HUMAN_SCATTERING_POINT_HEIGHT*self._height
            return mi.Point3f(0., 0., height - 0.5*self._height)

        positions = {
            "front": (0.5*self._length, 0., 0.),
            "back": (-0.5*self._length, 0., 0.),
            "left": (0., 0.5*self._width, 0.),
            "right": (0., -0.5*self._width, 0.),
            "roof": (0., 0., 0.5*self._height),
        }
        try:
            selected = [positions[lobe.name] for lobe in lobes]
        except KeyError as err:
            raise ValueError(
                f"No position defined for scattering point '{err.args[0]}' "
                f"of '{self._object_type}'") from err

        x, y, z = zip(*selected)
        return mi.Point3f(list(x), list(y), list(z))

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name='{self.name}', "
                f"object_type='{self._object_type}', "
                f"model_type={self._model_type}, length={self.length}, "
                f"width={self.width}, height={self.height})")


def _validate_dimension(name: str, value: float) -> float:
    """Validates one user-provided target dimension"""
    value = float(value)
    if value <= 0.:
        raise ValueError(f"`{name}` must be strictly positive")
    return value
