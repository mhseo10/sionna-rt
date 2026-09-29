#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Classes and functions related to Sensing Target"""

from __future__ import annotations

from typing import Tuple
import mitsuba as mi

from sionna.rt.constants import DEFAULT_SENSING_TARGET_COLOR, \
                                DEFAULT_SENSING_TARGET_OPACITY
from sionna.rt.radio_materials import AbsorberRadioMaterial, RadioMaterialBase
from sionna.rt.scene_object import SceneObject
from sionna.rt.radio_devices import RadioDevice
from sionna.rt.utils.meshes import clone_mesh
from .scattering_model import ScatteringModel
from .scattering_model_viewer import ScatteringModelViewer


class SensingTarget(SceneObject):
    # pylint: disable=line-too-long
    r"""
    Class defining a sensing target

    A sensing target is a scene object whose scattering response is described
    by a scattering model, i.e., by a collection of scattering points in its
    local coordinate system (LCS), rather than by a radio material. Its radio
    material is therefore always an :class:`~sionna.rt.AbsorberRadioMaterial`,
    which cannot be changed: no energy is scattered by the surface of the
    target itself.

    The shape of a sensing target can be specified in three mutually exclusive
    ways: from a Mitsuba shape (``mi_mesh``), from a mesh file (``fname``), or
    as a cuboid of the given dimensions (``length``, ``width``, and
    ``height``).

    To create a sensing target shaped as a cuboid of 4m x 2m x 1.5m, with a
    single scattering point at the center of its roof, i.e., 0.75m above its
    center:

    .. code-block:: python

        model = ScatteringModel([0, 0, 0.75], rcs="my_rcs",
                                cpm="my_cpm")
        target = SensingTarget(name="car", scattering_model=model,
                               length=4., width=2., height=1.5)

    As with any other scene object, a sensing target is added to a scene using
    :meth:`~sionna.rt.Scene.add` or :meth:`~sionna.rt.Scene.edit`. Its position
    and orientation can be set at construction, or through the corresponding
    properties:

    .. code-block:: python

        target.position = [10, 0, 0]
        scene.add(target)
        target.orientation = [dr.pi/2, 0, 0]

    The scattering points of a target are given in its unscaled LCS, and are
    scaled along with its geometry by its
    :attr:`~sionna.rt.SceneObject.scaling`, so that a point keeps its place
    relative to the shape it describes.

    Both :class:`~sionna.rt.rcs.RCSSolver` and
    :class:`~sionna.rt.PathSolver` trace paths through the shapes of the
    sensing targets and see them as absorbers, so that a target casts a radio
    shadow. As the scattering response of a target is entirely described by its
    scattering model, :class:`~sionna.rt.rcs.RCSSolver` makes an exception
    for the scattering points a target contains, i.e., the ones which lie
    within its bounding box: a leg which starts from such a point is only
    tested for occlusion once it has left that bounding box. A scattering
    point placed outside of that box, e.g., above the roof of a vehicle, is on
    the other hand occluded by its own target as it is by any other object of
    the scene.

    The :attr:`~sionna.rt.SceneObject.velocity` of a target is accounted for
    by :class:`~sionna.rt.rcs.RCSSolver` when computing the Doppler shifts
    of the paths scattered by its scattering points. It must be set
    explicitly, as moving a target by updating its
    :attr:`~sionna.rt.SceneObject.position` does not set it:

    .. code-block:: python

        target.velocity = [10, 0, 0]

    :param name: Name of the sensing target

    :param scattering_model: Scattering model of the target

    :param mi_mesh: Mitsuba shape.
        Mutually exclusive with ``fname`` and the cuboid dimensions.

    :param fname: Filename of a valid mesh ( "*.ply" | "*.obj").
        Mutually exclusive with ``mi_mesh`` and the cuboid dimensions.

    :param length: Length, i.e., size along the x-axis of the LCS, of the
        cuboid representing the target [m]. Must be specified together with
        ``width`` and ``height``, and is mutually exclusive with ``mi_mesh``
        and ``fname``.

    :param width: Width, i.e., size along the y-axis of the LCS, of the cuboid
        representing the target [m]

    :param height: Height, i.e., size along the z-axis of the LCS, of the
        cuboid representing the target [m]

    :param color: Defines the RGB (red, green, blue) ``color`` parameter for
        the target as displayed in the previewer and renderer. Each RGB
        component must have a value within the range :math:`\in [0,1]`.

    :param display_opacity: Defines the opacity with which the target is
        displayed in the previewer and renderer, within the range
        :math:`\in [0,1]`. Targets are displayed as semi-transparent by
        default, so that their scattering points remain visible.

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

    def __init__(self,
                 name: str,
                 scattering_model: ScatteringModel,
                 mi_mesh: mi.Mesh | None = None,
                 fname: str | None = None,
                 length: float | None = None,
                 width: float | None = None,
                 height: float | None = None,
                 color: Tuple[float, float, float]
                        = DEFAULT_SENSING_TARGET_COLOR,
                 display_opacity: float
                        = DEFAULT_SENSING_TARGET_OPACITY,
                 position: mi.Point3f | None = None,
                 orientation: mi.Point3f | None = None,
                 look_at: (mi.Point3f | SceneObject | RadioDevice |
                           None) = None,
                 velocity: mi.Vector3f | None = None):

        if not isinstance(scattering_model, ScatteringModel):
            raise ValueError("`scattering_model` must be a ScatteringModel "
                             "instance")

        dimensions = (length, width, height)
        cuboid = any(d is not None for d in dimensions)
        num_shapes = (mi_mesh is not None) + (fname is not None) + cuboid
        if num_shapes != 1:
            raise ValueError("Exactly one of a Mitsuba shape (`mi_mesh`), a"
                             " filename (`fname`), or the dimensions of a"
                             " cuboid (`length`, `width`, and `height`) must be"
                             " provided")

        if cuboid:
            if any(d is None for d in dimensions):
                raise ValueError("`length`, `width`, and `height` must all be"
                                 " provided to build a cuboid")
            if any(d <= 0. for d in dimensions):
                raise ValueError("`length`, `width`, and `height` must be"
                                 " positive")
            mi_mesh = self._cuboid(*dimensions)

        # The surface of a sensing target scatters no energy, as its scattering
        # response is entirely described by its scattering model
        radio_material = AbsorberRadioMaterial(name=f"{name}-material",
                                               color=color)

        super().__init__(mi_mesh=mi_mesh,
                         name=name,
                         fname=fname,
                         radio_material=radio_material,
                         remove_duplicate_vertices=cuboid,
                         position=position,
                         orientation=orientation,
                         look_at=look_at,
                         velocity=velocity)

        self._scattering_model = scattering_model
        self.display_opacity = display_opacity

    @property
    def display_opacity(self):
        r"""
        :py:class:`float`: Get/set the opacity with which this target is
        displayed in the previewer and renderer. A value of 1 makes the target
        fully opaque, and a value of 0 makes it invisible.
        """
        return self._display_opacity

    @display_opacity.setter
    def display_opacity(self, opacity):
        if (opacity < 0.) or (opacity > 1.):
            raise ValueError("The display opacity must be in the range [0, 1]")
        self._display_opacity = float(opacity)

    @property
    def scattering_model(self):
        """
        Scattering model of the sensing target

        :type: :class:`~sionna.rt.rcs.ScatteringModel`
        """
        return self._scattering_model

    @property
    def radio_material(self):
        r"""(read-only) Radio material of the sensing target

        The surface of a sensing target absorbs all the incident energy, as its
        scattering response is entirely described by its
        :attr:`~sionna.rt.rcs.SensingTarget.scattering_model`. This material
        can therefore not be changed.

        :type: :class:`~sionna.rt.AbsorberRadioMaterial`
        """
        return SceneObject.radio_material.fget(self)

    @radio_material.setter
    def radio_material(self, mat: str | RadioMaterialBase):
        raise ValueError("The radio material of a sensing target absorbs all"
                         " the incident energy and cannot be changed")

    def show(self,
             k_i: mi.Vector3f,
             **kwargs) -> None:
        # pylint: disable=line-too-long
        r"""
        In an interactive notebook environment, opens an interactive 3D viewer
        of the scattering model of this target

        The target is shown in its local coordinate system (LCS), i.e., as it
        is before being positioned and oriented, since that is the frame in
        which its scattering points and their RCS and CPM are defined. The
        viewer displays the mesh of the target, the bounding box of that mesh
        as a wireframe, every scattering point with the triad of its local axes,
        the incident direction, and, around every scattering point, a surface
        showing the bistatic radar cross-section it scatters in every
        direction [dBsm].

        Controls:

        * Mouse left: Rotate
        * Scroll wheel: Zoom
        * Mouse right: Move

        :param k_i: Incident direction of propagation, **in the LCS of this
            target**. As in
            :meth:`~sionna.rt.rcs.ScatteringPoints.eval_rcs`, this is a
            direction of propagation and therefore points *towards* the
            scattering points. A direction :math:`\mathbf{k}` given in the
            global coordinate system is brought to the LCS with
            ``rotation_matrix(target.orientation).T @ k``.

        :param kwargs: Additional display options, as accepted by
            :class:`~sionna.rt.rcs.ScatteringModelViewer`

        Example
        -------
        .. code-block:: python

            import drjit as dr
            from sionna.rt.rcs import TR38901SensingTarget
            from sionna.rt.utils import r_hat

            target = TR38901SensingTarget(name="st",
                                          object_type="vehicle-multi-sp")

            # Wave incident from the front of the car, i.e., from the +x-axis
            # side of its LCS, at a zenith and an azimuth angle of 45 deg
            target.show(k_i=-r_hat(dr.pi/4, dr.pi/4))
        """
        ScatteringModelViewer(self, k_i, **kwargs).display()

    def clone(self, name: str | None = None, as_mesh: bool = False,
              props: mi.Properties | None = None) -> SensingTarget | mi.Mesh:
        r"""
        Creates a clone of the current sensing target

        The clone shares the scattering model of the original target and has
        the same geometry, color, and display opacity, but is assigned a new
        name.

        :param name: Name (id) of the cloned target.
            If :py:class:`None`, the clone will be named as
            ``<original_name>-clone``.

        :param as_mesh: If set to `True`, the clone will be returned as a
            :py:class:`mitsuba.Mesh` object. Otherwise, a
            :py:class:`sionna.rt.rcs.SensingTarget` will be returned.

        :param props: Pre-populated properties to be used in the new Mitsuba
            shape. Allows overriding the BSDF, emitter, etc.

        :return: A clone of the current sensing target
        """
        cloned_mesh = clone_mesh(self.mi_mesh, name=name, props=props)

        if as_mesh:
            cloned_mesh.set_bsdf(self.radio_material)
            return cloned_mesh

        clone = SensingTarget(name=cloned_mesh.id(),
                              scattering_model=self.scattering_model,
                              mi_mesh=cloned_mesh,
                              color=self.radio_material.color,
                              display_opacity=self.display_opacity)
        self._carry_over_pose(clone)
        return clone

    ##################################################
    # Internal methods
    ##################################################

    @staticmethod
    def _cuboid(length: float, width: float, height: float) -> mi.Mesh:
        r"""
        Builds a cuboid mesh centered at the origin and aligned with the axes

        :param length: Size of the cuboid along the x-axis [m]
        :param width: Size of the cuboid along the y-axis [m]
        :param height: Size of the cuboid along the z-axis [m]

        :return: Cuboid mesh
        """
        # The Mitsuba cube spans [-1,1] along every axis
        scaling = [0.5*length, 0.5*width, 0.5*height]
        return mi.load_dict({
            "type": "cube",
            "face_normals": True,
            "to_world": mi.ScalarTransform4f().scale(scaling),
        })
