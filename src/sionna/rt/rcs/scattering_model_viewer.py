#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Interactive viewer of the scattering model of a sensing target"""

from __future__ import annotations

from typing import Callable, Tuple

import drjit as dr
import matplotlib as mpl
import mitsuba as mi
import numpy as np
from ipywidgets import widgets
import pythreejs as p3s

from sionna.rt.utils import r_hat, rotation_matrix
from sionna.rt.viewer import Viewer, rgb_to_html

#: Color of the scattering points
SCATTERING_POINT_COLOR = (0.0, 0.0, 0.0)

#: Colors of the local x-, y-, and z-axis of a scattering point
AXES_COLORS = ((0.9, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.3, 0.9))

#: Color of the incident direction
INCIDENT_COLOR = (0.1, 0.1, 0.1)

#: Color of the wireframe box
BOX_COLOR = (0.35, 0.35, 0.35)

#: Cross-section [m^2] below which a direction is considered to scatter no
#: energy at all. 1e-30 m^2 is -300 dBsm, far below any usable dynamic range.
_MIN_SIGMA = 1e-30

#: The same cross-section [dBsm]
_MIN_SIGMA_DB = 10.*np.log10(_MIN_SIGMA)

#: Smallest range [dB] a surface is allowed to span. A point which scatters
#: the same cross-section in every direction has no range at all, which would
#: otherwise leave its surface with a radius of zero everywhere.
_MIN_RANGE = 1.

#: Zenith and azimuth angles [rad] of the candidate camera positions, in order
#: of preference. The first one is the default view of
#: :class:`~sionna.rt.viewer.Viewer`.
_CAMERA_ANGLES = tuple((theta, phi)
                       for theta in (np.pi/4, np.pi/3)
                       for phi in (np.pi/4, 3*np.pi/4, 7*np.pi/4, 5*np.pi/4))

#: Smallest cosine of the angle between the camera position and the direction
#: the wave comes from, which keeps the camera on the illuminated side of the
#: target
_MIN_COS_ILLUMINATED = 0.25


class ScatteringModelViewer(Viewer):
    # pylint: disable=line-too-long
    r"""
    Interactive viewer of the scattering model of a sensing target,
    illuminated from a given incident direction

    The viewer shows, in the local coordinate system (LCS) of the target, its
    mesh, the bounding box of that mesh as a wireframe, and every scattering
    point of its
    :attr:`~sionna.rt.rcs.SensingTarget.scattering_model` together with the
    triad of its local axes. Around every scattering point, a surface shows
    the bistatic radar cross-section
    :math:`\sigma(\mathbf{k}_i, \mathbf{k}_s)` [dBsm] returned by the RCS
    callable of the point, as a function of the scattered direction
    :math:`\mathbf{k}_s`. The cross-polarization matrix of the point does not
    affect the surface, as :func:`~sionna.rt.rcs.register_cpm` requires it to
    leave the scattered power unchanged.

    The color of a surface always encodes the cross-section in dBsm. Its
    radius encodes the same quantity in dB by default, which keeps deep nulls
    readable, or the cross-section itself when ``radial_scale`` is set to
    ``"linear"``. Both cover the range of the cross-sections that are shown
    rather than a fixed window, so that the shape of a surface comes out
    whatever the directivity of its scattering point.

    :meth:`~sionna.rt.rcs.SensingTarget.show` creates this viewer and
    displays it. The viewer can also be created directly, e.g., to display it
    later or to access its
    :attr:`~sionna.rt.rcs.ScatteringModelViewer.cross_section_range`:

    .. code-block:: python

        viewer = ScatteringModelViewer(target, k_i=(0, 0, -1))
        viewer.display()

    :param target: Sensing target whose scattering model is shown

    :param k_i: Incident direction of propagation, in the LCS of the target.
        It points *towards* the scattering points, following the convention
        of :meth:`~sionna.rt.rcs.ScatteringPoints.eval_rcs`.

    :param num_zenith: Number of zenith angles at which the cross-section is
        evaluated

    :param num_azimuth: Number of azimuth angles at which the cross-section is
        evaluated

    :param dynamic_range: Largest range [dB] below the strongest direction
        that the colors, and the radii when ``radial_scale`` is ``"db"``,
        cover. Directions scattering less than that are drawn as if they
        scattered exactly that, which keeps a deep null from flattening the
        rest of a surface.

    :param pattern_scale: Size of the surfaces, relative to the largest size
        at which they stay within the target and do not reach into one
        another. A value of 1 therefore makes every surface as large as it can
        be without overlapping its neighbours.

    :param normalization: ``"global"`` to normalize every surface by the
        strongest direction over all scattering points, so that their sizes
        and colors are comparable, which collapses a point scattering as
        little as the weakest direction shown onto its marker, or
        ``"per_point"`` to normalize every surface by its own strongest
        direction, which brings out the shape of the weaker points.

    :param radial_scale: ``"db"`` to make the radius of a surface proportional
        to the cross-section in dB, or ``"linear"`` to make it proportional to
        the cross-section itself

    :param cmap: Colormap used for the surfaces, given as a name or as a
        Matplotlib colormap

    :param show_mesh: Show the mesh of the target

    :param show_box: Show the bounding box of the mesh as a wireframe

    :param show_orientations: Show the local axes of every scattering point

    :param show_lobes: Show the cross-section surfaces

    :param resolution: Size of the viewer figure

    :param fov: Field of view [deg]

    :param background: Background color in hex format prefixed by "#"
    """

    def __init__(self,
                 target,
                 k_i: mi.Vector3f,
                 *,
                 num_zenith: int = 65,
                 num_azimuth: int = 128,
                 dynamic_range: float = 40.,
                 pattern_scale: float = 1.,
                 normalization: str = "global",
                 radial_scale: str = "db",
                 cmap: str | Callable = "turbo",
                 show_mesh: bool = True,
                 show_box: bool = True,
                 show_orientations: bool = True,
                 show_lobes: bool = True,
                 resolution: Tuple[int, int] = (655, 500),
                 fov: float = 45.,
                 background: str = "white"):

        if num_zenith < 2:
            raise ValueError("`num_zenith` must be at least 2")
        if num_azimuth < 3:
            raise ValueError("`num_azimuth` must be at least 3")
        if dynamic_range <= 0.:
            raise ValueError("`dynamic_range` must be positive")
        if pattern_scale <= 0.:
            raise ValueError("`pattern_scale` must be positive")
        if normalization not in ("global", "per_point"):
            raise ValueError('`normalization` must be either "global" or '
                             '"per_point"')
        if radial_scale not in ("db", "linear"):
            raise ValueError('`radial_scale` must be either "db" or "linear"')

        k_i = mi.Vector3f(k_i)
        if dr.width(k_i) != 1:
            raise ValueError("`k_i` must be a single direction")
        if dr.all(dr.norm(k_i) == 0.):
            raise ValueError("`k_i` must not be the zero vector")
        k_i = dr.normalize(k_i)

        super().__init__(resolution=resolution, fov=fov,
                         background=background)

        self._num_zenith = num_zenith
        self._num_azimuth = num_azimuth
        self._dynamic_range = float(dynamic_range)
        self._colormap = mpl.colormaps.get_cmap(cmap) \
            if isinstance(cmap, str) else cmap
        self._colorbar: widgets.HTML | None = None
        self._cross_section_range: Tuple[float, float] | None = None

        # The scattering model, the scattering points, and their callables
        # are all defined in the LCS of the target, so the mesh is brought
        # back to that frame rather than the model being posed. This undoes
        # the transformation that `SceneObject` applied to the vertices.
        vertices, faces = _unposed_geometry(target)
        box_min = np.min(vertices, axis=0)
        box_max = np.max(vertices, axis=0)

        if show_mesh and (target.display_opacity > 0.):
            self._plot_mesh(vertices, faces, persist=False,
                            colors=np.array(target.radio_material.color),
                            opacity=target.display_opacity)

        if show_box:
            starts, ends = _box_edges(box_min, box_max)
            self._plot_lines(starts, ends,
                             np.tile(np.array(BOX_COLOR), (starts.shape[0], 1)),
                             width=1.)

        spst = target.scattering_model.spst
        num_points = dr.width(spst.lcs_positions)
        if num_points == 0:
            # A scattering model can be empty, in which case only the geometry
            # of the target is shown
            self._finalize_view()
            return

        # The geometry above keeps the scaling of the target, which the
        # scattering points are therefore scaled by as well, exactly as the
        # solver does when it brings them to the GCS
        positions = np.transpose((target.scaling*spst.lcs_positions).numpy())

        # Size which every annotation is scaled by, and radius of the
        # strongest direction of a surface
        extent = _lobe_extent(box_min, box_max, positions)
        lobe_radius = pattern_scale*extent

        # The surfaces are opaque and surround the points they belong to, so
        # the markers are drawn over them to stay visible
        self._plot_points(positions, persist=False,
                          colors=np.array(SCATTERING_POINT_COLOR),
                          radius=0.1*extent, on_top=True)
        self._add_legend_items([
            ("Scattering point",
             self._circular_legend_item(SCATTERING_POINT_COLOR, diameter=8))])

        if show_orientations:
            self._plot_orientations(spst, positions,
                                    0.5*lobe_radius + 0.25*extent)

        # Radius at which the incident arrow stops. Its head lands on the
        # surface, in the direction the wave comes from, which is where the
        # monostatic cross-section of the point is read off.
        head_radius = np.zeros((num_points,))

        if show_lobes:
            vmin, vmax = self._plot_lobes(spst, k_i, positions, lobe_radius,
                                          normalization, radial_scale)
            head_radius = self._backscatter_radius(spst, k_i, lobe_radius,
                                                   radial_scale, vmin, vmax)
            self._cross_section_range = (float(np.min(vmin)),
                                         float(np.max(vmax)))
            # Every surface has its own scale under a per-point
            # normalization, so only a relative colorbar is meaningful
            self._add_colorbar(float(np.min(vmin)), float(np.max(vmax)),
                               relative=normalization == "per_point")

        self._plot_incident(k_i, positions, head_radius,
                            0.5*(lobe_radius + extent))

        self._finalize_view(k_i)

    ##################################################
    # Accessors
    ##################################################

    @property
    def cross_section_range(self) -> Tuple[float, float] | None:
        r"""
        Lowest and highest bistatic radar cross-section [dBsm] covered by the
        displayed surfaces, i.e. the range spanned by the colormap

        The highest value is the strongest direction of the strongest
        scattering point. The lowest one is the weakest direction that is
        shown, which lies at most ``dynamic_range`` below the strongest
        direction of a surface.

        Set to :py:class:`None` when the surfaces are not displayed.

        :type: (:py:class:`float`, :py:class:`float`) | :py:class:`None`
        """
        return self._cross_section_range

    ##################################################
    # Internal methods
    ##################################################

    def _legend_widgets(self):
        r"""
        Returns the widgets making the legend column, i.e. the legend entries
        followed by the colorbar of the cross-section surfaces
        """
        legend = super()._legend_widgets()
        if self._colorbar is not None:
            legend = legend + [self._colorbar]
        return legend

    def _finalize_view(self, k_i: mi.Vector3f | None = None):
        r"""
        Points the camera at the displayed objects and pushes its far plane
        beyond them

        :param k_i: Incident direction of propagation in the LCS, which the
            camera is kept away from. If set to :py:class:`None`, the default
            view of :class:`~sionna.rt.viewer.Viewer` is used.
        """
        if k_i is None:
            self.center_view()
        else:
            self.center_view(*_camera_angles(k_i))

        far = 10000.
        if self._bbox.valid():
            bbox = mi.ScalarBoundingBox3f(self._bbox.min, self._bbox.max)
            bbox.expand(self._camera.position)
            far = dr.clip(dr.norm(bbox.extents()), 10000., 100000.)
        self._camera.far = float(far)

    def _plot_orientations(self,
                           spst,
                           positions: np.ndarray,
                           length: float):
        r"""
        Plots the triad of local axes of every scattering point

        The RCS of a scattering point is evaluated in its own frame, which the
        full triad shows, rather than only the direction the point faces.

        The arrows are shorter than the surface they start in, so they are
        drawn over it to remain visible.

        :param spst: Scattering points of the model
        :param positions: Positions of the scattering points in the LCS [m]
        :param length: Length of the arrows [m]
        """
        # With the sensing target left unposed, the frame of a scattering
        # point is set by its LCS orientation alone
        to_local = rotation_matrix(spst.lcs_orientations)

        labels = ("Local x-axis", "Local y-axis", "Local z-axis")
        units = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        for unit, color, label in zip(units, AXES_COLORS, labels):
            axis = np.transpose((to_local@mi.Vector3f(*unit)).numpy())
            self._plot_arrows(positions, positions + length*axis,
                              np.array(color), 0.25*length, on_top=True)
            self._add_legend_items([(label,
                                     self._arrow_legend_item(color))])

    def _plot_incident(self,
                       k_i: mi.Vector3f,
                       positions: np.ndarray,
                       head_radius: np.ndarray,
                       length: float):
        r"""
        Plots the incident direction as an arrow reaching every scattering
        point

        ``k_i`` is a direction of propagation, so the wave travels along it
        and the arrow therefore points towards the scattering point.

        The arrows stop on the surface of the point they reach, and are
        measured back from there rather than from the point itself, so that
        they keep the same length however large that surface is.

        :param k_i: Incident direction of propagation in the LCS
        :param positions: Positions of the scattering points in the LCS [m]
        :param head_radius: Distance from every scattering point at which the
            arrow stops [m]
        :param length: Length of the arrows [m]
        """
        direction = np.array(dr.normalize(k_i).numpy()).reshape(3)

        ends = positions - head_radius[:, None]*direction[None, :]
        starts = ends - length*direction[None, :]

        self._plot_arrows(starts, ends, np.array(INCIDENT_COLOR),
                          0.15*length, on_top=True)
        self._add_legend_items([
            ("Incident direction", self._arrow_legend_item(INCIDENT_COLOR))])

    def _plot_lobes(self,
                    spst,
                    k_i: mi.Vector3f,
                    positions: np.ndarray,
                    lobe_radius: float,
                    normalization: str,
                    radial_scale: str) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Plots the cross-section surface of every scattering point

        Every surface is a spherical grid of directions whose radius and color
        are set by the cross-section scattered in that direction. The grids of
        every point share their triangles, so all of them are drawn at once.

        :param spst: Scattering points of the model
        :param k_i: Incident direction of propagation in the LCS
        :param positions: Positions of the scattering points in the LCS [m]
        :param lobe_radius: Radius of the strongest direction [m]
        :param normalization: Either ``"global"`` or ``"per_point"``
        :param radial_scale: Either ``"db"`` or ``"linear"``

        :return: Lowest cross-section every surface shows [dBsm]
        :return: Highest cross-section of every point [dBsm]
        """
        num_points = positions.shape[0]
        num_zenith = self._num_zenith
        num_azimuth = self._num_azimuth
        num_directions = num_zenith*num_azimuth

        # Scattered directions, with the zenith angle varying the slowest so
        # that the direction of index `i*num_azimuth + j` has zenith `i` and
        # azimuth `j`
        theta = dr.linspace(mi.Float, 0., dr.pi, num_zenith)
        phi = dr.linspace(mi.Float, -dr.pi, dr.pi, num_azimuth, False)
        theta, phi = dr.meshgrid(theta, phi, indexing="ij")
        k_s = r_hat(theta, phi)

        sigma = _eval_sigma(spst, k_i, dr.tile(k_s, num_points),
                            dr.repeat(dr.arange(mi.UInt32, num_points),
                                      num_directions))
        sigma = sigma.numpy().reshape(num_points, num_directions)

        sigma_db = 10.*np.log10(np.maximum(sigma, _MIN_SIGMA))
        vmax = np.max(sigma_db, axis=1, keepdims=True)
        vmin = np.min(sigma_db, axis=1, keepdims=True)
        if normalization == "global":
            vmax = np.full_like(vmax, np.max(vmax))
            vmin = np.full_like(vmin, np.min(vmin))
        vmin = _clip_range(vmin, vmax, self._dynamic_range)

        # The color always encodes the cross-section in dB, so that the
        # colorbar remains meaningful whichever radial scale is used
        levels = _normalize_db(sigma_db, vmin, vmax - vmin)
        radii = self._radii(sigma, sigma_db, vmin, vmax, radial_scale)

        vertices = positions[:, None, :] \
            + (lobe_radius*radii)[:, :, None]*np.transpose(k_s.numpy())
        vertices = vertices.reshape(-1, 3).astype(np.float32)
        colors = self._colormap(levels.reshape(-1))[:, :3].astype(np.float32)

        faces = _sphere_grid_faces(num_zenith, num_azimuth)
        offsets = num_directions*np.arange(num_points, dtype=np.uint32)
        faces = (faces[None, :, :] + offsets[:, None, None]).reshape(-1, 3)

        geometry = p3s.BufferGeometry(attributes={
            "index": p3s.BufferAttribute(faces.ravel(), normalized=False),
            "position": p3s.BufferAttribute(vertices, normalized=False),
            "color": p3s.BufferAttribute(colors, normalized=False),
        })
        # The color of a surface is the value it encodes, so it must not be
        # shaded by the lighting of the viewer
        material = p3s.MeshBasicMaterial(vertexColors="VertexColors",
                                         side="DoubleSide")
        mesh = p3s.Mesh(geometry, material)
        self._add_child(mesh, np.min(vertices, axis=0),
                        np.max(vertices, axis=0), persist=False)

        return vmin, vmax

    def _backscatter_radius(self,
                            spst,
                            k_i: mi.Vector3f,
                            lobe_radius: float,
                            radial_scale: str,
                            vmin: np.ndarray,
                            vmax: np.ndarray) -> np.ndarray:
        r"""
        Returns the radius of the surface of every scattering point in the
        direction the incident wave comes from

        The cross-section is evaluated exactly in that direction rather than
        interpolated on the grid of the surface.

        :param spst: Scattering points of the model
        :param k_i: Incident direction of propagation in the LCS
        :param lobe_radius: Radius of the strongest direction [m]
        :param radial_scale: Either ``"db"`` or ``"linear"``
        :param vmin: Lowest cross-section every surface shows [dBsm]
        :param vmax: Highest cross-section of every point [dBsm]

        :return: Radius of the backscattered direction of every point [m]
        """
        num_points = dr.width(spst.lcs_positions)

        # Backscattering, i.e. the scattered direction of propagation points
        # back to where the incident one came from
        k_s = dr.tile(-dr.normalize(mi.Vector3f(k_i)), num_points)
        sigma = _eval_sigma(spst, k_i, k_s,
                            dr.arange(mi.UInt32, num_points))
        sigma = sigma.numpy().reshape(num_points, 1)
        sigma_db = 10.*np.log10(np.maximum(sigma, _MIN_SIGMA))
        radii = self._radii(sigma, sigma_db, vmin, vmax, radial_scale)

        return lobe_radius*radii.reshape(-1)

    def _radii(self,
               sigma: np.ndarray,
               sigma_db: np.ndarray,
               vmin: np.ndarray,
               vmax: np.ndarray,
               radial_scale: str) -> np.ndarray:
        r"""
        Maps cross-sections to radii in the range `[0,1]`

        :param sigma: Cross-sections [:math:`\text{m}^2`], of shape `[n, m]`
        :param sigma_db: The same cross-sections [dBsm]
        :param vmin: Value mapped to 0 of every point [dBsm], of shape `[n,1]`
        :param vmax: Value mapped to 1 of every point [dBsm], of shape `[n,1]`
        :param radial_scale: Either ``"db"`` or ``"linear"``

        :return: Radii, of shape `[n, m]`
        """
        if radial_scale == "db":
            radii = _normalize_db(sigma_db, vmin, vmax - vmin)
        else:
            radii = np.clip(sigma*np.power(10., -0.1*vmax), 0., 1.)

        # A point which scatters nothing in any direction has no shape to
        # show, so its surface collapses onto it rather than being stretched
        # to the full radius by the normalization
        return np.where(vmax <= _MIN_SIGMA_DB, 0., radii)

    def _add_colorbar(self,
                      vmin: float,
                      vmax: float,
                      relative: bool,
                      num_stops: int = 33):
        r"""
        Adds to the legend a colorbar of the cross-section surfaces

        PyThreeJS has no colorbar, so it is built as an HTML gradient.

        :param vmin: Value at the bottom of the colorbar [dBsm]
        :param vmax: Value at the top of the colorbar [dBsm]
        :param relative: If set to `True`, the colorbar is labelled relative
            to the strongest direction of every surface rather than in
            absolute terms, as is required when every surface carries its own
            normalization and hence its own range
        :param num_stops: Number of colors sampled from the colormap
        """
        stops = ", ".join(
            f"{rgb_to_html(self._colormap(v)[:3])} {100.*v:.1f}%"
            for v in np.linspace(0., 1., num_stops))

        if relative:
            title = "RCS"
            labels = ("peak", "", "weakest")
        else:
            title = "RCS [dBsm]"
            labels = (f"{vmax:.1f}", f"{0.5*(vmin + vmax):.1f}",
                      f"{vmin:.1f}")

        ticks = "".join(f"<span>{label}</span>" for label in labels)

        self._colorbar = widgets.HTML(value=(
            '<div style="margin-top: 10px; font-size: 11px;">'
            f'{title}'
            '<div style="display: flex; height: 150px; margin-top: 3px;">'
            '<div style="width: 14px; border: 1px solid #888;'
            f' background: linear-gradient(to top, {stops});"></div>'
            '<div style="display: flex; flex-direction: column;'
            ' justify-content: space-between; margin-left: 4px;">'
            f'{ticks}</div></div></div>'
        ))


def _camera_angles(k_i: mi.Vector3f) -> Tuple[float, float]:
    r"""
    Returns the zenith and azimuth angles [rad] of the default camera

    The incident arrows are foreshortened, down to single points, as the
    camera approaches their axis. Among the candidate positions on the
    illuminated side of the target, the one furthest from that axis is
    therefore chosen. If no candidate lies on that side, e.g. for a wave
    coming from below, the candidate furthest from the axis is chosen.

    :param k_i: Incident direction of propagation in the LCS
    """
    source = -np.array(dr.normalize(k_i).numpy()).reshape(3)

    def cos_source(angles):
        theta, phi = angles
        position = np.array([np.sin(theta)*np.cos(phi),
                             np.sin(theta)*np.sin(phi),
                             np.cos(theta)])
        return float(position @ source)

    # Rounding keeps candidates on the threshold, and lets symmetric
    # candidates tie so that the order of preference decides between them,
    # rather than rounding errors
    illuminated = [a for a in _CAMERA_ANGLES
                   if round(cos_source(a), 6) >= _MIN_COS_ILLUMINATED]
    return min(illuminated or _CAMERA_ANGLES,
               key=lambda a: round(abs(cos_source(a)), 6))


def _unposed_geometry(target) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Returns the vertices and faces of the mesh of a sensing target, in the
    local coordinate system (LCS) of that target

    The position and orientation of a scene object are baked into the vertices
    of its mesh, whereas its scattering points are given in its LCS. The
    vertices are therefore brought back to that frame, which is the inverse of
    the transformation :class:`~sionna.rt.SceneObject` applied to them. The
    scaling of the target is kept, so the returned geometry has the size the
    target is displayed with, and the scattering points shown alongside it
    must be scaled to match.

    Note that the pose of an object is only fully described by its position
    and orientation as long as its axis-aligned bounding box (AABB) is
    centered on the origin of its LCS, as the position of an object is the
    center of its AABB. Rotating a mesh which is not symmetric about that
    center moves it, in which case the box shown by the viewer is offset from
    the scattering points by that same amount, exactly as the solver sees it.

    :param target: Sensing target
    :return: Vertices of the mesh in the LCS [m], of shape `[n, 3]`
    :return: Indices of the triangles of the mesh, of shape `[m, 3]`
    """
    mesh = target.mi_mesh

    vertices = mesh.vertex_position(dr.arange(mi.UInt32, mesh.vertex_count()))
    to_local = rotation_matrix(target.orientation).T
    vertices = to_local@(vertices - target.position)

    faces = mesh.face_indices(dr.arange(mi.UInt32, mesh.face_count()))

    # Transposing yields a view, which is copied as the buffers uploaded for
    # display must be contiguous
    return (np.ascontiguousarray(np.transpose(vertices.numpy())),
            np.ascontiguousarray(np.transpose(faces.numpy())))


def _lobe_extent(box_min: np.ndarray,
                 box_max: np.ndarray,
                 positions: np.ndarray) -> float:
    r"""
    Returns the radius at which the strongest direction of a cross-section
    surface is drawn

    A surface should be as large as possible to be readable, while remaining
    within the target and not reaching into its neighbours. The smallest
    dimension of the target and the distance to the closest other scattering
    point respectively bound these two.

    Degenerate dimensions, such as those of a flat mesh, and points sharing a
    position are left out, as they would otherwise leave nothing to display.
    If everything is degenerate, an arbitrary radius of 1 m is returned.

    :param box_min: Lowest corner of the bounding box of the target [m]
    :param box_max: Highest corner of the bounding box of the target [m]
    :param positions: Positions of the scattering points in the LCS [m]

    :return: Radius of the strongest direction [m]
    """
    bounds = [0.5*float(size) for size in box_max - box_min if size > 0.]

    num_points = positions.shape[0]
    if num_points > 1:
        distances = np.linalg.norm(positions[:, None, :] - positions[None],
                                   axis=-1)
        distances = distances[~np.eye(num_points, dtype=bool)]
        distances = distances[distances > 0.]
        if distances.size > 0:
            bounds.append(0.5*float(np.min(distances)))

    return min(bounds) if bounds else 1.


def _box_edges(box_min: np.ndarray,
               box_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Returns the 12 edges of an axis-aligned box

    :param box_min: Lowest corner of the box
    :param box_max: Highest corner of the box

    :return: Starting points of the edges, of shape `[12, 3]`
    :return: Ending points of the edges, of shape `[12, 3]`
    """
    bounds = np.stack((box_min, box_max))
    # Corner `i` takes its k-th coordinate from `box_max` if the k-th bit of
    # `i` is set, and from `box_min` otherwise
    corners = np.array([[bounds[(i >> k) & 1, k] for k in range(3)]
                        for i in range(8)])

    # Two corners are joined by an edge if they differ along a single axis
    starts, ends = [], []
    for i in range(8):
        for k in range(3):
            if not (i >> k) & 1:
                starts.append(corners[i])
                ends.append(corners[i | (1 << k)])

    return np.array(starts), np.array(ends)


def _sphere_grid_faces(num_zenith: int, num_azimuth: int) -> np.ndarray:
    r"""
    Returns the triangles of a grid of directions over the sphere

    The grid is indexed as `i*num_azimuth + j`, with `i` the index of the
    zenith angle and `j` the index of the azimuth angle. The azimuth angles
    span the sphere without repeating the first one, so the last column of the
    grid is joined back to the first one. The rows at both poles hold
    `num_azimuth` copies of a single direction, which only yields degenerate
    triangles of zero area.

    :param num_zenith: Number of zenith angles
    :param num_azimuth: Number of azimuth angles

    :return: Indices of the triangles, of shape `[m, 3]`
    """
    i = np.arange(num_zenith - 1)
    j = np.arange(num_azimuth)
    i, j = np.meshgrid(i, j, indexing="ij")
    j_next = (j + 1) % num_azimuth

    lower = i*num_azimuth + j
    lower_next = i*num_azimuth + j_next
    upper = (i + 1)*num_azimuth + j
    upper_next = (i + 1)*num_azimuth + j_next

    # Every cell of the grid is split into two triangles
    faces = np.concatenate((
        np.stack((lower, upper, upper_next), axis=-1).reshape(-1, 3),
        np.stack((lower, upper_next, lower_next), axis=-1).reshape(-1, 3),
    ))

    return faces.astype(np.uint32)


def _eval_sigma(spst,
                k_i: mi.Vector3f,
                k_s: mi.Vector3f,
                spst_indices: mi.UInt32) -> mi.Float:
    r"""
    Evaluates the bistatic radar cross-section of scattering points

    Leaving the orientation of the sensing target at zero reduces the
    world-to-local transformation of
    :meth:`~sionna.rt.rcs.ScatteringPoints.eval_rcs` to the orientation of
    the scattering point alone, which is what directions given in the local
    coordinate system of the target call for.

    :param spst: Scattering points of the model
    :param k_i: Incident direction of propagation in the LCS
    :param k_s: Scattered directions of propagation in the LCS, one per sample
    :param spst_indices: Index of the scattering point of every sample

    :return: Bistatic radar cross-section of every sample [:math:`\text{m}^2`]
    """
    num_samples = dr.width(k_s)

    k_i = dr.tile(dr.normalize(mi.Vector3f(k_i)), num_samples)
    return spst.eval_rcs(k_i, dr.normalize(k_s), mi.Point3f(0.),
                         dr.zeros(mi.UInt32, num_samples), spst_indices)


def _clip_range(vmin: np.ndarray,
                vmax: np.ndarray,
                dynamic_range: float) -> np.ndarray:
    r"""
    Clips the range spanned by a surface to a usable one

    Surfaces span the range of the cross-sections they show rather than a
    fixed window, so that their shape comes out whatever the directivity of
    the point. That range is bounded below, as the deepest nulls would
    otherwise flatten everything else, and above, as a point scattering the
    same cross-section in every direction has no range at all.

    :param vmin: Lowest cross-section of every point [dBsm]
    :param vmax: Highest cross-section of every point [dBsm]
    :param dynamic_range: Largest range a surface may span [dB]

    :return: Lowest cross-section every surface shows [dBsm]
    """
    vmin = np.maximum(vmin, vmax - dynamic_range)
    return np.minimum(vmin, vmax - _MIN_RANGE)


def _normalize_db(values_db: np.ndarray,
                  vmin: np.ndarray,
                  span: np.ndarray) -> np.ndarray:
    r"""
    Maps values in dB to the range `[0,1]`, with the bottom of the range
    mapped to 0

    :param values_db: Values to map [dB]
    :param vmin: Value mapped to 0 [dB]
    :param span: Range mapped to `[0,1]` [dB]

    :return: Mapped values
    """
    return np.clip((values_db - vmin)/span, 0., 1.)
