#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Interactive 3D viewer of Sionna RT"""

import mitsuba as mi
import numpy as np
from ipywidgets import widgets
from ipywidgets.embed import embed_snippet
import pythreejs as p3s
from IPython.display import display


class Viewer:
    """
    Interactive viewer widget using `pythreejs`

    This class holds the display machinery that does not depend on what is
    being visualized: the camera and its controls, the renderer, the legend,
    and helpers to plot meshes, points, lines, and arrows. Subclasses add the
    methods that plot the objects of interest, such as
    :class:`~sionna.rt.Previewer` for a scene.

    Input
    ------
    resolution: [2], int
        Size of the viewer figure.
        Defaults to (655,500).

    fov: float
        Field of view, in degrees.
        Defaults to 45 degrees.

    background: str
        Background color in hex format prefixed by '#'.
        Defaults to 'white'.
    """

    def __init__(self, resolution=(655,500), fov=45., background='white'):

        self._disk_sprite = None

        # List of objects being displayed
        self._objects = []
        # Bounding box of the objects being displayed
        self._bbox = mi.ScalarBoundingBox3f()

        ####################################################
        # Setup the viewer
        ####################################################

        # Lighting
        ambient_light = p3s.AmbientLight(intensity=0.90)
        camera_light = p3s.DirectionalLight(
            position=[0, 0, 0], intensity=0.25
        )

        # Camera & controls
        self._camera = p3s.PerspectiveCamera(
            fov=fov, aspect=resolution[0] / resolution[1],
            up=[0, 0, 1], children=[camera_light],
        )

        self._orbit = p3s.OrbitControls(
            controlling = self._camera
        )

        # Scene & renderer
        self._p3s_scene = p3s.Scene(
            background=background, children=[self._camera, ambient_light]
        )
        self._renderer = p3s.Renderer(
            scene=self._p3s_scene, camera=self._camera, controls=[self._orbit],
            width=resolution[0], height=resolution[1], antialias=True
        )

        self._legend_labels: dict[str, widgets.HTML] = {}

    def reset(self):
        """
        Removes objects that are not flagged as persistent
        """
        remaining = []
        for obj, persist in self._objects:
            if persist:
                remaining.append((obj, persist))
            else:
                self._p3s_scene.remove(obj)
        self._objects = remaining

    def display(self):
        """Display the viewer and its companion widgets in a Jupyter
        notebook."""

        display(self._display_widget())

    def center_view(self, theta: float = np.pi*0.25, phi: float = np.pi*0.25):
        """
        Automatically place the camera such that it is located at spherical
        coordinates (r = scene_scale*3/2, theta, phi) relative to the center
        of the displayed objects, and oriented toward that center.

        :param theta: Zenith angle of the camera [rad]
        :param phi: Azimuth angle of the camera [rad]
        """
        bbox = self._bbox if self._bbox.valid() else mi.ScalarBoundingBox3f(0.)
        center = bbox.center()

        sc = self._scene_scale()
        r = sc * 1.5
        position = (np.sin(theta) * np.cos(phi) * r + center.x,
                    np.sin(theta) * np.sin(phi) * r + center.y,
                    np.cos(theta) * r + center.z)
        self._camera.position = tuple(position)

        self._camera.lookAt(center)
        self._orbit.exec_three_obj_method('update')
        self._camera.exec_three_obj_method('updateProjectionMatrix')

    ##################################################
    # Accessors
    ##################################################

    @property
    def resolution(self) -> tuple[int, int]:
        """
        (float, float): Rendering resolution `(width, height)`
        """
        return (self._renderer.width, self._renderer.height)

    @property
    def camera(self) -> p3s.PerspectiveCamera:
        return self._camera

    @property
    def orbit(self) -> p3s.OrbitControls:
        return self._orbit

    ##################################################
    # Internal methods
    ##################################################

    def _display_widget(self):
        """
        Returns the widget holding the renderer and its legend column

        A new widget is built on every call, as the legend grows as objects
        are added to the viewer.

        Output
        -------
        : :class:`~ipywidgets.HBox`
            Widget to display
        """
        legend = widgets.VBox(
            self._legend_widgets(),
            layout=widgets.Layout(padding="0 0 0 5px")
        )
        return widgets.HBox([self._renderer, legend])

    def _scene_scale(self):
        """
        Returns the size of the displayed objects, i.e., the diameter of the
        smallest sphere containing them.
        If nothing is displayed, the scale is arbitrarily set to 1.

        Output
        -------
        : float
            Size of the displayed objects
        """
        if not self._bbox.valid():
            return 1.
        sc = 2. * self._bbox.bounding_sphere().radius
        if np.isnan(sc) or (sc == 0.):
            sc = 1.
        return sc

    def _plot_mesh(self, vertices, faces, persist, colors=None, opacity=1.):
        """
        Plots a mesh.

        Input
        ------
        vertices: [n,3], float
            Position of the vertices

        faces: [n,3], int
            Indices of the triangles associated with ``vertices``

        persist: bool
            Flag indicating if the mesh is persistent, i.e., should not be
            erased when ``reset()`` is called.

        colors: [n,3] | [3] | None
            Colors of the vertices. If `None`, black is used.
            Defaults to `None`.

        opacity: float
            Opacity of the mesh, within the range [0,1].
            Defaults to 1, i.e., a fully opaque mesh.
        """
        if not (vertices.ndim == 2 and vertices.shape[1] == 3):
            raise ValueError("`vertices` must have shape [n, 3]")
        if not (faces.ndim == 2 and faces.shape[1] == 3):
            raise ValueError("`faces` must have shape [n, 3]")
        n_v = vertices.shape[0]
        pmin, pmax = np.min(vertices, axis=0), np.max(vertices, axis=0)

        # Assuming per-vertex colors
        if colors is None:
            # Black is default
            colors = np.zeros((n_v, 3), dtype=np.float32)
        elif colors.ndim == 1:
            colors = np.tile(colors[None, :], (n_v, 1))
        colors = colors.astype(np.float32)
        if not ((colors.ndim == 2)
                and (colors.shape[1] == 3)
                and (colors.shape[0] == n_v)):
            raise ValueError(
                "`colors` must have shape [n, 3] matching `vertices`")

        # Closer match to Mitsuba and Blender
        colors = np.power(colors, 1/1.8)

        geo = p3s.BufferGeometry(
            attributes={
                'index': p3s.BufferAttribute(faces.ravel(), normalized=False),
                'position': p3s.BufferAttribute(vertices, normalized=False),
                'color': p3s.BufferAttribute(colors, normalized=False)
            }
        )

        translucent = opacity < 1.
        mat = p3s.MeshStandardMaterial(
            side='DoubleSide', metalness=0., roughness=1.0,
            vertexColors='VertexColors', flatShading=True,
            transparent=translucent, opacity=opacity,
            # Translucent meshes must not write depth, as they would otherwise
            # hide the paths and radio devices located behind them.
            depthWrite=not translucent,
        )
        mesh = p3s.Mesh(geo, mat)
        self._add_child(mesh, pmin, pmax, persist=persist)

    def _plot_points(self, points: np.ndarray, persist: bool,
                     colors: np.ndarray | None = None,
                     radius: float = 0.05,
                     on_top: bool = False):
        """
        Plots a set of `n` points.

        Input
        -------
        points: [n, 3], float
            Coordinates of the `n` points.

        persist: bool
            Indicates if the points are persistent, i.e., should not be erased
            when ``reset()`` is called.

        colors: [n, 3], float | [3], float | None
            Colors of the points.

        radius: float
            Radius of the points.

        on_top: bool
            If set to `True`, the points are drawn over everything else
            instead of being hidden by the objects in front of them, which
            keeps markers visible whatever the point of view.
            Defaults to `False`.
        """
        if not (points.ndim == 2 and points.shape[1] == 3):
            raise ValueError("`points` must have shape [n, 3]")
        n = points.shape[0]
        pmin, pmax = np.min(points, axis=0), np.max(points, axis=0)

        # Assuming per-vertex colors
        if colors is None:
            colors = np.zeros((n, 3), dtype=np.float32)
        elif colors.ndim == 1:
            colors = np.tile(colors[None, :], (n, 1))
        colors = colors.astype(np.float32)
        if not ((colors.ndim == 2)
                and (colors.shape[1] == 3)
                and (colors.shape[0] == n)):
            raise ValueError(
                "`colors` must have shape [n, 3] matching `points`")

        tex = p3s.DataTexture(data=self._get_disk_sprite(), format="RGBAFormat",
                              type="FloatType")

        points = points.astype(np.float32)
        geo = p3s.BufferGeometry(attributes={
            'position': p3s.BufferAttribute(points, normalized=False),
            'color': p3s.BufferAttribute(colors, normalized=False),
        })
        mat = p3s.PointsMaterial(
            size=2 * radius, sizeAttenuation=True, vertexColors='VertexColors',
            map=tex, alphaTest=0.5, transparent=True,
        )
        mesh = p3s.Points(geo, mat)
        if on_top:
            self._draw_on_top(mesh, mat)
        self._add_child(mesh, pmin, pmax, persist=persist)

    def _plot_lines(self, starts, ends, colors, width, on_top=False):
        """
        Plots a set of `n` lines.

        Input
        ------
        starts: [n, 3], float
            Coordinates of the lines starting points

        ends: [n, 3], float
            Coordinates of the lines ending points

        color: str
            Color of the lines.

        width: float
            Width of the lines.

        on_top: bool
            If set to `True`, the lines are drawn over everything else instead
            of being hidden by the objects in front of them.
            Defaults to `False`.
        """

        if not (starts.ndim == 2 and starts.shape[1] == 3):
            raise ValueError("`starts` must have shape [n, 3]")
        if not (ends.ndim == 2 and ends.shape[1] == 3):
            raise ValueError("`ends` must have shape [n, 3]")
        if starts.shape[0] != ends.shape[0]:
            raise ValueError("`starts` and `ends` must have the same length")

        segments = np.hstack((starts, ends)).astype(np.float32).reshape(-1,2,3)
        pmin = np.min(segments, axis=(0, 1))
        pmax = np.max(segments, axis=(0, 1))

        colors = np.hstack((colors, colors)).astype(np.float32).reshape(-1,2,3)
        geo = p3s.LineSegmentsGeometry(positions=segments, colors=colors)
        mat = p3s.LineMaterial(linewidth=width, vertexColors='VertexColors')
        mesh = p3s.LineSegments2(geo, mat)
        if on_top:
            self._draw_on_top(mesh, mat)

        # Lines are not flagged as persistent as they correspond to objects,
        # such as paths, which can change from one display to the next.
        self._add_child(mesh, pmin, pmax, persist=False)

    def _plot_arrows(self, starts, ends, colors, head_lengths, width=2.,
                     on_top=False):
        """
        Plots a set of `n` arrows, each drawn as a line segment ending in a
        cone whose tip is located at the end of the arrow.

        Input
        ------
        starts: [n, 3], float
            Coordinates of the arrows starting points

        ends: [n, 3], float
            Coordinates of the arrows ending points, where the cones are placed

        colors: [n, 3], float | [3], float
            Colors of the arrows

        head_lengths: [n], float | float
            Lengths of the cones

        width: float
            Width of the lines.
            Defaults to 2.

        on_top: bool
            If set to `True`, the arrows are drawn over everything else
            instead of being hidden by the objects in front of them, which
            keeps them visible however short they are.
            Defaults to `False`.
        """
        starts = np.asarray(starts, dtype=np.float32).reshape(-1, 3)
        ends = np.asarray(ends, dtype=np.float32).reshape(-1, 3)
        if starts.shape[0] != ends.shape[0]:
            raise ValueError("`starts` and `ends` must have the same length")
        n = starts.shape[0]

        colors = np.asarray(colors, dtype=np.float32)
        if colors.ndim == 1:
            colors = np.tile(colors[None, :], (n, 1))
        head_lengths = np.broadcast_to(
            np.asarray(head_lengths, dtype=np.float32), (n,))

        # The cones are small compared to the arrows, so they are not used to
        # extend the bounding box of the displayed objects
        zeros = np.zeros((3,))

        for start, end, color, head_length in zip(starts, ends, colors,
                                                  head_lengths):
            geo = p3s.CylinderGeometry(
                radiusTop=0, radiusBottom=0.3 * head_length,
                height=head_length, radialSegments=8,
                heightSegments=0, openEnded=False)
            mat = p3s.MeshLambertMaterial(color=rgb_to_html(color))
            mesh = p3s.Mesh(geo, mat)
            if on_top:
                self._draw_on_top(mesh, mat)

            direction = end - start
            norm = np.linalg.norm(direction)
            if norm == 0.:
                continue
            mesh.lookAt(tuple(direction / norm))
            mesh.rotateX(0.5 * np.pi)

            mesh.position = tuple(float(v) for v in end)

            self._add_child(mesh, zeros, zeros, persist=False)

        self._plot_lines(starts, ends, colors, width, on_top=on_top)

    @staticmethod
    def _draw_on_top(obj, material):
        """
        Makes an object be drawn over everything else instead of being hidden
        by the objects in front of it, as annotations must be.

        Input
        ------
        obj: :class:`~pythreejs.Object3D`
            Object to draw on top

        material: :class:`~pythreejs.Material`
            Material of that object
        """
        material.depthTest = False
        # Ignoring the depth buffer is not enough on its own, as the objects
        # drawn after this one would otherwise paint over it
        obj.renderOrder = 1

    def _add_child(self, obj, pmin, pmax, persist):
        """
        Adds an object for display

        Input
        ------
        obj: :class:`~pythreejs.Mesh`
            Mesh to display

        pmin: [3], float
            Lowest position for the bounding box

        pmax: [3], float
            Highest position for the bounding box

        persist: bool
            Flag that indicates if the object is persistent, i.e., if it should
            be removed from the display when `reset()` is called.
        """
        self._objects.append((obj, persist))
        self._p3s_scene.add(obj)

        self._bbox.expand(pmin)
        self._bbox.expand(pmax)

    def _legend_widgets(self) -> list[widgets.Widget]:
        r"""
        Returns the widgets making the legend column displayed next to the
        renderer
        """
        return list(self._legend_labels.values())

    def _add_legend_items(self, legend_items):
        r"""
        Adds entries to the legend

        Input
        ------
        legend_items: [n], (str, str)
            Label and CSS style of every entry
        """
        self._legend_labels.update({
            label: widgets.HTML(value=f"<div style='{style}'></div> {label}")
            for label, style in legend_items
        })

    @staticmethod
    def _circular_legend_item(color, diameter: int = 20):
        r"""
        Returns the CSS style of a legend entry shown as a disk of the given
        color and diameter [px]
        """
        #pylint: disable=unnecessary-semicolon
        # Narrower disks are centered on the text rather than sat on its
        # baseline, which would leave them hanging below a line of it
        align = "text-bottom" if diameter >= 20 else "middle"
        s = f"background-color: {rgb_to_html(color)};" \
            f" width: {diameter}px; height: {diameter}px;" \
            " border-radius: 50%;" \
            f" margin-right: {25 - diameter}px; display: inline-block;" \
            f" vertical-align: {align};"
        return s

    @staticmethod
    def _arrow_legend_item(color):
        r"""
        Returns the CSS style of a legend entry shown as an arrow of the given
        color, pointing to the right
        """
        #pylint: disable=unnecessary-semicolon
        # A legend entry is styled from a single element, so the shaft and the
        # head cannot be drawn as two of them. A box is clipped to the outline
        # of an arrow instead, which falls back to that plain box should the
        # browser not support clipping.
        s = f"background-color: {rgb_to_html(color)};" \
            " width: 20px; height: 8px;" \
            " clip-path: polygon(0 3px, 12px 3px, 12px 0," \
            " 20px 4px, 12px 8px, 12px 5px, 0 5px);" \
            " margin-right: 5px; display: inline-block;" \
            " vertical-align: middle;"
        return s

    @staticmethod
    def _segment_legend_item(color):
        r"""
        Returns the CSS style of a legend entry shown as a segment of the given
        color
        """
        #pylint: disable=unnecessary-semicolon
        s = f"background-color: {rgb_to_html(color)};" \
            " width: 20px; height: 2px;" \
            " margin-right: 5px; display: inline-block;" \
            " vertical-align: middle;"
        return s

    def _get_disk_sprite(self):
        """
        Returns the sprite used to represent points though ``_plot_points()``.

        Output
        ------
        : [n,n,4], float
            Sprite
        """
        if self._disk_sprite is not None:
            return self._disk_sprite

        n = 128
        sprite = np.ones((n, n, 4))
        sprite[:, :, 3] = 0.
        # Draw a disk with an empty circle close to the edge
        ij = np.mgrid[:n, :n]
        ij = ij.reshape(2, -1)

        p = (ij + 0.5) / n - 0.5
        t = np.linalg.norm(p, axis=0).reshape((n, n))
        inside = t < 0.48
        in_band = (t < 0.45) & (t > 0.42)
        sprite[inside & (~in_band), 3] = 1.0

        sprite = sprite.astype(np.float32)
        self._disk_sprite = sprite
        return sprite

    # The following methods are required for
    # integration in Jupyter notebooks

    # pylint: disable=unused-argument
    def _repr_mimebundle_(self, **kwargs):
        # pylint: disable=protected-access,not-callable
        # The legend is part of the displayed widget, so that a viewer
        # returned as the result of a cell shows it as `display()` would
        widget = self._display_widget()
        bundle = widget._repr_mimebundle_()
        if 'text/html' in bundle:
            raise RuntimeError(
                "Unexpected 'text/html' entry in renderer MIME bundle")
        bundle['text/html'] = embed_snippet(widget, requirejs=True)
        return bundle

    def _repr_html_(self):
        """
        Standalone HTML display, i.e. outside of an interactive Jupyter
        notebook environment.
        """

        html = embed_snippet(self._display_widget(), requirejs=True)
        return html


def rgb_to_html(rgb: tuple[float, float, float]) -> str:
    """
    Convert an RGB tuple to an HTML color string.
    """
    return f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"
