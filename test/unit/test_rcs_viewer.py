#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the interactive viewer of a scattering model."""

import numpy as np
import pytest
import drjit as dr
import mitsuba as mi
from ipywidgets import widgets
import pythreejs as p3s

from sionna.rt.rcs import (ConstantCPM, ConstantRCS, ConstantRCSSensingTarget,
                           ScatteringModel, ScatteringModelViewer,
                           SensingTarget, TR38901SensingTarget)
from sionna.rt.utils import load_mesh, r_hat
from sionna.rt.viewer import Viewer
import sionna.rt


def _cuboid_target(rcs=None, cpm=None, positions=None, orientations=None,
                   **kwargs):
    """Cuboid target of 4m x 2m x 1.5m carrying the given scattering points."""
    if positions is None:
        positions = mi.Point3f(0., 0., 0.)
    if rcs is None:
        rcs = ConstantRCS(sigma=1.)
    if cpm is None:
        cpm = ConstantCPM()
    model = ScatteringModel(positions, orientations, rcs=rcs, cpm=cpm)
    return SensingTarget("st", scattering_model=model, length=4., width=2.,
                         height=1.5, **kwargs)


def _view(target, k_i, **kwargs):
    """Viewer of the scattering model of `target`, built without displaying
    it."""
    return ScatteringModelViewer(target, k_i, **kwargs)


def _lobe_mesh(viewer):
    """The single mesh holding the cross-section surfaces, or `None`."""
    meshes = [obj for obj, _ in viewer._objects
              if isinstance(obj, p3s.Mesh)
              and isinstance(obj.material, p3s.MeshBasicMaterial)]
    assert len(meshes) <= 1, "The surfaces must be drawn in a single mesh"
    return meshes[0] if meshes else None


def _line_segments(viewer):
    """Vertices of every set of line segments, each of shape [n, 2, 3]."""
    return [np.array(obj.geometry.positions) for obj, _ in viewer._objects
            if isinstance(obj, p3s.LineSegments2)]


def _box_segments(viewer):
    """Vertices of the wireframe box, of shape [12, 2, 3]."""
    # The box is the first set of line segments to be drawn
    return _line_segments(viewer)[0]


def _target_mesh_vertices(viewer):
    """Vertices of the mesh of the target, of shape [n, 3]."""
    meshes = [obj for obj, _ in viewer._objects
              if isinstance(obj, p3s.Mesh)
              and isinstance(obj.material, p3s.MeshStandardMaterial)]
    assert len(meshes) == 1
    return np.array(meshes[0].geometry.attributes["position"].array)


def test_viewer_builds_for_a_cuboid_target():
    viewer = _view(_cuboid_target(), k_i=(0., 0., -1.))

    assert _lobe_mesh(viewer) is not None
    # The mesh of the target, the box, the points, the three axes, and the
    # incident direction are all drawn
    assert len(viewer._objects) > 6


def test_viewer_builds_for_a_mesh_target():
    mesh = load_mesh(sionna.rt.scene.sphere)
    target = ConstantRCSSensingTarget("st", sigma=2., mi_mesh=mesh)

    viewer = _view(target, k_i=(1., 0., 0.))

    assert _lobe_mesh(viewer) is not None


def test_viewer_builds_for_a_multi_point_target():
    target = TR38901SensingTarget("car", object_type="vehicle-multi-sp")
    num_points = dr.width(target.scattering_model.spst.lcs_positions)
    assert num_points == 5

    viewer = _view(target, k_i=(1., 0., 0.), num_zenith=17, num_azimuth=32)

    vertices = np.array(
        _lobe_mesh(viewer).geometry.attributes["position"].array)
    assert vertices.shape == (num_points*17*32, 3)


@pytest.mark.parametrize("num_zenith,num_azimuth", ((9, 16), (33, 64)))
def test_viewer_lobe_vertex_and_face_counts(num_zenith, num_azimuth):
    positions = mi.Point3f([0., 1.], [0., 0.], [0., 0.])
    viewer = _view(_cuboid_target(positions=positions), k_i=(0., 0., -1.),
                   num_zenith=num_zenith, num_azimuth=num_azimuth)

    geometry = _lobe_mesh(viewer).geometry
    num_directions = num_zenith*num_azimuth

    vertices = np.array(geometry.attributes["position"].array)
    assert vertices.shape == (2*num_directions, 3)

    colors = np.array(geometry.attributes["color"].array)
    assert colors.shape == (2*num_directions, 3)

    # Two triangles per cell of the grid, the azimuth wrapping around so that
    # every zenith band is closed
    faces = np.array(geometry.attributes["index"].array)
    assert faces.size == 2*2*3*(num_zenith - 1)*num_azimuth
    # Every index addresses a vertex of the surface it belongs to
    assert faces.max() == 2*num_directions - 1


def test_viewer_skips_the_lobes_of_an_empty_model():
    target = SensingTarget("st", scattering_model=ScatteringModel(),
                           length=1., width=1., height=1.)

    viewer = _view(target, k_i=(0., 0., -1.))

    assert _lobe_mesh(viewer) is None
    # The geometry of the target is still shown
    assert len(viewer._objects) == 2


def test_viewer_omits_the_elements_that_are_turned_off():
    viewer = _view(_cuboid_target(), k_i=(0., 0., -1.), show_mesh=False,
                   show_box=False, show_orientations=False, show_lobes=False)

    # Only the scattering points and the incident direction remain
    assert _lobe_mesh(viewer) is None
    assert not any(isinstance(obj, p3s.Mesh) and
                   isinstance(obj.material, p3s.MeshStandardMaterial)
                   for obj, _ in viewer._objects)


def test_viewer_skips_the_mesh_of_a_fully_transparent_target():
    viewer = _view(_cuboid_target(display_opacity=0.), k_i=(0., 0., -1.))

    assert not any(isinstance(obj, p3s.Mesh) and
                   isinstance(obj.material, p3s.MeshStandardMaterial)
                   for obj, _ in viewer._objects)


def test_viewer_lobe_peaks_at_the_constant_cross_section():
    # A constant RCS scatters the same cross-section in every direction, so
    # every direction of the surface must sit at the top of the colormap and
    # at the largest radius
    sigma = 7.
    viewer = _view(_cuboid_target(rcs=ConstantRCS(sigma=sigma)),
                   k_i=(0., 0., -1.), num_zenith=9, num_azimuth=16,
                   pattern_scale=1.)

    geometry = _lobe_mesh(viewer).geometry
    radii = np.linalg.norm(np.array(geometry.attributes["position"].array),
                           axis=1)
    # The scattering point sits at the center of the target, whose smallest
    # half-dimension is 0.75m, and `pattern_scale` is 1
    assert np.allclose(radii, 0.75, atol=1e-5)

    # The colormap covers the cross-section of the point, which the colorbar
    # reports in absolute terms under the default global normalization
    assert np.isclose(viewer.cross_section_range[1], 10.*np.log10(sigma))
    assert f"{10.*np.log10(sigma):.1f}" in viewer._colorbar.value
    assert "RCS [dBsm]" in viewer._colorbar.value


def test_viewer_surfaces_ignore_the_cpm():
    # A CPM leaves the scattered power unchanged, so it must not show up in
    # the surfaces, which report the cross-section of the RCS alone
    kwargs = {"k_i": (0., 0., -1.), "num_zenith": 9, "num_azimuth": 16,
              "pattern_scale": 1.}
    plain = _view(_cuboid_target(rcs=ConstantRCS(sigma=7.)), **kwargs)
    depolarizing = _view(_cuboid_target(rcs=ConstantRCS(sigma=7.),
                                        cpm=ConstantCPM(xpr_db=0.)),
                         **kwargs)

    assert np.allclose(depolarizing.cross_section_range,
                       plain.cross_section_range)
    assert np.allclose(
        np.array(_lobe_mesh(depolarizing).geometry.attributes["position"]
                 .array),
        np.array(_lobe_mesh(plain).geometry.attributes["position"].array))


def test_viewer_clamps_the_lobes_to_the_dynamic_range():
    # A cross-section that vanishes in some directions must be floored rather
    # than produce an infinite radius
    def rcs(k_i, k_s, seed=0):
        # Scatters only towards the upper hemisphere
        return dr.select(k_s.z > 0., 1., 0.)

    viewer = _view(_cuboid_target(rcs=rcs), k_i=(0., 0., -1.), num_zenith=9,
                   num_azimuth=16, dynamic_range=20.)

    vertices = np.array(
        _lobe_mesh(viewer).geometry.attributes["position"].array)
    radii = np.linalg.norm(vertices, axis=1)

    assert np.all(np.isfinite(radii))
    # The floored directions collapse onto the scattering point, and the
    # strongest ones reach the smallest half-dimension of the target
    assert np.isclose(radii.min(), 0., atol=1e-5)
    assert np.isclose(radii.max(), 0.75, atol=1e-5)

    # The cross-section vanishes below the horizon, so it is the dynamic range
    # and not the data that sets the bottom of the range being shown
    assert np.allclose(viewer.cross_section_range, (-20., 0.), atol=1e-3)


def test_viewer_normalizations_scale_the_lobes_differently():
    # Two points of very different cross-sections, so that the normalization
    # is visible in the radii
    positions = mi.Point3f([-1., 1.], [0., 0.], [0., 0.])
    model = ScatteringModel(positions,
                            rcs=[ConstantRCS(sigma=1.),
                                 ConstantRCS(sigma=1e-3)],
                            cpm=ConstantCPM())
    target = SensingTarget("st", scattering_model=model, length=4., width=2.,
                           height=1.5)

    def radii_per_point(**kwargs):
        viewer = _view(target, k_i=(0., 0., -1.), num_zenith=9,
                       num_azimuth=16, **kwargs)
        vertices = np.array(
            _lobe_mesh(viewer).geometry.attributes["position"].array)
        vertices = vertices.reshape(2, -1, 3) \
            - np.transpose(positions.numpy())[:, None, :]
        return np.linalg.norm(vertices, axis=2)

    # Shared across the points, so the weaker one is 30dB down and therefore
    # smaller than the stronger one
    shared = radii_per_point(normalization="global", dynamic_range=40.)
    assert shared[1].max() < shared[0].max()

    # Normalized independently, so both reach the same size
    separate = radii_per_point(normalization="per_point", dynamic_range=40.)
    assert np.isclose(separate[0].max(), separate[1].max())

    # Every surface carries its own range in that case, so the colorbar can
    # only be labelled relative to the peak of each of them
    viewer = _view(target, k_i=(0., 0., -1.), normalization="per_point",
                   dynamic_range=40.)
    assert "peak" in viewer._colorbar.value
    assert "dBsm" not in viewer._colorbar.value

    # Both points are isotropic and hence have no range of their own, so the
    # smallest range a surface may span sets the bottom of the weaker one
    assert np.isclose(viewer.cross_section_range[1], 0., atol=1e-3)
    assert np.isclose(viewer.cross_section_range[0], -31., atol=1e-3)


def test_viewer_linear_radial_scale_is_proportional_to_the_cross_section():
    # A cross-section that halves between the two hemispheres, which a linear
    # radial scale must reproduce as a factor of two on the radii
    def rcs(k_i, k_s, seed=0):
        # Cross-section of 1 above the horizon and 0.5 below it
        return dr.select(k_s.z > 0., 1., 0.5)

    viewer = _view(_cuboid_target(rcs=rcs), k_i=(0., 0., -1.), num_zenith=9,
                   num_azimuth=16, radial_scale="linear")

    radii = np.linalg.norm(
        np.array(_lobe_mesh(viewer).geometry.attributes["position"].array),
        axis=1)
    assert np.isclose(radii.max(), 0.75)
    assert np.isclose(radii.min(), 0.375)


def test_viewer_lobes_stay_within_the_target_and_apart_from_each_other():
    # Surfaces large enough to leave the target or to reach into each other
    # hide the very geometry they are drawn against
    spacing = 0.4
    positions = mi.Point3f([-0.5*spacing, 0.5*spacing], [0., 0.], [0., 0.])
    target = _cuboid_target(positions=positions, rcs=ConstantRCS(sigma=1.))

    viewer = _view(target, k_i=(0., 0., -1.), num_zenith=9, num_azimuth=16)

    vertices = np.array(
        _lobe_mesh(viewer).geometry.attributes["position"].array)
    centers = np.transpose(positions.numpy())
    radii = np.linalg.norm(vertices.reshape(2, -1, 3) - centers[:, None, :],
                           axis=2)

    # The points are closer to each other than to the sides of the target, so
    # their spacing is what bounds the surfaces
    assert np.isclose(radii.max(), 0.5*spacing, atol=1e-5)

    # The surfaces therefore touch at most, and stay inside the target
    box = _box_segments(viewer).reshape(-1, 3)
    assert np.all(vertices >= box.min(axis=0) - 1e-4)
    assert np.all(vertices <= box.max(axis=0) + 1e-4)


def test_viewer_lobes_span_the_full_radius_range():
    # A pattern whose range is much narrower than the dynamic range must still
    # come out as a shape rather than as a sphere of nearly constant radius
    def rcs(k_i, k_s, seed=0):
        # Cross-section within [0.5, 1], i.e. a range of 3dB only
        return 0.75 + 0.25*k_s.z

    viewer = _view(_cuboid_target(rcs=rcs), k_i=(0., 0., -1.), num_zenith=9,
                   num_azimuth=16, dynamic_range=40.)

    radii = np.linalg.norm(
        np.array(_lobe_mesh(viewer).geometry.attributes["position"].array),
        axis=1)

    # The whole range of radii is used, from the scattering point out to the
    # largest radius, rather than the fraction a 40dB window would leave
    assert np.isclose(radii.max(), 0.75, atol=1e-5)
    assert np.isclose(radii.min(), 0., atol=1e-5)

    # The colormap covers the range of the pattern and nothing below it
    assert np.allclose(viewer.cross_section_range,
                       (10.*np.log10(0.5), 0.), atol=1e-3)


def test_viewer_orientation_arrows_are_drawn_over_the_lobes():
    # The arrows are shorter than the opaque surface they start in, so they
    # would be invisible if they were hidden by the objects in front of them
    orientations = mi.Point3f(dr.pi/2., 0., 0.)
    viewer = _view(_cuboid_target(rcs=ConstantRCS(sigma=1.),
                                  orientations=orientations),
                   k_i=(0., 0., -1.), show_box=False, num_zenith=9,
                   num_azimuth=16)

    # The three axes are drawn before the incident direction, and the box is
    # turned off
    axes = [obj for obj, _ in viewer._objects
            if isinstance(obj, p3s.LineSegments2)][:3]

    vertices = np.array(
        _lobe_mesh(viewer).geometry.attributes["position"].array)
    lobe_radius = np.linalg.norm(vertices, axis=1).max()

    # A rotation of 90 degrees about the z-axis takes the local x-axis of the
    # point onto +y and its local y-axis onto -x
    expected = ((0., 1., 0.), (-1., 0., 0.), (0., 0., 1.))
    for obj, unit in zip(axes, expected):
        start, end = np.array(obj.geometry.positions).reshape(2, 3)
        # The arrows start at the scattering point, which sits at the origin
        assert np.allclose(start, 0., atol=1e-5)
        assert np.allclose(end/np.linalg.norm(end), unit, atol=1e-5)

        # Three quarters of the smallest half-dimension of the target, which
        # the surface of the isotropic point reaches past
        assert np.isclose(np.linalg.norm(end), 0.75*0.75, atol=1e-5)
        assert np.linalg.norm(end) < lobe_radius

        assert not obj.material.depthTest
        assert obj.renderOrder > 0

    # The heads of the arrows are drawn over the surfaces as well
    cones = [obj for obj, _ in viewer._objects
             if isinstance(obj, p3s.Mesh)
             and isinstance(obj.material, p3s.MeshLambertMaterial)]
    assert len(cones) == 4
    assert all(not obj.material.depthTest and obj.renderOrder > 0
               for obj in cones)


def test_viewer_draws_the_scattering_points_over_the_lobes():
    # The surfaces are opaque and surround the point they belong to, so the
    # markers would be invisible if they were hidden by what is in front of
    # them, or if they were drawn before the surfaces
    viewer = _view(_cuboid_target(), k_i=(0., 0., -1.))

    markers = [obj for obj, _ in viewer._objects
               if isinstance(obj, p3s.Points)]
    assert len(markers) == 1
    assert not markers[0].material.depthTest
    assert markers[0].renderOrder > 0

    # Shown as black dots, in the view and in the legend alike
    colors = np.array(markers[0].geometry.attributes["color"].array)
    assert np.allclose(colors, 0.)
    assert "rgb(0, 0, 0)" in viewer._legend_labels["Scattering point"].value


def test_viewer_displays_the_colorbar_next_to_the_renderer():
    viewer = _view(_cuboid_target(), k_i=(0., 0., -1.))

    renderer, legend = viewer._display_widget().children
    assert renderer is viewer._renderer
    assert viewer._colorbar in legend.children

    # The legend must reach the notebook when the viewer is the result of a
    # cell, and not only when `display()` is called on it
    assert "dBsm" in viewer._repr_mimebundle_()["text/html"]


def test_viewer_box_is_the_local_bounding_box_of_a_posed_target():
    # The scattering points of a target are placed against its un-posed mesh,
    # so the box must be the bounding box of that mesh and not the
    # world-axis-aligned bounding box of the posed one, which a rotation
    # inflates
    dimensions = np.array([4., 2., 1.5])
    positions = mi.Point3f([2., 0.], [0., 1.], [0., 0.])

    for orientation in (mi.Point3f(0., 0., 0.),
                        mi.Point3f(dr.pi/4., 0., 0.),
                        mi.Point3f(0.3, -0.2, 0.1)):
        target = _cuboid_target(positions=positions,
                                position=mi.Point3f(10., -5., 3.),
                                orientation=orientation)
        viewer = _view(target, k_i=(0., 0., -1.), show_lobes=False)

        box = _box_segments(viewer).reshape(-1, 3)
        assert np.allclose(box.max(axis=0) - box.min(axis=0), dimensions,
                           atol=1e-4)
        # The box is centered on the origin of the local frame, where the
        # scattering points are given
        assert np.allclose(box.max(axis=0) + box.min(axis=0), 0., atol=1e-4)

        # The mesh is un-posed as well, and therefore fills the box
        mesh = _target_mesh_vertices(viewer)
        assert np.allclose(mesh.min(axis=0), box.min(axis=0), atol=1e-4)
        assert np.allclose(mesh.max(axis=0), box.max(axis=0), atol=1e-4)

        # Every scattering point lies inside the box
        points = np.transpose(positions.numpy())
        assert np.all(points >= box.min(axis=0) - 1e-4)
        assert np.all(points <= box.max(axis=0) + 1e-4)


def test_viewer_scales_the_scattering_points_with_their_target():
    # The geometry is shown with the scaling of the target baked in, so the
    # scattering points have to be scaled by it as well to keep their place
    # against the box
    dimensions = np.array([4., 2., 1.5])
    scaling = np.array([0.3, 0.5, 2.])
    positions = mi.Point3f([2., 0.], [0., 1.], [0., 0.])

    target = _cuboid_target(positions=positions)
    target.scaling = mi.Vector3f(*scaling.tolist())
    viewer = _view(target, k_i=(0., 0., -1.), show_lobes=False)

    box = _box_segments(viewer).reshape(-1, 3)
    assert np.allclose(box.max(axis=0) - box.min(axis=0), scaling*dimensions,
                       atol=1e-4)

    markers = [obj for obj, _ in viewer._objects
               if isinstance(obj, p3s.Points)]
    assert len(markers) == 1
    drawn = np.array(markers[0].geometry.attributes["position"].array)

    # The points still sit on the faces of the box, as they do unscaled
    assert np.allclose(drawn, scaling*np.transpose(positions.numpy()),
                       atol=1e-4)
    assert np.all(drawn >= box.min(axis=0) - 1e-4)
    assert np.all(drawn <= box.max(axis=0) + 1e-4)


def test_viewer_rotates_the_incident_direction_into_the_point_frame():
    # A scattering point whose local frame is rotated by 90 degrees about the
    # z-axis sees an incident direction rotated by the same amount, so the
    # surface of an RCS that only depends on the incident direction must come
    # out rotated as well
    def rcs(k_i, k_s, seed=0):
        # Cross-section of `max(k_i.x, 0.01)`, i.e. peaking when the incident
        # direction of propagation is along +x
        return dr.maximum(k_i.x, 0.01)

    # Illuminated along +x: the unrotated point is at its peak, the point
    # rotated by 90 degrees about z sees the direction along -y and is not
    aligned = _view(_cuboid_target(rcs=rcs), k_i=(1., 0., 0.), num_zenith=5,
                    num_azimuth=8)
    rotated = _view(_cuboid_target(rcs=rcs,
                                   orientations=mi.Point3f(dr.pi/2., 0., 0.)),
                    k_i=(1., 0., 0.), num_zenith=5, num_azimuth=8)

    # The peak of the surface is the top of the range the colormap covers
    peak_aligned = aligned.cross_section_range[1]
    peak_rotated = rotated.cross_section_range[1]

    assert peak_aligned > peak_rotated
    # The unrotated point sees `k_i.x = 1` and hence a cross-section of 1
    assert np.isclose(peak_aligned, 0., atol=1e-3)
    # The rotated one sees `k_i.x = 0`, which the callable floors at 0.01
    assert np.isclose(peak_rotated, 10.*np.log10(0.01), atol=1e-3)


def test_viewer_incident_arrow_points_towards_the_scattering_point():
    # `k_i` is a direction of propagation, so the wave travels along it and
    # the arrow must therefore approach the point from the opposite side
    k_i = np.array([0., 0., -1.])
    viewer = _view(_cuboid_target(rcs=ConstantRCS(sigma=1.)),
                   k_i=k_i.tolist(), show_box=False, show_orientations=False)

    # The incident direction is the only set of line segments left
    segments = _box_segments(viewer).reshape(-1, 3)
    start, end = segments[0], segments[1]

    direction = end - start
    assert np.allclose(direction/np.linalg.norm(direction), k_i, atol=1e-5)
    # The arrow stops on the surface of the point, which sits at the origin
    assert np.linalg.norm(end) < np.linalg.norm(start)


@pytest.mark.parametrize("theta,phi", [(np.pi/4, np.pi/4), (0., 0.),
                                       (np.pi/2, 0.), (np.pi/3, 5*np.pi/4)])
def test_viewer_camera_stays_away_from_the_incident_direction(theta, phi):
    # The incident arrows collapse to points when seen along their axis, which
    # the default camera of `Viewer` would do for a wave from (pi/4, pi/4)
    source = np.array(r_hat(theta, phi).numpy()).reshape(3)
    viewer = _view(_cuboid_target(), k_i=(-source).tolist())

    # The camera is placed relative to the center of the displayed objects
    camera = (np.array(viewer._camera.position)
              - np.array(viewer._bbox.center()))
    camera /= np.linalg.norm(camera)
    angle = np.degrees(np.arccos(np.clip(abs(camera @ source), 0., 1.)))
    assert angle >= 60. - 1e-6
    # The camera looks at the illuminated side of the target
    assert camera @ source > 0.


def test_viewer_rejects_invalid_arguments():
    target = _cuboid_target()

    with pytest.raises(ValueError, match="zero vector"):
        target.show(k_i=(0., 0., 0.))
    with pytest.raises(ValueError, match="single direction"):
        target.show(k_i=mi.Vector3f([0., 1.], [0., 0.], [1., 0.]))
    with pytest.raises(ValueError, match="num_zenith"):
        target.show(k_i=(0., 0., -1.), num_zenith=1)
    with pytest.raises(ValueError, match="num_azimuth"):
        target.show(k_i=(0., 0., -1.), num_azimuth=2)
    with pytest.raises(ValueError, match="dynamic_range"):
        target.show(k_i=(0., 0., -1.), dynamic_range=0.)
    with pytest.raises(ValueError, match="pattern_scale"):
        target.show(k_i=(0., 0., -1.), pattern_scale=-1.)
    with pytest.raises(ValueError, match="normalization"):
        target.show(k_i=(0., 0., -1.), normalization="none")
    with pytest.raises(ValueError, match="radial_scale"):
        target.show(k_i=(0., 0., -1.), radial_scale="log")


def test_show_displays_the_viewer(monkeypatch):
    # `show()` displays the viewer itself, like `Scene.preview()`, so that it
    # appears even when not called as the last expression of a cell
    displayed = []
    monkeypatch.setattr(sionna.rt.viewer, "display", displayed.append)

    assert _cuboid_target().show(k_i=(0., 0., -1.)) is None

    assert len(displayed) == 1
    assert isinstance(displayed[0], widgets.HBox)
    assert isinstance(displayed[0].children[0], p3s.Renderer)


def test_viewer_renders_in_a_notebook():
    viewer = _view(_cuboid_target(), k_i=(0., 0., -1.))

    bundle = viewer._repr_mimebundle_()
    assert "text/html" in bundle

    assert isinstance(viewer, ScatteringModelViewer)
    assert viewer.resolution == (655, 500)
