#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the interactive scene preview widget."""

import drjit as dr
import pytest

import pythreejs as p3s

from sionna import rt
from sionna.rt import PathSolver
from sionna.rt.rcs import ScatteringModel, SensingTarget
from sionna.rt.radio_materials.itu import itu_material
from sionna.rt.preview import Previewer
from sionna.rt.scene import Scene, load_scene


def add_example_radio_devices(scene: Scene):
    # Note: hardcoded for `box_two_screens.xml` as an example.
    scene.add(rt.Transmitter("tr-1", position=[-3.0, 0.0, 1.5]))
    scene.add(rt.Receiver("rc-1", position=[3.0, 0.0, 1.5]))
    scene.add(
        rt.Receiver(
            "rc-2", position=[1.0, -2.0, 3.5], color=(0.9, 0.9, 0.2),
            display_radius=0.9
        )
    )

    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="tr38901", polarization="VH"
    )
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="tr38901", polarization="VH"
    )


def get_example_paths(scene: Scene):
    # Ray tracing parameters
    num_samples_per_src = int(1e6)
    max_num_paths = int(1e7)
    max_depth = 3

    solver = PathSolver()
    paths = solver(
        scene,
        max_depth=max_depth,
        max_num_paths_per_src=max_num_paths,
        samples_per_src=num_samples_per_src,
    )

    return paths


@pytest.mark.parametrize("has_paths", (True, False))
def test01_preview_basic(has_paths):
    scene = load_scene(rt.scene.box_two_screens)

    eta_r, sigma = itu_material("metal", 3e9)  # ITU material evaluated at 3GHz
    for sh in scene.mi_scene.shapes():
        material = sh.bsdf()
        material.relative_permittivity = eta_r
        material.conductivity = sigma
        material.scattering_coefficient = 0.01
        material.xpd_coefficient = 0.2

    add_example_radio_devices(scene)
    paths = get_example_paths(scene)

    if not has_paths:
        paths._valid &= False
        assert dr.count(paths.valid) == 0

    # Should work with or without valid paths.
    # Note: we don't verify that the preview widget is actually functional,
    # simply that no exception is thrown.
    scene.preview(paths=paths)


def _scene_materials(scene: Scene):
    """Materials of the meshes that the previewer draws for `scene`."""
    previewer = Previewer(scene)
    return [obj.material for obj, _ in previewer._objects
            if isinstance(obj, p3s.Mesh)]


def test_preview_draws_sensing_targets_as_translucent():
    scene = load_scene(rt.scene.box_two_screens)
    scene.edit(add=SensingTarget("st", scattering_model=ScatteringModel(),
                                 length=1., width=1., height=1.,
                                 display_opacity=0.25))

    materials = _scene_materials(scene)
    opaque = [m for m in materials if not m.transparent]
    translucent = [m for m in materials if m.transparent]

    # One draw call for the opaque scene geometry, one for the target
    assert len(opaque) == 1
    assert len(translucent) == 1
    assert translucent[0].opacity == 0.25
    # Translucent geometry must not hide the paths and devices behind it
    assert not translucent[0].depthWrite
    assert opaque[0].depthWrite


def test_preview_groups_sensing_targets_sharing_an_opacity():
    scene = load_scene(rt.scene.box_two_screens)
    for name, opacity in (("st-1", 0.25), ("st-2", 0.25), ("st-3", 0.75)):
        scene.edit(add=SensingTarget(name,
                                     scattering_model=ScatteringModel(),
                                     length=1., width=1., height=1.,
                                     display_opacity=opacity))

    # The two targets sharing an opacity are drawn together
    opacities = sorted(m.opacity for m in _scene_materials(scene))
    assert opacities == [0.25, 0.75, 1.]


def test_preview_skips_fully_transparent_sensing_targets():
    scene = load_scene(rt.scene.box_two_screens)
    scene.edit(add=SensingTarget("st", scattering_model=ScatteringModel(),
                                 length=1., width=1., height=1.,
                                 display_opacity=0.))

    assert [m.opacity for m in _scene_materials(scene)] == [1.]


def test_preview_with_sensing_target_and_paths():
    scene = load_scene(rt.scene.box_two_screens)
    add_example_radio_devices(scene)
    scene.edit(add=SensingTarget("st", scattering_model=ScatteringModel(),
                                 length=1., width=1., height=1.))

    scene.preview(paths=get_example_paths(scene))


def test_preview_empty_paths_object():
    """Preview accepts a Paths object that contains zero paths."""
    scene = load_scene(rt.scene.box_two_screens)
    add_example_radio_devices(scene)
    paths = PathSolver()(
        scene,
        los=False,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
    )
    assert paths.vertices.shape[-2] == 0
    scene.preview(paths=paths)
