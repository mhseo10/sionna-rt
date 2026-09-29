#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np
import drjit as dr
import mitsuba as mi
import sionna
from sionna.rt import load_scene, SceneObject, Transmitter, ITURadioMaterial


def make_sphere(name, **kwargs):
    """Builds a detached scene object shaped as a sphere"""
    material = ITURadioMaterial(name=f"{name}-material", itu_type="metal",
                                thickness=0.01)
    return SceneObject(fname=sionna.rt.scene.sphere, name=name,
                       radio_material=material, **kwargs)


def test_scene_object_clone():
    scene = load_scene(sionna.rt.scene.box_one_screen,merge_shapes=False)

    screen = scene.objects["screen"]
    screen_clone = screen.clone(name="my-screen-clone")

    assert isinstance(screen_clone, SceneObject)

    # Check the name is correctly set
    assert screen_clone.name == "my-screen-clone"

    # The object should not be in the scene
    assert screen_clone.name not in scene.objects

    # Same radio material
    assert screen_clone.radio_material is screen.radio_material

    # Identical but not shared geometry
    assert dr.all(screen_clone.mi_mesh.faces_buffer() == screen.mi_mesh.faces_buffer())
    assert dr.all(screen_clone.mi_mesh.vertex_positions_buffer() == screen.mi_mesh.vertex_positions_buffer())
    assert screen_clone.mi_mesh.faces_buffer() is not screen.mi_mesh.faces_buffer()
    assert screen_clone.mi_mesh.vertex_positions_buffer() is not screen.mi_mesh.vertex_positions_buffer()

    ##################
    # Test is_mesh
    ##################

    screen = scene.objects["screen"]
    screen_clone = screen.clone(name="my-screen-clone", as_mesh=True)

    assert isinstance(screen_clone, mi.Mesh)

    # Check the name is correctly set
    assert screen_clone.id() == "my-screen-clone"

    # Same radio material
    assert screen_clone.bsdf() is screen.radio_material

    # Identical but not shared geometry
    assert dr.all(screen_clone.faces_buffer() == screen.mi_mesh.faces_buffer())
    assert dr.all(screen_clone.vertex_positions_buffer() == screen.mi_mesh.vertex_positions_buffer())
    assert screen_clone.faces_buffer() is not screen.mi_mesh.faces_buffer()
    assert screen_clone.vertex_positions_buffer() is not screen.mi_mesh.vertex_positions_buffer()


def test_scene_object_look_at_string_and_instance_targets():
    scene = load_scene(sionna.rt.scene.box, merge_shapes=False)
    obj = next(iter(scene.objects.values()))
    # Place the TX far along +x from the object center so the look-at angles
    # are unambiguous.
    obj_pos = np.array(obj.position.numpy()[:, 0])
    tx_pos = obj_pos + np.array([10.0, 0.0, 0.0])
    tx = Transmitter("tx-look", position=tx_pos.tolist())
    scene.add(tx)

    obj.look_at("tx-look")
    orient_named = obj.orientation.numpy()[:, 0].copy()
    obj.look_at(tx)
    orient_inst = obj.orientation.numpy()[:, 0]
    assert np.allclose(orient_named, orient_inst, atol=1e-5)
    # Looking along +x → (α, β, γ) = (0, 0, 0)
    assert np.allclose(orient_inst, [0.0, 0.0, 0.0], atol=1e-4)

    other = obj.clone(name="other-box")
    scene.edit(add=other)
    other.position = mi.Point3f(float(obj_pos[0]),
                                float(obj_pos[1] + 5.0),
                                float(obj_pos[2]))
    obj.look_at("other-box")
    # Looking along +y → α = π/2
    assert np.allclose(obj.orientation.numpy()[:, 0],
                       [np.pi / 2, 0.0, 0.0], atol=1e-4)


def test_scene_object_look_at_string_error_paths():
    scene = load_scene(sionna.rt.scene.box, merge_shapes=False)
    obj = next(iter(scene.objects.values()))

    with pytest.raises(ValueError, match="Unknown target"):
        obj.look_at("does-not-exist")

    mat_name = next(iter(scene.radio_materials))
    with pytest.raises(ValueError, match="Cannot look at"):
        obj.look_at(mat_name)

    detached = obj.clone(name="detached")
    with pytest.raises(ValueError, match="Scene is not set"):
        detached.look_at("tx")


def test_scene_object_transform_before_being_added():
    """Position, orientation, and scaling can be set while detached, and are
    preserved when the object is added to a scene."""
    obj = make_sphere("sphere")

    extents = lambda o: (np.array(o.mi_mesh.bbox().max)
                         - np.array(o.mi_mesh.bbox().min))
    original_extents = extents(obj)

    obj.position = mi.Point3f(10.0, 2.0, 5.0)
    assert np.allclose(obj.position.numpy()[:, 0], [10.0, 2.0, 5.0], atol=1e-5)

    obj.orientation = mi.Point3f(np.pi / 2, 0.0, 0.0)
    assert np.allclose(obj.orientation.numpy()[:, 0], [np.pi / 2, 0.0, 0.0])
    # A rotation about the center leaves the position unchanged
    assert np.allclose(obj.position.numpy()[:, 0], [10.0, 2.0, 5.0], atol=1e-5)

    obj.scaling = 2.0
    assert np.allclose(extents(obj), 2.0 * original_extents, rtol=1e-5)
    assert np.allclose(obj.position.numpy()[:, 0], [10.0, 2.0, 5.0], atol=1e-5)

    # The geometry carries over unchanged when the object joins the scene
    vertices = obj.mi_mesh.vertex_positions_buffer().numpy().copy()
    scene = load_scene(sionna.rt.scene.box, merge_shapes=False)
    scene.add(obj)
    assert scene.get("sphere") is obj
    assert np.allclose(obj.mi_mesh.vertex_positions_buffer().numpy(), vertices)
    assert np.allclose(obj.position.numpy()[:, 0], [10.0, 2.0, 5.0], atol=1e-5)
    assert np.allclose(obj.orientation.numpy()[:, 0], [np.pi / 2, 0.0, 0.0])

    # Setting the position still works once the object is part of the scene
    obj.position = mi.Point3f(0.0, 0.0, 0.0)
    assert np.allclose(obj.position.numpy()[:, 0], [0.0, 0.0, 0.0], atol=1e-5)


def test_scene_object_pose_constructor_arguments():
    obj = make_sphere("sphere",
                      position=mi.Point3f(1.0, 2.0, 3.0),
                      orientation=mi.Point3f(0.25, 0.5, 0.75),
                      velocity=mi.Vector3f(4.0, 5.0, 6.0))
    assert np.allclose(obj.position.numpy()[:, 0], [1.0, 2.0, 3.0], atol=1e-5)
    assert np.allclose(obj.orientation.numpy()[:, 0], [0.25, 0.5, 0.75])
    assert np.allclose(obj.velocity.numpy()[:, 0], [4.0, 5.0, 6.0])

    # `look_at` is applied relative to the position, which is set first
    looking = make_sphere("looking",
                          position=mi.Point3f(1.0, 0.0, 0.0),
                          look_at=mi.Point3f(1.0, 10.0, 0.0))
    # Looking along +y → α = π/2
    assert np.allclose(looking.orientation.numpy()[:, 0],
                       [np.pi / 2, 0.0, 0.0], atol=1e-5)

    # A radio device or another scene object can be looked at as well, and
    # doing so at construction matches a later call to `look_at()`
    tx = Transmitter("tx", position=[7.0, 3.0, 10.0])
    at_device = make_sphere("at-device", look_at=tx)
    expected = make_sphere("expected")
    expected.look_at(tx)
    assert np.allclose(at_device.orientation.numpy(),
                       expected.orientation.numpy(), atol=1e-5)

    at_object = make_sphere("at-object", position=mi.Point3f(-5.0, 0.0, 0.0),
                            look_at=expected)
    # `expected` sits at the origin, so the look-at direction is +x
    assert np.allclose(at_object.orientation.numpy()[:, 0],
                       [0.0, 0.0, 0.0], atol=1e-4)


def test_scene_object_pose_constructor_errors():
    with pytest.raises(ValueError, match="Only one of `orientation` or"):
        make_sphere("sphere",
                    orientation=mi.Point3f(0.0),
                    look_at=mi.Point3f(1.0, 0.0, 0.0))

    # A named target cannot be resolved before the object belongs to a scene
    with pytest.raises(ValueError, match="only be specified by name once"):
        make_sphere("sphere", look_at="tx")


def test_scene_object_clone_preserves_pose():
    obj = make_sphere("sphere",
                      position=mi.Point3f(3.0, 4.0, 5.0),
                      orientation=mi.Point3f(0.3, 0.0, 0.0),
                      velocity=mi.Vector3f(1.0, 0.0, 0.0))
    obj.scaling = mi.Vector3f(2.0, 3.0, 4.0)

    clone = obj.clone(name="sphere-clone")
    assert np.allclose(clone.position.numpy(), obj.position.numpy(), atol=1e-5)
    assert np.allclose(clone.orientation.numpy(), obj.orientation.numpy())
    assert np.allclose(clone.scaling.numpy(), obj.scaling.numpy())
    assert np.allclose(clone.velocity.numpy(), obj.velocity.numpy())

    # Because the pose was carried over, re-applying it to the clone is a no-op
    vertices = clone.mi_mesh.vertex_positions_buffer().numpy().copy()
    clone.orientation = obj.orientation
    clone.scaling = obj.scaling
    assert np.allclose(clone.mi_mesh.vertex_positions_buffer().numpy(),
                       vertices, atol=1e-5)


def test_scene_object_velocity_width_check():
    """Velocity validation uses raise (not assert), so it survives python -O."""
    scene = load_scene(sionna.rt.scene.box, merge_shapes=False)
    obj = next(iter(scene.objects.values()))

    obj.velocity = mi.Vector3f(1.0, 2.0, 3.0)
    assert np.allclose(obj.velocity.numpy()[:, 0], [1.0, 2.0, 3.0])

    wide = mi.Vector3f([1.0, 4.0], [2.0, 5.0], [3.0, 6.0])
    assert dr.width(wide) == 2
    with pytest.raises(ValueError, match="Only a single velocity vector"):
        obj.velocity = wide
