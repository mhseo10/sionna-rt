#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import copy

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from sionna.rt import (InteractionType, INVALID_PRIMITIVE, INVALID_SHAPE,
                       PathSolver, Paths, PlanarArray, Receiver, Transmitter,
                       load_scene)


def _tensor(tensor_type, array_type, values, shape):
    values = np.asarray(values).reshape(-1)
    return dr.reshape(tensor_type, array_type(values), shape)


def _make_paths(values, depth, synthetic_array, template=None):
    """Builds a small, fully materialized Paths object with known values."""
    values = np.asarray(values)
    num_paths = len(values)
    link_shape = [1, 1, num_paths] if synthetic_array \
        else [1, 1, 1, 1, num_paths]
    coefficient_shape = [1, 1, 1, 1, num_paths]

    paths = copy.copy(template) if template is not None else object.__new__(Paths)
    if template is None:
        paths._src_positions = mi.Point3f(0., 0., 0.)
        paths._tgt_positions = mi.Point3f(1., 0., 0.)
        paths._tx_velocities = mi.Vector3f(0., 0., 0.)
        paths._rx_velocities = mi.Vector3f(0., 0., 0.)
        paths._tx_array = object()
        paths._rx_array = object()
        paths._num_tx = 1
        paths._num_rx = 1
        paths._synthetic_array = synthetic_array
        paths._wavelength = mi.Float(0.1)
        paths._frequency = mi.Float(3.e9)

    paths._valid = _tensor(mi.TensorXb, mi.Bool,
                           np.ones(num_paths, dtype=bool), link_shape)
    for offset, name in enumerate((
        "_a_real",
        "_a_imag",
    )):
        setattr(paths, name,
                _tensor(mi.TensorXf, mi.Float, values + offset,
                        coefficient_shape))
    for offset, name in enumerate((
        "_tau",
        "_theta_t",
        "_phi_t",
        "_theta_r",
        "_phi_r",
        "_doppler",
    )):
        setattr(paths, name,
                _tensor(mi.TensorXf, mi.Float, values + 2 + offset,
                        link_shape))

    component_shape = [depth] + link_shape
    component_values = np.broadcast_to(
        values.reshape([1] + [1] * (len(link_shape) - 1) + [num_paths]),
        component_shape,
    ).copy()
    paths._interactions = _tensor(
        mi.TensorXu, mi.UInt, component_values, component_shape)
    paths._shapes = _tensor(
        mi.TensorXu, mi.UInt, component_values + 10, component_shape)
    paths._primitives = _tensor(
        mi.TensorXu, mi.UInt, component_values + 20, component_shape)
    vertices = np.stack((component_values,
                         component_values + 30,
                         component_values + 60), axis=-1)
    paths._vertices = _tensor(
        mi.TensorXf, mi.Float, vertices, component_shape + [3])
    paths._paths_components_built = True
    paths._max_num_paths = num_paths
    paths._paths_buffer = None
    return paths


@pytest.mark.parametrize("synthetic_array", [True, False])
def test_concat_appends_all_path_tensors_in_order(synthetic_array):
    a = _make_paths([1, 2], 1, synthetic_array)
    b = _make_paths([3], 1, synthetic_array, a)
    c = _make_paths([4, 5], 1, synthetic_array, a)

    concatenated = a.concat([b, c])

    assert concatenated.valid.shape[-1] == 5
    assert np.array_equal(concatenated.valid.numpy().reshape(-1),
                          np.ones(5, dtype=bool))
    expected = np.arange(1, 6)
    for offset, tensor in enumerate((
        *concatenated.a,
        concatenated.tau,
        concatenated.theta_t,
        concatenated.phi_t,
        concatenated.theta_r,
        concatenated.phi_r,
        concatenated.doppler,
    )):
        assert np.array_equal(tensor.numpy().reshape(-1),
                              expected + offset)

    assert np.array_equal(
        concatenated.interactions.numpy().reshape(-1), expected)
    assert np.array_equal(
        concatenated.objects.numpy().reshape(-1), expected + 10)
    assert np.array_equal(
        concatenated.primitives.numpy().reshape(-1), expected + 20)
    assert np.array_equal(
        concatenated.vertices.numpy().reshape(-1, 3),
        np.stack((expected, expected + 30, expected + 60), axis=-1))


@pytest.mark.parametrize("synthetic_array", [True, False])
def test_concat_pads_components_to_largest_depth(synthetic_array):
    a = _make_paths([1, 2], 1, synthetic_array)
    b = _make_paths([3], 3, synthetic_array, a)

    concatenated = a.concat([b])
    path_axis = -1

    assert concatenated.interactions.shape[0] == 3
    interactions = concatenated.interactions.numpy()
    shapes = concatenated.objects.numpy()
    primitives = concatenated.primitives.numpy()
    vertices = concatenated.vertices.numpy()

    assert np.all(interactions[1:, ..., :2] == InteractionType.NONE)
    assert np.all(shapes[1:, ..., :2] == INVALID_SHAPE)
    assert np.all(primitives[1:, ..., :2] == INVALID_PRIMITIVE)
    assert np.all(vertices[1:, ..., :2, :] == 0.)
    assert np.all(interactions[..., -1] == 3)
    assert concatenated.interactions.shape[path_axis] == 3
    assert concatenated.vertices.shape[-2] == 3


def test_concat_leaves_lazy_operand_unchanged_and_result_is_independent():
    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
    scene.add(Transmitter("tx", [0., 0., 0.]))
    scene.add(Receiver("rx", [1., 0., 0.]))
    paths = PathSolver()(
        scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
    )
    assert not paths._paths_components_built
    tau_before = paths.tau.numpy().copy()

    concatenated = paths.concat([])

    assert not paths._paths_components_built
    assert concatenated._paths_components_built
    assert concatenated._paths_buffer is None
    concatenated._tau += 1.
    assert np.array_equal(paths.tau.numpy(), tau_before)

    a, tau = concatenated.cir(normalize_delays=False, out_type="numpy")
    assert a.shape[-2] == tau.shape[-1]


def test_concat_handles_empty_paths_and_matches_repeated_concat():
    a = _make_paths([1, 2], 1, True)
    empty = _make_paths([], 3, True, a)
    b = _make_paths([3], 2, True, a)

    at_once = a.concat([empty, b])
    repeated = a.concat([empty]).concat([b])

    assert at_once.tau.shape[-1] == 3
    assert at_once.interactions.shape[0] == 3
    for name in (
        "valid", "tau", "theta_t", "phi_t", "theta_r", "phi_r",
        "doppler", "interactions", "objects", "primitives", "vertices",
    ):
        assert np.array_equal(getattr(at_once, name).numpy(),
                              getattr(repeated, name).numpy())
    for actual, expected in zip(at_once.a, repeated.a):
        assert np.array_equal(actual.numpy(), expected.numpy())


def test_concat_rejects_invalid_or_incompatible_inputs():
    a = _make_paths([1], 1, True)
    b = _make_paths([2], 1, True, a)

    with pytest.raises(ValueError, match="list of Paths"):
        a.concat([b, "not-paths"])

    incompatible = _make_paths([2], 1, True, a)
    incompatible._frequency = mi.Float(1.)
    with pytest.raises(ValueError, match="frequency"):
        a.concat([incompatible])

    incompatible = _make_paths([2], 1, False)
    with pytest.raises(ValueError, match="radio and antenna"):
        a.concat([incompatible])
