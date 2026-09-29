#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the paths buffer"""

import numpy as np
import pytest
import drjit as dr
import mitsuba as mi

import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, \
                      PathSolver
from sionna.rt.constants import InteractionType, INVALID_SHAPE, \
                                INVALID_PRIMITIVE
from sionna.rt.path_solvers import SBCandidateGenerator
from sionna.rt.path_solvers.paths_buffer import PathsBuffer

############################################################
# Utilities
############################################################

def solve_paths_buffer(max_depth, diffraction):
    """
    Runs the path solver on a small scene and returns the resulting paths buffer

    The returned buffer stores channel coefficients, delays, and Doppler
    shifts, as well as the diffracting wedges if ``diffraction`` is set to
    `True`.
    """
    scene = load_scene(sionna.rt.scene.simple_reflector)
    scene.add(Transmitter("tx-1", [-10, 0, 10]))
    scene.add(Transmitter("tx-2", [-10, 1, 10]))
    scene.add(Receiver("rx-1", [10, 0, 10]))
    scene.add(Receiver("rx-2", [10, -1, 10]))
    scene.tx_array = PlanarArray(num_cols=1, num_rows=1, pattern="iso",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_cols=1, num_rows=1, pattern="iso",
                                 polarization="VH")
    scene.get("rx-2").velocity = [-5, 8, 6]
    p_solver = PathSolver()
    paths = p_solver(scene,
                     max_depth=max_depth,
                     los=True,
                     specular_reflection=True,
                     diffuse_reflection=False,
                     refraction=False,
                     diffraction=diffraction,
                     edge_diffraction=diffraction)
    # pylint: disable=protected-access
    return paths._paths_buffer

def solve_deep_paths_buffer():
    """
    Runs the path solver on a scene that leads to paths with up to three
    interactions, including diffractions, and returns the scene together with
    the resulting paths buffer
    """
    scene = load_scene(sionna.rt.scene.box_one_screen)
    scene.add(Transmitter("tx-1", [-3, -1, 1]))
    scene.add(Transmitter("tx-2", [-3, 1, 1]))
    scene.add(Receiver("rx-1", [3, 1, 1.5]))
    scene.add(Receiver("rx-2", [3, -1, 1.5]))
    scene.tx_array = PlanarArray(num_cols=1, num_rows=1, pattern="iso",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_cols=1, num_rows=1, pattern="iso",
                                 polarization="VH")
    p_solver = PathSolver()
    paths = p_solver(scene,
                     max_depth=3,
                     los=True,
                     specular_reflection=True,
                     diffuse_reflection=False,
                     refraction=False,
                     diffraction=True,
                     edge_diffraction=True)
    # pylint: disable=protected-access
    return scene, paths._paths_buffer

def generate_candidate_paths_buffer():
    """
    Runs the candidate generator on a small scene and returns the resulting
    paths buffer

    In contrast to :func:`solve_paths_buffer`, the returned buffer stores
    neither channel coefficients, delays, nor Doppler shifts.
    """
    scene = load_scene(sionna.rt.scene.box, merge_shapes=False)
    sb = SBCandidateGenerator()
    return sb(mi_scene=scene.mi_scene,
              src_positions=mi.Point3f([0.5, -0.5], [0.5, -0.5], [0.5, 0.5]),
              tgt_positions=mi.Point3f([-0.5], [0.5], [1.0]),
              samples_per_src=1000,
              max_num_paths_per_src=100,
              max_depth=3,
              los=True,
              specular_reflection=True,
              diffuse_reflection=False,
              refraction=False,
              diffraction=False,
              edge_diffraction=False,
              seed=1)

def sensing_paths_buffer(num_paths=8):
    """
    Builds a paths buffer storing a single sensing interaction per path

    The sensing target and sensing point indices are stored in the shapes and
    primitives fields, respectively.
    """
    buffer = PathsBuffer(num_paths, 1, diffraction=False)
    active = dr.full(mi.Bool, True, num_paths)
    depth = dr.ones(mi.UInt, num_paths)
    buffer.valid = active
    buffer.set_interaction_type(
        depth, dr.full(mi.UInt, InteractionType.SENSING, num_paths), active)
    buffer.set_shape(depth, mi.UInt(np.arange(num_paths) % 3), active)
    buffer.set_primitive(depth, mi.UInt(np.arange(num_paths)*2), active)
    buffer.advance_paths_counter(num_paths)
    return buffer

def vec_to_numpy(v):
    """Converts a 3D vector to a numpy array of shape `[n, 3]`"""
    return np.stack([v.x.numpy(), v.y.numpy(), v.z.numpy()], axis=-1)

def wedges_to_numpy(wedges):
    """Converts the fields of a ``WedgeGeometry`` to numpy arrays"""
    return {
        "shape" : dr.reinterpret_array(mi.UInt, wedges.shape).numpy(),
        "prim0" : wedges.prim0.numpy(),
        "primn" : wedges.primn.numpy(),
        "local_edge" : wedges.local_edge.numpy(),
        "o" : vec_to_numpy(wedges.o),
        "e_hat" : vec_to_numpy(wedges.e_hat),
        "length" : wedges.length.numpy(),
        "n0" : vec_to_numpy(wedges.n0),
        "nn" : vec_to_numpy(wedges.nn),
    }

def buffer_to_numpy(buffer):
    """
    Converts the content of a paths buffer to a dictionary of numpy arrays
    indexed by the paths along their first dimension
    """
    fields = {
        "valid" : buffer.valid.numpy(),
        "source_indices" : buffer.source_indices.numpy(),
        "target_indices" : buffer.target_indices.numpy(),
        "k_tx" : vec_to_numpy(buffer.k_tx),
        "k_rx" : vec_to_numpy(buffer.k_rx),
        "interaction_types" : buffer.interaction_types.numpy(),
        "vertices_x" : buffer.vertices_x.numpy(),
        "vertices_y" : buffer.vertices_y.numpy(),
        "vertices_z" : buffer.vertices_z.numpy(),
        "shapes" : buffer.shapes.numpy(),
        "primitives" : buffer.primitives.numpy(),
        "probs" : buffer.probs.numpy(),
    }
    if buffer.tau is not None:
        fields["tau"] = buffer.tau.numpy()
    if buffer.doppler is not None:
        fields["doppler"] = buffer.doppler.numpy()
    if buffer.a is not None:
        for n, a_ in enumerate(buffer.a):
            for m, a in enumerate(a_):
                fields[f"a_{n}_{m}"] = dr.real(a).numpy() \
                                        + 1j*dr.imag(a).numpy()
    if buffer.diffracting_wedges is not None:
        for name, value in wedges_to_numpy(buffer.diffracting_wedges).items():
            fields["wedge_" + name] = value
    return fields

def arrays_equal(a, b):
    """
    Compares two arrays, considering `NaN` values as equal

    Fields that are not set for every path, such as the diffracting wedges of
    paths without diffraction, can store `NaN` values.
    """
    return np.array_equal(a, b, equal_nan=np.issubdtype(a.dtype, np.inexact))

def assert_gathered(buffer, gathered_buffer, indices):
    """
    Asserts that ``gathered_buffer`` stores the paths of ``buffer`` with
    indices ``indices``
    """
    num_paths = indices.size

    assert gathered_buffer.buffer_size == num_paths
    assert dr.all(gathered_buffer.paths_counter == num_paths)
    assert gathered_buffer.max_depth == buffer.max_depth
    assert gathered_buffer.depth_dim_size == buffer.depth_dim_size

    reference = buffer_to_numpy(buffer)
    gathered = buffer_to_numpy(gathered_buffer)

    # The same set of fields should be set in both buffers
    assert reference.keys() == gathered.keys()

    for name, value in gathered.items():
        assert value.shape[0] == num_paths, name
        assert arrays_equal(value, reference[name][indices]), name

# Fields storing data for every interaction of the paths
DEPTH_FIELDS = ("interaction_types", "vertices_x", "vertices_y", "vertices_z",
                "shapes", "primitives", "probs")

# Value stored by the fields of DEPTH_FIELDS for items that do not correspond
# to an interaction
DEPTH_FIELD_DEFAULTS = {"interaction_types" : InteractionType.NONE,
                        "shapes" : INVALID_SHAPE,
                        "primitives" : INVALID_PRIMITIVE,
                        "vertices_x" : 0., "vertices_y" : 0.,
                        "vertices_z" : 0., "probs" : 0.}

def path_depths(buffer):
    """Number of interactions of every path of ``buffer``"""
    interaction_types = buffer.interaction_types.numpy()
    return np.sum(interaction_types != InteractionType.NONE, axis=-1)

def reverse_reference(fields, depths):
    """
    Computes the expected content of a reversed paths buffer from the content
    ``fields`` of the buffer before reversal
    """
    reference = dict(fields)
    # Sources and targets, as well as directions of departure and arrival, are
    # swapped
    for name, other in (("source_indices", "target_indices"),
                        ("k_tx", "k_rx")):
        reference[name] = fields[other]
        reference[other] = fields[name]
    # The interactions of every path are reversed
    for name in DEPTH_FIELDS:
        value = fields[name].copy()
        for i, depth in enumerate(depths):
            value[i, :depth] = value[i, :depth][::-1]
        reference[name] = value
    return reference

def chain_reference(fields, depths, depth_dim_size):
    """
    Computes the expected content of the buffer chaining the buffers whose
    contents are ``fields`` and whose path depths are ``depths``

    :param fields: For every chained buffer, its content as returned by
        :func:`buffer_to_numpy`
    :param depths: For every chained buffer, the number of interactions of
        every of its paths
    :param depth_dim_size: Size of the depth dimension of the chained buffer
    """
    reference = {}
    # The chained paths originate from the sources of the first buffer and
    # connect to the targets of the last one
    reference["valid"] = np.logical_and.reduce([f["valid"] for f in fields])
    reference["source_indices"] = fields[0]["source_indices"]
    reference["k_tx"] = fields[0]["k_tx"]
    reference["target_indices"] = fields[-1]["target_indices"]
    reference["k_rx"] = fields[-1]["k_rx"]
    # The interactions of the chained buffers are concatenated
    num_paths = depths[0].size
    for name in DEPTH_FIELDS:
        value = np.full((num_paths, depth_dim_size),
                        DEPTH_FIELD_DEFAULTS[name], fields[0][name].dtype)
        for i in range(num_paths):
            offset = 0
            for fields_, depths_ in zip(fields, depths):
                depth = depths_[i]
                value[i, offset:offset+depth] = fields_[name][i, :depth]
                offset += depth
        reference[name] = value
    return reference

def assert_chained(buffers, chained_buffer):
    """
    Asserts that ``chained_buffer`` stores the paths of ``buffers`` chained
    together
    """
    num_paths = buffers[0].buffer_size

    assert chained_buffer.buffer_size == num_paths
    assert dr.all(chained_buffer.paths_counter == num_paths)
    assert chained_buffer.max_depth == sum(b.max_depth for b in buffers)

    reference = chain_reference([buffer_to_numpy(b) for b in buffers],
                                [path_depths(b) for b in buffers],
                                chained_buffer.depth_dim_size)
    chained = buffer_to_numpy(chained_buffer)

    # The channel impulse response and the diffracting wedges are dropped
    assert reference.keys() == chained.keys()

    for name, value in chained.items():
        assert value.shape[0] == num_paths, name
        assert arrays_equal(value, reference[name]), name

def chainable_buffers(num_buffers, num_paths=8):
    """
    Builds ``num_buffers`` buffers storing ``num_paths`` paths each, with
    differing numbers of interactions per path
    """
    _, buffer = solve_deep_paths_buffer()
    assert buffer.buffer_size >= num_paths
    rng = np.random.default_rng(42)
    return [buffer.gather(mi.UInt(rng.permutation(buffer.buffer_size)
                                  [:num_paths]))
            for _ in range(num_buffers)]

def assert_directions_consistent(buffer, src_positions, tgt_positions):
    """
    Asserts that the directions of departure and arrival of the paths stored
    in ``buffer`` point from their source and target, respectively, towards the
    adjacent path vertex

    :param src_positions: Positions of the sources indexed by
        :attr:`~sionna.rt.PathsBuffer.source_indices`
    :param tgt_positions: Positions of the targets indexed by
        :attr:`~sionna.rt.PathsBuffer.target_indices`
    """
    fields = buffer_to_numpy(buffer)
    depths = path_depths(buffer)
    vertices = np.stack([fields["vertices_x"], fields["vertices_y"],
                         fields["vertices_z"]], axis=-1)

    sources = src_positions[fields["source_indices"]]
    targets = tgt_positions[fields["target_indices"]]
    has_interaction = depths[:, None] > 0
    # For line-of-sight paths, the vertex adjacent to the source is the target,
    # and vice-versa
    first = np.where(has_interaction, vertices[:, 0], targets)
    last = np.where(has_interaction,
                    vertices[np.arange(depths.size), np.maximum(depths-1, 0)],
                    sources)

    def unit_vec(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    k_t = fields["k_tx"]
    k_r = fields["k_rx"]
    assert np.allclose(k_t, unit_vec(first - sources), atol=1e-5)
    assert np.allclose(k_r, unit_vec(last - targets), atol=1e-5)

def endpoint_positions(scene):
    """Positions of the sources and targets of ``scene``"""
    return (scene.sources(True, False)[0].numpy().T,
            scene.targets(True, False)[0].numpy().T)

############################################################
# Unit tests
############################################################

@pytest.mark.parametrize("max_depth,diffraction", [(0, False), (3, False),
                                                   (3, True)])
def test01_gather_subset(max_depth, diffraction):
    """Gathering a subset of the paths in an arbitrary order"""
    buffer = solve_paths_buffer(max_depth, diffraction)
    num_paths = buffer.buffer_size
    assert num_paths > 1
    if diffraction:
        assert buffer.diffracting_wedges is not None

    rng = np.random.default_rng(42)
    indices = rng.permutation(num_paths)[:max(1, num_paths//2)]

    gathered_buffer = buffer.gather(mi.UInt(indices))

    assert_gathered(buffer, gathered_buffer, indices)

@pytest.mark.parametrize("max_depth,diffraction", [(0, False), (3, True)])
def test02_gather_all_paths(max_depth, diffraction):
    """Gathering all the paths in order returns a copy of the buffer"""
    buffer = solve_paths_buffer(max_depth, diffraction)
    indices = np.arange(buffer.buffer_size)

    gathered_buffer = buffer.gather(mi.UInt(indices))

    assert_gathered(buffer, gathered_buffer, indices)

def test04_gather_no_paths():
    """Gathering no path returns an empty buffer"""
    buffer = solve_paths_buffer(3, True)
    indices = np.array([], dtype=np.uint32)

    gathered_buffer = buffer.gather(mi.UInt(indices))

    assert_gathered(buffer, gathered_buffer, indices)

def test05_gather_leaves_source_buffer_unchanged():
    """Gathering does not modify the buffer it is applied to"""
    buffer = solve_paths_buffer(3, True)
    reference = buffer_to_numpy(buffer)
    buffer_size = buffer.buffer_size

    buffer.gather(mi.UInt([0, 1, 2]))

    assert buffer.buffer_size == buffer_size
    gathered = buffer_to_numpy(buffer)
    assert reference.keys() == gathered.keys()
    for name, value in gathered.items():
        assert arrays_equal(value, reference[name]), name

def test06_gather_candidate_paths():
    """Gathering from a buffer that only stores candidate paths"""
    buffer = generate_candidate_paths_buffer()
    assert buffer.a is None
    assert buffer.tau is None
    assert buffer.doppler is None
    assert buffer.diffracting_wedges is None

    indices = np.arange(0, 32, 3)
    gathered_buffer = buffer.gather(mi.UInt(indices))

    assert gathered_buffer.a is None
    assert gathered_buffer.tau is None
    assert gathered_buffer.doppler is None
    assert gathered_buffer.diffracting_wedges is None
    assert_gathered(buffer, gathered_buffer, indices)

def test07_reverse():
    """Reversing paths swaps their endpoints and reverses their interactions"""
    _, buffer = solve_deep_paths_buffer()
    depths = path_depths(buffer)
    # Ensure that paths with more than one interaction are tested
    assert np.max(depths) > 1
    fields = buffer_to_numpy(buffer)

    reversed_buffer = buffer.reverse()

    reference = reverse_reference(fields, depths)
    reversed_fields = buffer_to_numpy(reversed_buffer)
    assert reference.keys() == reversed_fields.keys()
    for name, value in reversed_fields.items():
        assert arrays_equal(value, reference[name]), name

def test08_reverse_twice():
    """Reversing paths twice restores the original buffer"""
    _, buffer = solve_deep_paths_buffer()
    reference = buffer_to_numpy(buffer)

    reversed_buffer = buffer.reverse().reverse()

    fields = buffer_to_numpy(reversed_buffer)
    for name, value in fields.items():
        assert arrays_equal(value, reference[name]), name

def test09_reverse_directions():
    """Directions of the reversed paths are consistent with their geometry"""
    scene, buffer = solve_deep_paths_buffer()
    src_positions, tgt_positions = endpoint_positions(scene)
    assert_directions_consistent(buffer, src_positions, tgt_positions)

    reversed_buffer = buffer.reverse()

    # The sources of the reversed paths are the targets of the original ones
    assert_directions_consistent(reversed_buffer, tgt_positions, src_positions)

def test10_reverse_los_paths():
    """Reversing line-of-sight paths only swaps endpoints and directions"""
    buffer = solve_paths_buffer(0, False)
    assert np.all(path_depths(buffer) == 0)
    fields = buffer_to_numpy(buffer)

    reversed_buffer = buffer.reverse()

    reversed_fields = buffer_to_numpy(reversed_buffer)
    assert np.array_equal(reversed_fields["source_indices"],
                          fields["target_indices"])
    assert np.array_equal(reversed_fields["target_indices"],
                          fields["source_indices"])
    assert np.array_equal(reversed_fields["k_tx"], fields["k_rx"])
    for name in DEPTH_FIELDS:
        assert arrays_equal(reversed_fields[name], fields[name]), name

def test11_reverse_empty_buffer():
    """Reversing an empty buffer is a no-op"""
    buffer = solve_paths_buffer(3, True).gather(dr.zeros(mi.UInt, 0))

    reversed_buffer = buffer.reverse()

    assert reversed_buffer.buffer_size == 0
    for name, value in buffer_to_numpy(reversed_buffer).items():
        assert value.shape[0] == 0, name

def test12_reverse_leaves_source_buffer_unchanged():
    """Reversing does not modify the buffer it is applied to"""
    _, buffer = solve_deep_paths_buffer()
    reference = buffer_to_numpy(buffer)
    buffer_size = buffer.buffer_size

    buffer.reverse()

    assert buffer.buffer_size == buffer_size
    fields = buffer_to_numpy(buffer)
    assert reference.keys() == fields.keys()
    for name, value in fields.items():
        assert arrays_equal(value, reference[name]), name

@pytest.mark.parametrize("count", [1, 3])
def test13_tile(count):
    """Tiling repeats the whole buffer ``count`` times"""
    buffer = solve_paths_buffer(3, True)
    indices = np.tile(np.arange(buffer.buffer_size), count)

    tiled_buffer = buffer.tile(count)

    assert_gathered(buffer, tiled_buffer, indices)

@pytest.mark.parametrize("count", [1, 3])
def test14_repeat(count):
    """Repeating repeats every path of the buffer ``count`` times"""
    buffer = solve_paths_buffer(3, True)
    indices = np.repeat(np.arange(buffer.buffer_size), count)

    repeated_buffer = buffer.repeat(count)

    assert_gathered(buffer, repeated_buffer, indices)

def test15_tile_repeat_candidate_paths():
    """Tiling and repeating a buffer that only stores candidate paths"""
    buffer = generate_candidate_paths_buffer()
    num_paths = buffer.buffer_size

    assert_gathered(buffer, buffer.tile(2), np.tile(np.arange(num_paths), 2))
    assert_gathered(buffer, buffer.repeat(2),
                    np.repeat(np.arange(num_paths), 2))

def test16_tile_repeat_empty_buffer():
    """Tiling and repeating an empty buffer returns an empty buffer"""
    buffer = solve_paths_buffer(3, True).gather(dr.zeros(mi.UInt, 0))
    indices = np.array([], dtype=np.uint32)

    assert_gathered(buffer, buffer.tile(3), indices)
    assert_gathered(buffer, buffer.repeat(3), indices)

@pytest.mark.parametrize("num_others", [1, 2])
def test17_chain(num_others):
    """Chaining paths concatenates their interactions along the depth dim"""
    buffers = chainable_buffers(num_others + 1)
    # Ensure that paths with more than one interaction are tested
    assert max(np.max(path_depths(b)) for b in buffers) > 1

    chained_buffer = buffers[0].chain(buffers[1:])

    assert_chained(buffers, chained_buffer)

def test18_chain_drops_fields():
    """
    Chaining paths drops their coefficients, delays, Doppler shifts, and wedges
    """
    buffers = chainable_buffers(2)
    for buffer in buffers:
        assert buffer.a is not None
        assert buffer.tau is not None
        assert buffer.doppler is not None
        assert buffer.diffracting_wedges is not None

    chained_buffer = buffers[0].chain(buffers[1:])

    assert chained_buffer.a is None
    assert chained_buffer.tau is None
    assert chained_buffer.doppler is None
    assert chained_buffer.diffracting_wedges is None

def test19_chain_no_others():
    """Chaining a buffer with no other buffer returns a copy of it"""
    buffer = chainable_buffers(1)[0]

    chained_buffer = buffer.chain([])

    assert chained_buffer.max_depth == buffer.max_depth
    assert_chained([buffer], chained_buffer)

@pytest.mark.parametrize("los_index", [0, 1, 2])
def test20_chain_los_paths(los_index):
    """Chaining line-of-sight paths only contributes no interaction"""
    num_paths = 4
    buffers = chainable_buffers(2, num_paths)
    los_buffer = solve_paths_buffer(0, False).gather(
        dr.arange(mi.UInt, num_paths))
    assert los_buffer.depth_dim_size == 1
    assert np.all(path_depths(los_buffer) == 0)
    buffers.insert(los_index, los_buffer)

    chained_buffer = buffers[0].chain(buffers[1:])

    assert_chained(buffers, chained_buffer)

def test21_chain_empty_buffers():
    """Chaining empty buffers returns an empty buffer"""
    buffers = [solve_paths_buffer(3, True).gather(dr.zeros(mi.UInt, 0))
               for _ in range(2)]

    chained_buffer = buffers[0].chain(buffers[1:])

    assert chained_buffer.buffer_size == 0
    for name, value in buffer_to_numpy(chained_buffer).items():
        assert value.shape[0] == 0, name

def test22_chain_size_mismatch():
    """Chaining buffers storing different numbers of paths raises an error"""
    buffer = chainable_buffers(1)[0]
    smaller_buffer = buffer.gather(dr.arange(mi.UInt, buffer.buffer_size - 1))

    with pytest.raises(ValueError):
        buffer.chain([smaller_buffer])
    with pytest.raises(ValueError):
        buffer.chain([buffer, smaller_buffer])

def test23_chain_leaves_source_buffers_unchanged():
    """Chaining does not modify the buffers it is applied to"""
    buffers = chainable_buffers(3)
    reference = [buffer_to_numpy(b) for b in buffers]
    buffer_sizes = [b.buffer_size for b in buffers]

    buffers[0].chain(buffers[1:])

    for buffer, reference_, buffer_size in zip(buffers, reference,
                                               buffer_sizes):
        assert buffer.buffer_size == buffer_size
        fields = buffer_to_numpy(buffer)
        assert reference_.keys() == fields.keys()
        for name, value in fields.items():
            assert arrays_equal(value, reference_[name]), name

@pytest.mark.parametrize("sensing_index", [0, 1, 2])
def test24_chain_sensing_paths(sensing_index):
    """Chaining a sensing interaction keeps its target and point indices"""
    num_paths = 4
    buffers = chainable_buffers(2, num_paths)
    buffers.insert(sensing_index, sensing_paths_buffer(num_paths))

    chained_buffer = buffers[0].chain(buffers[1:])

    assert_chained(buffers, chained_buffer)
