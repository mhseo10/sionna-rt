#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the RCS module."""

import pytest
import drjit as dr
import mitsuba as mi
import numpy as np
from scipy.constants import speed_of_light

import sionna.rt
from sionna.rt import (load_scene, Transmitter, Receiver, PlanarArray,
                       AbsorberRadioMaterial, RadioMaterial)
from sionna.rt.constants import (InteractionType, DEFAULT_SENSING_TARGET_COLOR,
                                 DEFAULT_SENSING_TARGET_OPACITY)
from sionna.rt.rcs import (register_rcs, unregister_rcs, get_rcs,
                           register_cpm, unregister_cpm, get_cpm, RCSSolver,
                           ScatteringPoints, ScatteringModel, SensingTarget,
                           ConstantRCS, ConstantCPM,
                           ConstantRCSSensingTarget)
from sionna.rt.utils import rotation_matrix

# Cross-section returned by `_rcs_unit` [m^2]
_UNIT_RCS = 1.


def _rcs_unit(k_i, k_s, seed=0):
    """Cross-section of one square meter, whatever the directions."""
    return dr.ones(mi.Float, dr.width(k_i))


def _rcs_aspect(k_i, k_s, seed=0):
    """Cross-section depending on both directions, and never negative."""
    return 1. + dr.abs(k_i.x) + dr.abs(k_s.z)


def _rcs_probe(k_i, k_s, seed=0):
    """Reports the directions it is evaluated on, as a combination of their
    components that no rotation leaves unchanged."""
    return (k_i.x + 2.*k_i.y + 3.*k_i.z
            + 5.*k_s.x + 7.*k_s.y + 11.*k_s.z)


def _rcs_seed(k_i, k_s, seed=0):
    """Reports the seed it is evaluated with."""
    return dr.zeros(mi.Float, dr.width(k_i)) + float(seed)


def _cpm_real(k_i, k_s, seed=0):
    """Purely real CPM, with broadcast scalar entries."""
    zero = dr.zeros(mi.Float, dr.width(k_i))
    return (mi.Matrix2f(zero + k_i.x, zero, zero, zero + k_s.z),
            dr.zeros(mi.Matrix2f, dr.width(k_i)))


def _cpm_cplx(k_i, k_s, seed=0):
    """Complex CPM with all four entries populated."""
    return (mi.Matrix2f(k_i.x, k_i.y, k_i.z, k_s.x),
            mi.Matrix2f(k_s.y, k_s.z, 0, k_i.x))


def _cpm_directions(k_i, k_s, seed=0):
    """Reports the directions it is evaluated on: k_i as real part, k_s as
    imaginary part."""
    return (mi.Matrix2f(k_i.x, k_i.y, k_i.z, 0),
            mi.Matrix2f(k_s.x, k_s.y, k_s.z, 0))


def _cpm_seed(k_i, k_s, seed=0):
    """Reports the seed it is evaluated with, as the real part."""
    zero = dr.zeros(mi.Float, dr.width(k_i))
    return (mi.Matrix2f(zero + float(seed), zero, zero, zero),
            dr.zeros(mi.Matrix2f, dr.width(k_i)))


@pytest.fixture
def unique_rcs():
    """Register temporary RCS callables and unregister them afterwards."""
    names = []

    def _register(name, fn):
        register_rcs(name, fn)
        names.append(name)
        return name

    yield _register

    for name in names:
        unregister_rcs(name)


@pytest.fixture
def unique_cpm():
    """Register temporary CPM callables and unregister them afterwards."""
    names = []

    def _register(name, fn):
        register_cpm(name, fn)
        names.append(name)
        return name

    yield _register

    for name in names:
        unregister_cpm(name)


def test_register_get_unregister_rcs(unique_rcs):
    unique_rcs("unit_rcs_lookup", _rcs_unit)
    assert get_rcs("unit_rcs_lookup") is _rcs_unit
    unregister_rcs("unit_rcs_lookup")
    with pytest.raises(ValueError, match="not found"):
        get_rcs("unit_rcs_lookup")


def test_register_rcs_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="must be a string"):
        register_rcs(1, _rcs_unit)
    with pytest.raises(ValueError, match="must be a callable"):
        register_rcs("bad", "not-callable")
    with pytest.raises(ValueError, match="must be a string"):
        get_rcs(1)
    with pytest.raises(ValueError, match="must be a string"):
        unregister_rcs(1)
    with pytest.raises(ValueError, match="not found"):
        get_rcs("does_not_exist")


def test_register_get_unregister_cpm(unique_cpm):
    unique_cpm("unit_cpm_lookup", _cpm_real)
    assert get_cpm("unit_cpm_lookup") is _cpm_real
    unregister_cpm("unit_cpm_lookup")
    with pytest.raises(ValueError, match="not found"):
        get_cpm("unit_cpm_lookup")


def test_register_cpm_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="must be a string"):
        register_cpm(1, _cpm_real)
    with pytest.raises(ValueError, match="must be a callable"):
        register_cpm("bad", "not-callable")
    with pytest.raises(ValueError, match="must be a string"):
        get_cpm(1)
    with pytest.raises(ValueError, match="must be a string"):
        unregister_cpm(1)
    with pytest.raises(ValueError, match="not found"):
        get_cpm("does_not_exist")


def test_rcs_and_cpm_registries_are_independent(unique_rcs, unique_cpm):
    unique_rcs("unit_shared_name", _rcs_unit)
    unique_cpm("unit_shared_name", _cpm_real)

    assert get_rcs("unit_shared_name") is _rcs_unit
    assert get_cpm("unit_shared_name") is _cpm_real


def test_constant_rcs_broadcasts_its_cross_section():
    rcs = ConstantRCS(sigma=7.)
    assert dr.allclose(rcs.sigma, 7.)

    k_i = mi.Vector3f([1, 0, 0], [0, 1, 0], [0, 0, 1])
    k_s = mi.Vector3f([0, 0, 1], [0, 1, 0], [1, 0, 0])

    # One cross-section per sample, whatever the directions
    sigma = rcs(k_i, k_s)
    assert dr.width(sigma) == 3
    assert dr.allclose(sigma, 7.)

    rcs.sigma = 0.5
    assert dr.allclose(rcs(k_i, k_s), 0.5)

    with pytest.raises(ValueError, match="non-negative"):
        ConstantRCS(sigma=-1.)


@pytest.mark.parametrize("xpr_db", (None, 0., 6., -6.))
def test_constant_cpm_is_unitary_and_set_by_its_xpr(xpr_db):
    cpm = ConstantCPM(xpr_db=xpr_db)

    k_i = mi.Vector3f([1, 0], [0, 1], [0, 0])
    k_s = mi.Vector3f([0, 0], [0, 1], [1, 0])
    real, imag = cpm(k_i, k_s)

    # One real matrix per sample
    assert real.shape == (2, 2, 2)
    assert dr.allclose(imag, mi.Matrix2f(0))

    # The matrix is a rotation, so it redistributes the scattered power
    # without changing its total
    norm_sq = sum(dr.square(real[i, j]) for i in range(2) for j in range(2))
    assert dr.allclose(norm_sq, 2.)

    # The co- to cross-polarized power ratio is the XPR
    co_sq, cross_sq = dr.square(real[0, 0]), dr.square(real[1, 0])
    if xpr_db is None:
        assert dr.allclose(real, dr.identity(mi.Matrix2f, 2))
    else:
        assert dr.allclose(10.*dr.log(co_sq/cross_sq)/dr.log(10.), xpr_db,
                           atol=1e-5)


def _points(n):
    """n distinct scattering points, all with zero orientation."""
    coords = list(range(n))
    zeros = [0.0] * n
    return (mi.Point3f(coords, zeros, zeros), mi.Point3f(zeros, zeros, zeros))


def test_constructor_resolves_names(unique_rcs, unique_cpm):
    unique_rcs("unit_rcs_ctor", _rcs_aspect)
    unique_cpm("unit_cpm_ctor", _cpm_cplx)

    pos, ori = _points(3)
    sp = ScatteringPoints(pos, ori, [_rcs_unit, "unit_rcs_ctor", _rcs_unit],
                          [_cpm_real, "unit_cpm_ctor", _cpm_real])

    # One callable of each kind per point, in point order
    assert sp.rcs_callables == [_rcs_unit, _rcs_aspect, _rcs_unit]
    assert sp.cpm_callables == [_cpm_real, _cpm_cplx, _cpm_real]


def test_constructor_broadcasts_single_rcs(unique_rcs):
    unique_rcs("unit_rcs_bcast", _rcs_unit)

    pos, ori = _points(3)
    sp = ScatteringPoints(pos, ori, _rcs_unit, _cpm_real)
    assert sp.rcs_callables == [_rcs_unit] * 3

    sp_named = ScatteringPoints(*_points(2), "unit_rcs_bcast", _cpm_real)
    assert sp_named.rcs_callables == [_rcs_unit] * 2


def test_constructor_broadcasts_single_cpm(unique_cpm):
    unique_cpm("unit_cpm_bcast", _cpm_real)

    pos, ori = _points(3)
    sp = ScatteringPoints(pos, ori, _rcs_unit, _cpm_cplx)
    assert sp.cpm_callables == [_cpm_cplx] * 3

    sp_named = ScatteringPoints(*_points(2), _rcs_unit, "unit_cpm_bcast")
    assert sp_named.cpm_callables == [_cpm_real] * 2


def test_constructor_requires_a_cpm_for_every_point():
    # A CPM is required just like an RCS, so a point cannot be given only one
    # of the two
    pos, ori = _points(2)
    with pytest.raises(ValueError, match="one entry per scattering point"):
        ScatteringPoints(pos, ori, _rcs_unit)

    # A point which does not depolarize is described by a `ConstantCPM`
    # without a cross-polarization ratio
    sp = ScatteringPoints(pos, ori, _rcs_unit, ConstantCPM())
    real, imag = sp.eval_cpm(mi.Vector3f([1, 0], [0, 1], [0, 0]),
                             mi.Vector3f([0, 1], [1, 0], [0, 0]),
                             mi.Point3f(0, 0, 0), dr.zeros(mi.UInt, 2),
                             mi.UInt([0, 1]))
    assert dr.allclose(real, dr.identity(mi.Matrix2f, 2))
    assert dr.allclose(imag, mi.Matrix2f(0))


def test_constructor_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same width"):
        ScatteringPoints(mi.Point3f([0, 1], [0, 0], [0, 0]),
                         mi.Point3f(0, 0, 0))

    # One RCS per point is required
    with pytest.raises(ValueError, match="one entry per scattering point"):
        ScatteringPoints(*_points(2), [_rcs_unit])

    with pytest.raises(ValueError, match="one entry per scattering point"):
        ScatteringPoints(*_points(1), [_rcs_unit, _rcs_aspect])

    with pytest.raises(ValueError, match="must be a list"):
        ScatteringPoints(*_points(1), 1)

    with pytest.raises(ValueError, match="callable or string"):
        ScatteringPoints(*_points(1), [1])

    with pytest.raises(ValueError, match="not found"):
        ScatteringPoints(*_points(1), ["missing_rcs"])

    # One CPM per point is required as well
    with pytest.raises(ValueError, match="one entry per scattering point"):
        ScatteringPoints(*_points(2), _rcs_unit, [_cpm_real])

    with pytest.raises(ValueError, match="must be a list"):
        ScatteringPoints(*_points(1), _rcs_unit, 1)

    with pytest.raises(ValueError, match="callable or string"):
        ScatteringPoints(*_points(1), _rcs_unit, [1])

    with pytest.raises(ValueError, match="not found"):
        ScatteringPoints(*_points(1), _rcs_unit, ["missing_cpm"])


def test_default_collection_is_empty():
    sp = ScatteringPoints()
    assert dr.width(sp.lcs_positions) == 0
    assert dr.width(sp.lcs_orientations) == 0
    assert sp.rcs_callables == []
    assert sp.cpm_callables == []


def test_concat_with_empty_does_not_insert_dummy_origin():
    positions = mi.Point3f(1.0, 2.0, 3.0)
    orientations = mi.Point3f(0.1, 0.2, 0.3)
    sp = ScatteringPoints().concat(ScatteringPoints(positions, orientations,
                                                    [_rcs_unit], _cpm_real))

    assert dr.width(sp.lcs_positions) == 1
    assert dr.allclose(sp.lcs_positions, positions)
    assert dr.allclose(sp.lcs_orientations, orientations)
    assert sp.rcs_callables == [_rcs_unit]


def test_repeated_concat_concatenates_callables():
    sp = ScatteringPoints()
    sp = sp.concat(ScatteringPoints(*_points(2), [_rcs_unit, _rcs_unit],
                                    _cpm_real))
    sp = sp.concat(ScatteringPoints(*_points(2), [_rcs_unit, _rcs_aspect],
                                    _cpm_real))

    # Every point keeps its own callable, in point order
    assert dr.width(sp.lcs_positions) == 4
    assert sp.rcs_callables == [_rcs_unit, _rcs_unit, _rcs_unit, _rcs_aspect]


def test_concat_appends_callables_of_other():
    a = ScatteringPoints(*_points(1), [_rcs_unit], _cpm_real)
    b = ScatteringPoints(*_points(2), [_rcs_aspect, _rcs_unit], _cpm_real)

    concatenated = a.concat(b)
    assert concatenated.rcs_callables == [_rcs_unit, _rcs_aspect, _rcs_unit]
    assert b.rcs_callables == [_rcs_aspect, _rcs_unit]


def test_concat_keeps_rcs_and_cpm_aligned():
    a = ScatteringPoints(*_points(1), [_rcs_unit], [_cpm_real])
    b = ScatteringPoints(*_points(2), [_rcs_aspect, _rcs_probe],
                         [_cpm_cplx, _cpm_directions])

    concatenated = a.concat(b)

    # Both lists are appended in the same order, so a point keeps the pair of
    # callables it was given
    assert concatenated.rcs_callables == [_rcs_unit, _rcs_aspect, _rcs_probe]
    assert concatenated.cpm_callables == [_cpm_real, _cpm_cplx,
                                          _cpm_directions]


def test_concat_leaves_both_operands_untouched():
    a = ScatteringPoints(*_points(2), [_rcs_unit, _rcs_aspect], _cpm_real)
    b = ScatteringPoints(*_points(3), [_rcs_aspect, _rcs_unit, _rcs_aspect],
                         _cpm_real)

    c = a.concat(b)

    # c holds the combination: a's callables, then b's ones
    assert dr.width(c.lcs_positions) == 5
    assert c.rcs_callables == [_rcs_unit, _rcs_aspect,
                               _rcs_aspect, _rcs_unit, _rcs_aspect]

    assert dr.width(a.lcs_positions) == 2
    assert a.rcs_callables == [_rcs_unit, _rcs_aspect]

    assert dr.width(b.lcs_positions) == 3
    assert b.rcs_callables == [_rcs_aspect, _rcs_unit, _rcs_aspect]


def test_concat_from_empty_does_not_alias_operand():
    # Concatenating with an empty collection returns the operand's arrays
    # directly, so verify the result is still independent of the operand.
    b = ScatteringPoints(*_points(2), [_rcs_unit, _rcs_aspect], _cpm_real)

    c = ScatteringPoints().concat(b)
    c = c.concat(ScatteringPoints(*_points(1), [_rcs_unit], _cpm_real))

    assert dr.width(c.lcs_positions) == 3
    assert c.rcs_callables == [_rcs_unit, _rcs_aspect, _rcs_unit]

    assert dr.width(b.lcs_positions) == 2
    assert b.rcs_callables == [_rcs_unit, _rcs_aspect]


def test_concat_rejects_non_scattering_points():
    sp = ScatteringPoints()
    with pytest.raises(ValueError, match="ScatteringPoints"):
        sp.concat("not-points")
    with pytest.raises(ValueError, match="ScatteringPoints"):
        sp.concat([ScatteringPoints(), "not-points"])


def test_concat_list_matches_repeated_concat():
    a = ScatteringPoints(*_points(2), [_rcs_unit, _rcs_aspect], _cpm_real)
    b = ScatteringPoints(*_points(1), [_rcs_probe], _cpm_real)
    c = ScatteringPoints(*_points(3), [_rcs_aspect, _rcs_unit, _rcs_aspect],
                         _cpm_real)

    at_once = a.concat([b, c])
    one_by_one = a.concat(b).concat(c)

    assert dr.allclose(at_once.lcs_positions, one_by_one.lcs_positions)
    assert dr.allclose(at_once.lcs_orientations, one_by_one.lcs_orientations)
    assert at_once.rcs_callables == one_by_one.rcs_callables

    # Empty collections of the list are skipped
    with_empty = a.concat([ScatteringPoints(), b, ScatteringPoints(), c])
    assert dr.allclose(with_empty.lcs_positions, one_by_one.lcs_positions)
    assert with_empty.rcs_callables == one_by_one.rcs_callables


def test_gcs_positions_rotates_and_translates_to_targets():
    lcs_positions = mi.Point3f([1, 0, 0], [0, 1, 0], [0, 0, 1])
    sp = ScatteringPoints(lcs_positions,
                          rcs=[_rcs_unit, _rcs_unit, _rcs_unit],
                          cpm=_cpm_real)

    st_positions = mi.Point3f([10, -5], [0, 2], [0, 3])
    st_orientations = mi.Point3f([dr.pi/2, 0], [0, 0], [0, 0])
    st_indices = mi.UInt([0, 0, 1])

    gcs = sp.gcs_positions(st_positions, st_orientations, st_indices)

    # First target is rotated by 90 degrees around the z-axis, the second one
    # is not rotated
    assert dr.allclose(gcs, mi.Point3f([10, 9, -5], [1, 0, 2], [0, 0, 4]),
                       atol=1e-6)


def test_scaled_lcs_positions_scales_by_the_scaling_of_the_target():
    lcs_positions = mi.Point3f([1, 0, 0], [0, 1, 0], [0, 0, 1])
    sp = ScatteringPoints(lcs_positions,
                          rcs=[_rcs_unit, _rcs_unit, _rcs_unit],
                          cpm=_cpm_real)

    # The first target is scaled unevenly along its three axes, the second one
    # only along the z-axis
    st_scalings = mi.Vector3f([2, 1], [3, 1], [4, 0.5])
    st_indices = mi.UInt([0, 0, 1])

    scaled = sp.scaled_lcs_positions(st_scalings, st_indices)

    assert dr.allclose(scaled, mi.Point3f([2, 0, 0], [0, 3, 0], [0, 0, 0.5]),
                       atol=1e-6)


def test_gcs_positions_scales_before_rotating_and_translating():
    lcs_positions = mi.Point3f([1, 0, 0], [0, 1, 0], [0, 0, 1])
    sp = ScatteringPoints(lcs_positions,
                          rcs=[_rcs_unit, _rcs_unit, _rcs_unit],
                          cpm=_cpm_real)

    st_positions = mi.Point3f([10, -5], [0, 2], [0, 3])
    st_orientations = mi.Point3f([dr.pi/2, 0], [0, 0], [0, 0])
    st_scalings = mi.Vector3f([2, 1], [3, 1], [4, 0.5])
    st_indices = mi.UInt([0, 0, 1])

    gcs = sp.gcs_positions(st_positions, st_orientations, st_indices,
                           st_scalings=st_scalings)

    # The points are scaled in the frame of their target, so the rotation of
    # the first target turns its scaled offsets rather than the original ones
    assert dr.allclose(gcs, mi.Point3f([10, 7, -5], [2, 0, 2], [0, 0, 3.5]),
                       atol=1e-6)

    # Unit scalings leave the points where they are
    unit = sp.gcs_positions(st_positions, st_orientations, st_indices,
                            st_scalings=mi.Vector3f([1, 1], [1, 1], [1, 1]))
    assert dr.allclose(unit, sp.gcs_positions(st_positions, st_orientations,
                                              st_indices), atol=1e-6)


def test_remove_keeps_remaining_points_aligned():
    orientations = mi.Point3f([0.1, 0.2, 0.3], [0, 0, 0], [0, 0, 0])
    sp = ScatteringPoints(_points(3)[0], orientations,
                          [_rcs_unit, _rcs_aspect, _rcs_probe], _cpm_real)

    reduced = sp.remove(1)

    # The kept points keep their relative order, and so do their callables
    assert dr.allclose(reduced.lcs_positions,
                       mi.Point3f([0, 2], [0, 0], [0, 0]))
    assert dr.allclose(reduced.lcs_orientations,
                       mi.Point3f([0.1, 0.3], [0, 0], [0, 0]))
    assert reduced.rcs_callables == [_rcs_unit, _rcs_probe]


def test_remove_reindexes_callables_for_eval_rcs():
    sp = ScatteringPoints(*_points(3),
                          [_rcs_unit, _rcs_aspect, _rcs_probe],
                          [_cpm_real, _cpm_real, _cpm_directions])

    # After removing point 0, the callables of former point 2 are at index 1
    reduced = sp.remove([0])

    assert reduced.rcs_callables == [_rcs_aspect, _rcs_probe]
    assert reduced.cpm_callables == [_cpm_real, _cpm_directions]

    k_i = mi.Vector3f(1, 0, 0)
    k_s = mi.Vector3f(0, 1, 0)
    sigma = reduced.eval_rcs(k_i, k_s, mi.Point3f(0, 0, 0), mi.UInt(0),
                             mi.UInt(1))
    assert dr.allclose(sigma, _rcs_probe(k_i, k_s))

    real, imag = reduced.eval_cpm(k_i, k_s, mi.Point3f(0, 0, 0), mi.UInt(0),
                                  mi.UInt(1))
    assert dr.allclose(real, _cpm_directions(k_i, k_s)[0])
    assert dr.allclose(imag, _cpm_directions(k_i, k_s)[1])


def test_remove_multiple_and_duplicate_indices():
    sp = ScatteringPoints(*_points(4),
                          [_rcs_unit, _rcs_aspect, _rcs_unit, _rcs_aspect],
                          _cpm_real)

    reduced = sp.remove([2, 0, 2])

    assert dr.allclose(reduced.lcs_positions,
                       mi.Point3f([1, 3], [0, 0], [0, 0]))
    assert reduced.rcs_callables == [_rcs_aspect, _rcs_aspect]


def test_remove_all_points_yields_empty_collection():
    sp = ScatteringPoints(*_points(2), [_rcs_unit, _rcs_aspect], _cpm_real)

    reduced = sp.remove([0, 1])

    assert dr.width(reduced.lcs_positions) == 0
    assert dr.width(reduced.lcs_orientations) == 0
    assert reduced.rcs_callables == []


def test_remove_leaves_operand_untouched():
    sp = ScatteringPoints(*_points(2), [_rcs_unit, _rcs_aspect], _cpm_real)

    sp.remove(0)

    assert dr.width(sp.lcs_positions) == 2
    assert sp.rcs_callables == [_rcs_unit, _rcs_aspect]


def test_remove_rejects_invalid_indices():
    sp = ScatteringPoints(*_points(2), [_rcs_unit, _rcs_aspect], _cpm_real)

    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        sp.remove(2)
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        sp.remove(-1)
    with pytest.raises(ValueError, match="list of integers"):
        sp.remove("0")
    with pytest.raises(ValueError, match="list of integers"):
        sp.remove([0.0])

    # No point can be removed from an empty collection
    with pytest.raises(ValueError, match=r"\[0, 0\)"):
        ScatteringPoints().remove(0)


def _expected_per_sample(per_point, spst_indices, k_i, k_s, part=None):
    """Reference value: every sample evaluated with the callable of the
    scattering point it is associated with.

    With `part` left to `None` the callables return one value per sample,
    otherwise they return a pair of matrices of which `part` is taken.
    """
    num_samples = dr.width(k_i)
    zero = dr.zeros(mi.Float, num_samples)
    index = mi.UInt(spst_indices) * num_samples \
            + dr.arange(mi.UInt, num_samples)

    if part is None:
        values = [callable_(k_i, k_s) for callable_ in per_point]
        return dr.gather(mi.Float, dr.concat([v + zero for v in values]),
                         index)

    matrices = [callable_(k_i, k_s)[part] for callable_ in per_point]
    return mi.Matrix2f(*[
        dr.gather(mi.Float, dr.concat([m[i, j] + zero for m in matrices]),
                  index)
        for i in range(2) for j in range(2)])


# Samples of the evaluation tests: four directions, dispatched over two
# scattering points
_EVAL_K_I = mi.Vector3f([1, 2, 7, 8], [3, 4, 9, 1], [0, 0, 2, 3])
_EVAL_K_S = mi.Vector3f([0, 0, 1, 2], [0, 0, 3, 4], [5, 6, 7, 8])
_EVAL_SPST_INDICES = mi.UInt([0, 1, 1, 0])


def test_eval_rcs_dispatches_per_sample():
    per_point = [_rcs_unit, _rcs_aspect]
    sp = ScatteringPoints(*_points(2), per_point, _cpm_real)

    st_indices = dr.zeros(mi.UInt, 4)
    sigma = sp.eval_rcs(_EVAL_K_I, _EVAL_K_S, mi.Point3f(0, 0, 0), st_indices,
                        _EVAL_SPST_INDICES)

    # One cross-section per sample
    assert dr.width(sigma) == 4
    assert dr.allclose(sigma, _expected_per_sample(per_point,
                                                   _EVAL_SPST_INDICES,
                                                   _EVAL_K_I, _EVAL_K_S))


def test_eval_cpm_dispatches_per_sample():
    per_point = [_cpm_real, _cpm_cplx]
    sp = ScatteringPoints(*_points(2), _rcs_unit, per_point)

    st_indices = dr.zeros(mi.UInt, 4)
    real, imag = sp.eval_cpm(_EVAL_K_I, _EVAL_K_S, mi.Point3f(0, 0, 0),
                             st_indices, _EVAL_SPST_INDICES)

    # One matrix per sample
    assert real.shape == (2, 2, 4)
    assert imag.shape == (2, 2, 4)

    assert dr.allclose(real, _expected_per_sample(per_point,
                                                  _EVAL_SPST_INDICES,
                                                  _EVAL_K_I, _EVAL_K_S, 0))
    assert dr.allclose(imag, _expected_per_sample(per_point,
                                                  _EVAL_SPST_INDICES,
                                                  _EVAL_K_I, _EVAL_K_S, 1))


def test_eval_jones_matrix_combines_rcs_and_cpm():
    per_point_rcs = [_rcs_unit, _rcs_aspect]
    per_point_cpm = [_cpm_real, _cpm_cplx]
    sp = ScatteringPoints(*_points(2), per_point_rcs, per_point_cpm)

    st_indices = dr.zeros(mi.UInt, 4)
    args = (_EVAL_K_I, _EVAL_K_S, mi.Point3f(0, 0, 0), st_indices,
            _EVAL_SPST_INDICES)
    real, imag = sp.eval_jones_matrix(*args)

    # One Jones matrix per sample
    assert real.shape == (2, 2, 4)
    assert imag.shape == (2, 2, 4)

    # `J = sqrt(sigma) W`, with both factors evaluated per sample. The
    # imaginary part is carried over, so a complex CPM yields a complex Jones
    # matrix.
    amplitude = dr.sqrt(sp.eval_rcs(*args))
    cpm_real, cpm_imag = sp.eval_cpm(*args)
    assert dr.allclose(real, amplitude*cpm_real)
    assert dr.allclose(imag, amplitude*cpm_imag)
    assert not dr.allclose(imag, mi.Matrix2f(0))


def test_eval_passes_the_seed_to_the_callables():
    sp = ScatteringPoints(*_points(2), _rcs_seed, _cpm_seed)

    st_indices = dr.zeros(mi.UInt, 4)
    args = (_EVAL_K_I, _EVAL_K_S, mi.Point3f(0, 0, 0), st_indices,
            _EVAL_SPST_INDICES)

    # The callables report the seed they are evaluated with
    assert dr.allclose(sp.eval_rcs(*args, 7), mi.Float(7))
    assert dr.allclose(sp.eval_cpm(*args, 7)[0][0, 0], mi.Float(7))
    assert dr.allclose(sp.eval_jones_matrix(*args, 9)[0][0, 0],
                       mi.Float(9)*dr.sqrt(mi.Float(9)))

    # Without a seed, the callables are evaluated with the default one
    assert dr.allclose(sp.eval_rcs(*args), mi.Float(0))


def test_eval_passes_the_seed_through_the_dispatch():
    # Distinct callables per point are dispatched rather than evaluated
    # directly, which is the other path the seed travels
    sp = ScatteringPoints(*_points(2), [_rcs_seed, _rcs_unit],
                          [_cpm_seed, _cpm_real])

    st_indices = dr.zeros(mi.UInt, 4)
    args = (_EVAL_K_I, _EVAL_K_S, mi.Point3f(0, 0, 0), st_indices,
            _EVAL_SPST_INDICES)

    # Only the samples of the first point report the seed, the other ones
    # being evaluated with `_rcs_unit`
    expected = dr.select(_EVAL_SPST_INDICES == 0, 3., _UNIT_RCS)
    assert dr.allclose(sp.eval_rcs(*args, 3), expected)


def test_solver_passes_its_seed_to_the_scattering_points():
    # Same geometry as `test_solver_free_space_path`, with a scattering point
    # whose cross-section is its seed
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0), rcs=_rcs_seed)

    solver = RCSSolver()
    coefficients = []
    for seed in (4, 25):
        paths = solver(scene, max_depth=1, samples_per_sp=10**4, seed=seed)
        a_real, a_imag = paths.a
        coefficients.append(np.abs(a_real.numpy().squeeze()
                                   + 1j*a_imag.numpy().squeeze()))

    # The coefficient scales as the square-root of the cross-section, which is
    # the seed, so it doubles from a seed of 4 to a seed of 25
    assert np.allclose(coefficients[1]/coefficients[0],
                       np.sqrt(25./4.), rtol=1e-5)


def test_solver_draws_a_seed_when_it_is_not_given():
    # Same scene, so that the coefficient reports the seed the solver used
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0), rcs=_rcs_seed)

    solver = RCSSolver()

    def coefficient():
        a_real, a_imag = solver(scene, max_depth=1, samples_per_sp=10**3).a
        return np.abs(a_real.numpy().squeeze() + 1j*a_imag.numpy().squeeze())

    # Without a seed, every call draws its own
    np.random.seed(7)
    first, second = coefficient(), coefficient()
    assert first != second

    # The seed is drawn from the numpy global generator, so seeding it makes
    # the solver reproducible
    np.random.seed(7)
    assert coefficient() == first


def test_solver_rejects_an_invalid_seed():
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0))

    with pytest.raises(TypeError, match="`seed` must be an integer"):
        RCSSolver()(scene, seed=1.5)
    with pytest.raises(ValueError, match="`seed` must be greater"):
        RCSSolver()(scene, seed=-1)


def test_eval_rotates_directions_to_local_frame():
    lcs_orientations = mi.Point3f([0.1, -0.7], [0.2, 0.4], [0.3, 1.1])
    sp = ScatteringPoints(_points(2)[0], lcs_orientations,
                          [_rcs_probe, _rcs_probe],
                          [_cpm_directions, _cpm_directions])

    k_i = mi.Vector3f([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0])
    k_s = mi.Vector3f([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1])
    st_orientations = mi.Point3f([0.5, 0.0], [-0.2, 0.9], [0.8, 0.0])
    st_indices = mi.UInt([0, 1, 0, 1])
    spst_indices = mi.UInt([1, 0, 0, 1])
    args = (k_i, k_s, st_orientations, st_indices, spst_indices)

    # Directions the callables are expected to be evaluated on
    st_ort = dr.gather(mi.Point3f, st_orientations, st_indices)
    spst_ort = dr.gather(mi.Point3f, lcs_orientations, spst_indices)
    to_local = (rotation_matrix(st_ort)@rotation_matrix(spst_ort)).T
    k_i_local = to_local@k_i
    k_s_local = to_local@k_s

    # Both the RCS and the CPM report the directions they were evaluated on
    assert dr.allclose(sp.eval_rcs(*args), _rcs_probe(k_i_local, k_s_local))

    real, imag = sp.eval_cpm(*args)
    assert dr.allclose(real, mi.Matrix2f(k_i_local.x, k_i_local.y,
                                         k_i_local.z, 0))
    assert dr.allclose(imag, mi.Matrix2f(k_s_local.x, k_s_local.y,
                                         k_s_local.z, 0))

    # The rotation is not the identity, i.e., the directions did change
    assert not dr.allclose(k_i_local, k_i)
    assert not dr.allclose(k_s_local, k_s)


def test_eval_empty_catalog():
    sp = ScatteringPoints()
    args = (mi.Vector3f(1, 0, 0), mi.Vector3f(0, 1, 0), mi.Point3f(0, 0, 0),
            mi.UInt(0), mi.UInt(0))

    assert dr.allclose(sp.eval_rcs(*args), mi.Float(0))

    for real, imag in (sp.eval_cpm(*args), sp.eval_jones_matrix(*args)):
        assert dr.allclose(real, mi.Matrix2f(0))
        assert dr.allclose(imag, mi.Matrix2f(0))


def test_scattering_model_is_empty_by_default():
    sm = ScatteringModel()
    assert dr.width(sm.spst.lcs_positions) == 0
    assert sm.spst.rcs_callables == []


def test_scattering_model_defaults_orientations_to_zero():
    sm = ScatteringModel(mi.Point3f([1, 2], [0, 0], [0, 0]), rcs=_rcs_unit,
                         cpm=_cpm_real)

    assert dr.width(sm.spst.lcs_orientations) == 2
    assert dr.allclose(sm.spst.lcs_orientations, mi.Point3f(0, 0, 0))


def test_scattering_model_add_broadcasts_single_rcs(unique_rcs):
    unique_rcs("unit_rcs_add", _rcs_aspect)

    sm = ScatteringModel()
    sm.add_scattering_points(*_points(2), _rcs_unit, _cpm_real)
    sm.add_scattering_points(*_points(2), "unit_rcs_add", _cpm_real)

    assert dr.width(sm.spst.lcs_positions) == 4
    assert sm.spst.rcs_callables == [_rcs_unit, _rcs_unit,
                                     _rcs_aspect, _rcs_aspect]


def test_scattering_model_add_validates_through_scattering_points():
    sm = ScatteringModel()

    with pytest.raises(ValueError, match="same width"):
        sm.add_scattering_points(mi.Point3f([0, 1], [0, 0], [0, 0]),
                                 mi.Point3f(0, 0, 0))
    with pytest.raises(ValueError, match="one entry per scattering point"):
        sm.add_scattering_points(*_points(2), [_rcs_unit], _cpm_real)
    with pytest.raises(ValueError, match="not found"):
        sm.add_scattering_points(*_points(1), ["missing_rcs"], _cpm_real)

    # A CPM is required as well
    with pytest.raises(ValueError, match="one entry per scattering point"):
        sm.add_scattering_points(*_points(1), _rcs_unit)


def test_scattering_model_remove_scattering_points():
    sm = ScatteringModel(*_points(3), [_rcs_unit, _rcs_aspect, _rcs_unit],
                         _cpm_real)

    sm.remove_scattering_points(1)

    assert dr.allclose(sm.spst.lcs_positions,
                       mi.Point3f([0, 2], [0, 0], [0, 0]))
    assert sm.spst.rcs_callables == [_rcs_unit, _rcs_unit]


def test_scattering_model_mutations_do_not_affect_previous_spst():
    sm = ScatteringModel(*_points(2), [_rcs_unit, _rcs_aspect], _cpm_real)
    spst = sm.spst

    sm.add_scattering_points(*_points(1), [_rcs_unit], _cpm_real)
    sm.remove_scattering_points(0)

    # The collection handed out earlier is left untouched
    assert dr.width(spst.lcs_positions) == 2
    assert spst.rcs_callables == [_rcs_unit, _rcs_aspect]

    assert dr.width(sm.spst.lcs_positions) == 2
    assert sm.spst.rcs_callables == [_rcs_aspect, _rcs_unit]


def _cuboid_target(name="st", scattering_model=None, **kwargs):
    """Sensing target shaped as a unit cube."""
    if scattering_model is None:
        scattering_model = ScatteringModel(mi.Point3f(1, 0, 0), rcs=_rcs_unit,
                                           cpm=_cpm_real)
    kwargs.setdefault("length", 1.)
    kwargs.setdefault("width", 1.)
    kwargs.setdefault("height", 1.)
    return SensingTarget(name, scattering_model=scattering_model, **kwargs)


def test_sensing_target_rejects_invalid_scattering_model():
    with pytest.raises(ValueError, match="ScatteringModel"):
        _cuboid_target(scattering_model="nope")


def test_sensing_target_requires_exactly_one_shape():
    sm = ScatteringModel(mi.Point3f(1, 0, 0), rcs=_rcs_unit, cpm=_cpm_real)

    # No shape at all
    with pytest.raises(ValueError, match="Exactly one"):
        SensingTarget("st", scattering_model=sm)

    # Two shapes at once
    with pytest.raises(ValueError, match="Exactly one"):
        SensingTarget("st", scattering_model=sm, fname=sionna.rt.scene.sphere,
                      length=1., width=1., height=1.)
    with pytest.raises(ValueError, match="Exactly one"):
        SensingTarget("st", scattering_model=sm,
                      mi_mesh=_cuboid_target().mi_mesh, height=1.)

    # Incomplete set of dimensions
    with pytest.raises(ValueError, match="must all be provided"):
        SensingTarget("st", scattering_model=sm, length=1., width=1.)

    # Non-positive dimensions
    with pytest.raises(ValueError, match="must be positive"):
        SensingTarget("st", scattering_model=sm, length=1., width=1.,
                      height=0.)


def test_sensing_target_cuboid_has_requested_dimensions():
    st = _cuboid_target(length=4., width=2., height=1.5)

    bbox = st.mi_mesh.bbox()
    assert np.allclose(bbox.min, [-2., -1., -0.75])
    assert np.allclose(bbox.max, [2., 1., 0.75])
    assert np.allclose(st.position.numpy().squeeze(), [0., 0., 0.])
    # Duplicate vertices of the Mitsuba cube are merged
    assert st.mi_mesh.vertex_count() == 8


def test_sensing_target_can_be_built_from_a_mesh():
    sm = ScatteringModel(mi.Point3f(1, 0, 0), rcs=_rcs_unit, cpm=_cpm_real)

    from_file = SensingTarget("from-file", scattering_model=sm,
                              fname=sionna.rt.scene.sphere)
    from_mesh = SensingTarget("from-mesh", scattering_model=sm,
                              mi_mesh=sionna.rt.utils.load_mesh(
                                  sionna.rt.scene.sphere))

    assert from_file.mi_mesh.vertex_count() == from_mesh.mi_mesh.vertex_count()
    for st in (from_file, from_mesh):
        assert isinstance(st.radio_material, AbsorberRadioMaterial)


def test_sensing_target_material_absorbs_and_is_not_editable():
    st = _cuboid_target()

    assert isinstance(st.radio_material, AbsorberRadioMaterial)
    assert st.radio_material.name == "st-material"

    with pytest.raises(ValueError, match="cannot be changed"):
        st.radio_material = RadioMaterial(name="other")
    with pytest.raises(ValueError, match="cannot be changed"):
        st.radio_material = "itu_concrete"

    assert isinstance(st.radio_material, AbsorberRadioMaterial)


def test_sensing_target_color_defaults_and_is_applied_to_the_material():
    assert np.allclose(_cuboid_target().radio_material.color,
                       DEFAULT_SENSING_TARGET_COLOR)
    assert np.allclose(_cuboid_target(color=(1., 0.5, 0.)).radio_material.color,
                       (1., 0.5, 0.))


def test_sensing_target_display_opacity_defaults_and_is_settable():
    assert _cuboid_target().display_opacity == DEFAULT_SENSING_TARGET_OPACITY
    assert _cuboid_target(display_opacity=0.25).display_opacity == 0.25

    st = _cuboid_target()
    st.display_opacity = 1.
    assert st.display_opacity == 1.


@pytest.mark.parametrize("opacity", (-0.1, 1.1))
def test_sensing_target_rejects_invalid_display_opacity(opacity):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _cuboid_target(display_opacity=opacity)

    st = _cuboid_target()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        st.display_opacity = opacity


def test_sensing_target_clone_shares_the_scattering_model():
    sm = ScatteringModel(mi.Point3f(1, 0, 0), rcs=_rcs_unit, cpm=_cpm_real)
    st = _cuboid_target(scattering_model=sm, color=(1., 0.5, 0.),
                        display_opacity=0.25)

    clone = st.clone()

    assert isinstance(clone, SensingTarget)
    assert clone.name == "st-clone"
    assert clone.scattering_model is sm
    assert clone.mi_mesh is not st.mi_mesh
    assert np.allclose(clone.mi_mesh.bbox().min, st.mi_mesh.bbox().min)
    assert isinstance(clone.radio_material, AbsorberRadioMaterial)
    assert clone.radio_material is not st.radio_material
    assert np.allclose(clone.radio_material.color, (1., 0.5, 0.))
    assert clone.display_opacity == 0.25


def test_sensing_target_add_and_remove_scattering_points():
    sm = ScatteringModel(mi.Point3f(1, 0, 0), rcs=_rcs_unit, cpm=_cpm_real)
    st = _cuboid_target(scattering_model=sm)

    # The scattering points are reached through the model of the target
    assert st.scattering_model is sm

    st.scattering_model.add_scattering_points(mi.Point3f(0, 1, 0),
                                              mi.Point3f(0, 0, 0),
                                              [_rcs_aspect], _cpm_real)

    assert dr.width(sm.spst.lcs_positions) == 2
    assert sm.spst.rcs_callables == [_rcs_unit, _rcs_aspect]

    st.scattering_model.remove_scattering_points(0)

    assert dr.allclose(sm.spst.lcs_positions, mi.Point3f(0, 1, 0))
    assert sm.spst.rcs_callables == [_rcs_aspect]


def test_constant_rcs_target_defaults_to_a_centered_point_in_a_30cm_cube():
    st = ConstantRCSSensingTarget("st")

    assert st.dimensions == {"length": 0.3, "width": 0.3, "height": 0.3}
    bbox = st.mi_mesh.bbox()
    assert np.allclose(bbox.min, [-0.15, -0.15, -0.15])
    assert np.allclose(bbox.max, [0.15, 0.15, 0.15])

    spst = st.scattering_model.spst
    assert dr.width(spst.lcs_positions) == 1
    assert dr.allclose(spst.lcs_positions, mi.Point3f(0, 0, 0))
    assert dr.allclose(spst.lcs_orientations, mi.Point3f(0, 0, 0))
    assert spst.rcs_callables == [st.rcs]
    assert spst.cpm_callables == [st.cpm]
    assert isinstance(st.rcs, ConstantRCS)
    assert isinstance(st.cpm, ConstantCPM)


def test_constant_rcs_target_dimensions_follow_the_scaling():
    st = ConstantRCSSensingTarget("st", length=4., width=2., height=1.5)
    st.scaling = mi.Vector3f(0.5, 2., 4.)

    assert st.dimensions == pytest.approx({"length": 2., "width": 4.,
                                           "height": 6.})
    assert (st.length, st.width, st.height) == pytest.approx((2., 4., 6.))
    # The reported dimensions are those of the mesh the solver sees
    assert np.allclose(2.*st.lcs_half_extents.numpy().squeeze(),
                       [2., 4., 6.], atol=1e-5)

    # A clone is scaled like the target it was built from
    assert st.clone().dimensions == pytest.approx(st.dimensions)


def test_constant_rcs_target_dimensions_default_independently():
    st = ConstantRCSSensingTarget("st", length=4., height=1.5)

    assert st.length == 4.
    assert st.width == 0.3
    assert st.height == 1.5
    bbox = st.mi_mesh.bbox()
    assert np.allclose(bbox.min, [-2., -0.15, -0.75])
    assert np.allclose(bbox.max, [2., 0.15, 0.75])

    # The scattering point stays at the center of the LCS
    assert dr.allclose(st.scattering_model.spst.lcs_positions,
                       mi.Point3f(0, 0, 0))


@pytest.mark.parametrize("dimension", ("length", "width", "height"))
def test_constant_rcs_target_rejects_non_positive_dimensions(dimension):
    with pytest.raises(ValueError, match=f"`{dimension}` must be strictly"):
        ConstantRCSSensingTarget("st", **{dimension: 0.})
    with pytest.raises(ValueError, match=f"`{dimension}` must be strictly"):
        ConstantRCSSensingTarget("st", **{dimension: -1.})


def test_constant_rcs_target_takes_its_dimensions_from_a_mesh():
    from_file = ConstantRCSSensingTarget("from-file",
                                         fname=sionna.rt.scene.sphere)
    from_mesh = ConstantRCSSensingTarget(
        "from-mesh", mi_mesh=sionna.rt.utils.load_mesh(sionna.rt.scene.sphere))

    for st in (from_file, from_mesh):
        bbox = st.mi_mesh.bbox()
        extents = bbox.max - bbox.min
        assert np.allclose([st.length, st.width, st.height],
                           [extents.x, extents.y, extents.z])
        # The sphere of the built-in scenes is larger than the default cube
        assert st.length > 0.3
        assert isinstance(st.radio_material, AbsorberRadioMaterial)
        assert dr.allclose(st.scattering_model.spst.lcs_positions,
                           mi.Point3f(0, 0, 0))


def test_constant_rcs_target_rejects_conflicting_shapes():
    with pytest.raises(ValueError, match="Only one of"):
        ConstantRCSSensingTarget(
            "st", fname=sionna.rt.scene.sphere,
            mi_mesh=sionna.rt.utils.load_mesh(sionna.rt.scene.sphere))
    with pytest.raises(ValueError, match="mutually exclusive"):
        ConstantRCSSensingTarget("st", fname=sionna.rt.scene.sphere, width=1.)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ConstantRCSSensingTarget(
            "st", mi_mesh=sionna.rt.utils.load_mesh(sionna.rt.scene.sphere),
            length=1.)
    with pytest.raises(ValueError, match="must be a Mitsuba Mesh"):
        ConstantRCSSensingTarget("st", mi_mesh="nope")


def test_constant_rcs_target_exposes_the_rcs_parameters():
    # An XPR of 6dB splits the scattered power in a ratio of four to one
    # between the co-polarized and the cross-polarized components
    xpr_db = 10.*np.log10(4.)
    st = ConstantRCSSensingTarget("st", sigma=10., xpr_db=xpr_db)

    assert dr.allclose(st.sigma, 10.)
    assert dr.allclose(st.xpr_db, xpr_db)
    assert dr.allclose(st.rcs.sigma, 10.)
    assert dr.allclose(st.cpm.xpr_db, xpr_db)

    # The cross-section is constant, and the CPM is the unitary rotation of
    # the XPR
    k_i = mi.Vector3f([1, 0], [0, 1], [0, 0])
    k_s = mi.Vector3f([0, 0], [0, 1], [1, 0])
    spst = st.scattering_model.spst
    assert dr.allclose(spst.rcs_callables[0](k_i, k_s), 10.)

    co = float(np.sqrt(0.8))
    cross = float(np.sqrt(0.2))
    real, imag = spst.cpm_callables[0](k_i, k_s)
    assert dr.allclose(real, mi.Matrix2f(co, -cross, cross, co))
    assert dr.allclose(imag, mi.Matrix2f(0))

    # The parameters reach the model through the shared callables
    st.sigma = 40.
    st.xpr_db = None
    assert dr.allclose(spst.rcs_callables[0](k_i, k_s), 40.)
    real, imag = spst.cpm_callables[0](k_i, k_s)
    assert dr.allclose(real, dr.identity(mi.Matrix2f, 2))
    assert dr.allclose(imag, mi.Matrix2f(0))

    with pytest.raises(ValueError, match="non-negative"):
        st.sigma = -1.


def test_constant_rcs_target_clone_preserves_dimensions_and_rcs():
    st = ConstantRCSSensingTarget("st", sigma=10., xpr_db=3.,
                                  length=4., width=2., height=1.5,
                                  color=(1., 0.5, 0.), display_opacity=0.25)
    st.position = mi.Point3f(1., 2., 3.)
    st.orientation = mi.Point3f(0.1, 0.2, 0.3)
    st.velocity = mi.Vector3f(5., 0., 0.)

    clone = st.clone()

    assert isinstance(clone, ConstantRCSSensingTarget)
    assert clone.name == "st-clone"
    assert clone.dimensions == st.dimensions
    assert clone.scattering_model is st.scattering_model
    assert clone.rcs is st.rcs
    assert clone.cpm is st.cpm
    assert dr.allclose(clone.sigma, 10.)
    assert dr.allclose(clone.xpr_db, 3.)
    assert clone.mi_mesh is not st.mi_mesh
    assert dr.allclose(clone.position, st.position)
    assert dr.allclose(clone.orientation, st.orientation)
    assert dr.allclose(clone.velocity, st.velocity)
    assert np.allclose(clone.radio_material.color, (1., 0.5, 0.))
    assert clone.display_opacity == 0.25

    as_mesh = st.clone(name="st-mesh", as_mesh=True)
    assert isinstance(as_mesh, mi.Mesh)
    assert as_mesh.id() == "st-mesh"


def _expected_free_space_a(wavelength, sigma, r_1, r_2):
    """Channel coefficient of a free-space path scattered by a point of
    cross-section `sigma`, over legs of length `r_1` and `r_2`.

    The factors are composed as the electromagnetic primer does: the scaling
    `wavelength/(4*pi)` accounting for the effective area of the receive
    antenna applies once per path, the spreading factors `1/r` apply to every
    leg, and the scattering point re-radiates the power it intercepts over
    the sphere, which contributes `sqrt(sigma/(4*pi))`.
    """
    return wavelength/(4.*np.pi)*np.sqrt(sigma/(4.*np.pi))/(r_1*r_2)


# Height at which the cuboid of a target is held above its scattering points.
# A target does not occlude the scattering points it contains, so no elevation
# is needed. A nonzero elevation places the points outside of the cuboid, which
# then shadows them as any other object of the scene would.
_TARGET_ELEVATION = 0.
# Edge length of the cuboid representing a sensing target
_TARGET_SIZE = 0.2


def _add_sensing_target(scene, name, position, lcs_positions=None,
                        orientation=None, elevation=_TARGET_ELEVATION,
                        size=_TARGET_SIZE, rcs=None):
    """Adds to `scene` a cuboid sensing target whose scattering points all have
    a cross-section of one square meter and do not depolarize, unless `rcs` is
    specified.

    The scattering points end up at `position + rotation(orientation) @
    lcs_positions`, while the cuboid itself is held `elevation` above them.
    Lowering the scattering points by the elevation and raising the target by
    the same amount cancel out for any `orientation` around the z-axis, which
    leaves the elevation unchanged.
    """
    if lcs_positions is None:
        lcs_positions = mi.Point3f(0, 0, 0)
    lift = mi.Point3f(0, 0, elevation)

    model = ScatteringModel(mi.Point3f(lcs_positions) - lift,
                            rcs=rcs or _rcs_unit, cpm=ConstantCPM())
    target = SensingTarget(name, scattering_model=model, length=size,
                           width=size, height=size)

    scene.edit(add=target)
    target.position = mi.Point3f(position) + lift
    if orientation is not None:
        target.orientation = orientation

    return target


def _iso_scene(filename=None, **array_kwargs):
    """Scene with single-antenna isotropic arrays, empty by default."""
    scene = load_scene(filename)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, polarization="V",
                                 pattern="iso", **array_kwargs)
    scene.rx_array = scene.tx_array
    return scene


def test_scene_registers_sensing_targets_as_scene_objects():
    scene = load_scene()
    st = _add_sensing_target(scene, "st", mi.Point3f(1, 2, 3))

    # A sensing target is a scene object, and is also tracked as a target
    assert scene.objects == {"st": st}
    assert scene.sensing_targets == {"st": st}
    assert scene.get("st") is st
    # Its absorbing material was registered as well
    assert scene.get("st-material") is st.radio_material

    # Editing the scene keeps the instance held by the user
    other = _add_sensing_target(scene, "other", mi.Point3f(4, 0, 0))
    assert scene.sensing_targets == {"st": st, "other": other}

    scene.edit(remove="st")
    assert scene.objects == {"other": other}
    assert scene.sensing_targets == {"other": other}
    assert scene.get("st") is None


def test_scene_add_and_remove_sensing_targets():
    scene = load_scene()
    st = _cuboid_target()

    # Sensing targets are scene objects, and are added like any other one
    scene.add(st)
    assert scene.objects == {"st": st}
    assert scene.sensing_targets == {"st": st}

    # Adding the same target again is a no-op
    scene.add(st)
    assert scene.sensing_targets == {"st": st}

    scene.remove("st")
    assert not scene.objects
    assert not scene.sensing_targets


def test_sensing_target_pose_before_being_added():
    """A target can be posed at construction, and the pose survives the add."""
    target = SensingTarget("st", scattering_model=ScatteringModel(),
                           length=1., width=1., height=1.,
                           position=mi.Point3f(5., 6., 7.),
                           orientation=mi.Point3f(0.3, 0., 0.),
                           velocity=mi.Vector3f(1., 0., 0.))
    assert np.allclose(target.position.numpy()[:, 0], [5., 6., 7.], atol=1e-5)
    assert np.allclose(target.orientation.numpy()[:, 0], [0.3, 0., 0.])
    assert np.allclose(target.velocity.numpy()[:, 0], [1., 0., 0.])

    # The clone carries the pose of the original target over
    clone = target.clone(name="st-clone")
    assert np.allclose(clone.position.numpy(), target.position.numpy(),
                       atol=1e-5)
    assert np.allclose(clone.orientation.numpy(), target.orientation.numpy())
    assert np.allclose(clone.velocity.numpy(), target.velocity.numpy())

    scene = load_scene()
    scene.add([target, clone])
    assert scene.sensing_targets == {"st": target, "st-clone": clone}
    assert np.allclose(target.position.numpy()[:, 0], [5., 6., 7.], atol=1e-5)


def test_scene_rejects_sensing_target_name_clash():
    scene = load_scene()
    scene.add(Transmitter("device", position=mi.Point3f(0, 0, 0)))

    with pytest.raises(ValueError, match="already used"):
        scene.edit(add=_cuboid_target("device"))

    scene = load_scene()
    _add_sensing_target(scene, "st", mi.Point3f(1, 0, 0))
    with pytest.raises(ValueError, match="already used"):
        scene.add(Receiver("st", position=mi.Point3f(2, 0, 0)))


def test_solver_requires_sensing_targets():
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))

    with pytest.raises(ValueError, match="no sensing targets"):
        RCSSolver()(scene)

    # A target without any scattering point cannot scatter either
    scene.edit(add=SensingTarget("st", scattering_model=ScatteringModel(),
                                 length=1., width=1., height=1.))
    with pytest.raises(ValueError, match="no scattering points"):
        RCSSolver()(scene)


def test_solver_free_space_path():
    # Transmitter, scattering point, and receiver aligned along the x-axis, so
    # both legs are 10m long and the polarizations are aligned
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0))

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    a_real, a_imag = paths.a
    assert a_real.shape == (1, 1, 1, 1, 1)

    # Free-space propagation over both 10m legs, with a unit cross-section
    wavelength = scene.wavelength.numpy()[0]
    expected_a = _expected_free_space_a(wavelength, _UNIT_RCS, 10., 10.)
    assert np.allclose(a_real.numpy().squeeze(), expected_a, rtol=1e-4)
    assert np.allclose(a_imag.numpy().squeeze(), 0.)

    # Delay of the two legs
    assert np.allclose(paths.tau.numpy().squeeze(), 20./speed_of_light,
                       rtol=1e-5)
    assert np.allclose(paths.doppler.numpy().squeeze(), 0.)

    # The only interaction is the scattering event, which stores the indices of
    # the sensing target and of the scattering point within it
    assert paths.interactions.numpy().squeeze() == InteractionType.SENSING
    assert paths.objects.numpy().squeeze() == 0
    assert paths.primitives.numpy().squeeze() == 0
    assert np.allclose(paths.vertices.numpy().squeeze(), [10., 0., 0.])


@pytest.mark.parametrize("sigma", [0.1, 1., 13.3])
def test_solver_cross_section_normalization(sigma):
    # Same geometry as `test_solver_free_space_path`, but with a scattering
    # point of cross-section `sigma`. The channel coefficient must scale as
    # the square-root of the cross-section.
    def rcs(k_i, k_s, seed=0):
        return dr.full(mi.Float, sigma, dr.width(k_i))

    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0), rcs=rcs)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    a_real, a_imag = paths.a
    wavelength = scene.wavelength.numpy()[0]
    expected_a = _expected_free_space_a(wavelength, sigma, 10., 10.)
    assert np.allclose(a_real.numpy().squeeze(), expected_a, rtol=1e-4)
    assert np.allclose(a_imag.numpy().squeeze(), 0.)


def test_solver_scattering_points_are_placed_in_target_frame():
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    # The target is rotated by 90 degrees around the z-axis, so its scattering
    # point, which lies 4m ahead of it in its local frame, is 4m to its left
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0),
                        lcs_positions=mi.Point3f(4, 0, 0),
                        orientation=mi.Point3f(dr.pi/2, 0, 0))

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    assert np.allclose(paths.vertices.numpy().squeeze(), [10., 4., 0.],
                       atol=1e-5)
    expected_tau = 2.*np.sqrt(10.**2 + 4.**2)/speed_of_light
    assert np.allclose(paths.tau.numpy().squeeze(), expected_tau, rtol=1e-5)


def test_solver_scales_scattering_points_with_their_target():
    # Scaling a target resizes its mesh, and must move its scattering points
    # along with it, so that a point keeps its place relative to the shape it
    # describes
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))

    target = _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0),
                                 lcs_positions=mi.Point3f(0, 4, 0))
    target.scaling = mi.Vector3f(0.5)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    # The 4m offset of the point within its target is halved along with it
    assert np.allclose(paths.vertices.numpy().squeeze(), [10., 2., 0.],
                       atol=1e-5)
    expected_tau = 2.*np.sqrt(10.**2 + 2.**2)/speed_of_light
    assert np.allclose(paths.tau.numpy().squeeze(), expected_tau, rtol=1e-5)


def test_solver_doppler_accounts_for_device_mobility():
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0),
                          velocity=mi.Vector3f(3, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0),
                       velocity=mi.Vector3f(-2, 0, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0))

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    # The transmitter moves towards the scattering point and the receiver away
    # from it, both along the paths
    wavelength = scene.wavelength.numpy()[0]
    assert np.allclose(paths.doppler.numpy().squeeze(), 5./wavelength,
                       rtol=1e-5)


def test_solver_doppler_accounts_for_target_mobility():
    # Transmitter and receiver are collocated, so the target moves towards
    # them along both legs and its radial velocity is counted twice
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(0, 0, 0)))
    target = _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0))
    target.velocity = mi.Vector3f(-4, 0, 0)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    wavelength = scene.wavelength.numpy()[0]
    assert np.allclose(paths.doppler.numpy().squeeze(), 8./wavelength,
                       rtol=1e-5)


def test_solver_doppler_ignores_target_motion_along_the_path():
    # The target moves along the transmitter-receiver axis, so the incident
    # and scattered directions are identical, the length of the path is
    # constant, and no Doppler shift arises
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    target = _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0))
    target.velocity = mi.Vector3f(3, 0, 0)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    assert np.allclose(paths.doppler.numpy().squeeze(), 0., atol=1e-5)


def test_solver_doppler_sums_device_and_target_mobility():
    # The mobility of the target adds to the mobility of the devices, rather
    # than replacing it
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0),
                          velocity=mi.Vector3f(3, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0),
                       velocity=mi.Vector3f(-2, 0, 0)))
    target = _add_sensing_target(scene, "st", mi.Point3f(10, 0, 10))
    target.velocity = mi.Vector3f(0, 0, 6)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    # Both legs are unobstructed and make a 45 degree angle with the z-axis,
    # so the wave leaves the transmitter along (1,0,1)/sqrt(2) and reaches the
    # receiver along (1,0,-1)/sqrt(2)
    k_i = np.array([1., 0., 1.])/np.sqrt(2.)
    k_r = np.array([1., 0., -1.])/np.sqrt(2.)
    v_tx = np.array([3., 0., 0.])
    v_rx = np.array([-2., 0., 0.])
    expected = (np.dot(k_i, v_tx) - np.dot(k_r, v_rx)
                + np.dot(k_r - k_i, np.array([0., 0., 6.])))
    wavelength = scene.wavelength.numpy()[0]
    assert np.allclose(paths.doppler.numpy().squeeze(), expected/wavelength,
                       rtol=1e-5)


def test_solver_doppler_uses_the_velocity_of_every_target():
    # Two targets moving at different velocities, with collocated devices so
    # that the shift of a path is twice the radial velocity of its target.
    # The targets are held along two different axes, as they would otherwise
    # shadow each other.
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(0, 0, 0)))
    near = _add_sensing_target(scene, "near", mi.Point3f(5, 0, 0))
    far = _add_sensing_target(scene, "far", mi.Point3f(0, 10, 0))
    near.velocity = mi.Vector3f(-1, 0, 0)
    far.velocity = mi.Vector3f(0, 2, 0)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    # Every path is shifted according to the target it scatters off: the near
    # one approaches the devices, the far one recedes from them
    wavelength = scene.wavelength.numpy()[0]
    doppler = paths.doppler.numpy().squeeze()
    st_indices = paths.objects.numpy().squeeze()
    assert sorted(st_indices) == [0, 1]

    expected = {0: 2./wavelength, 1: -4./wavelength}
    for st_index, doppler_ in zip(st_indices, doppler):
        assert np.allclose(doppler_, expected[st_index], rtol=1e-5)


def test_solver_doppler_of_every_link_uses_its_own_geometry():
    # Two mobile transmitters, two mobile receivers, and two targets moving at
    # different velocities. Every link sees the two targets from its own
    # directions, so the shift of a path depends on both the link and the
    # target it scatters off.
    tx_positions = np.array([[0., 0., 0.], [-4., 6., 0.]])
    rx_positions = np.array([[20., 0., 0.], [18., -7., 0.]])
    st_positions = np.array([[10., 3., 0.], [7., -2., 0.]])
    tx_velocities = np.array([[3., 0., 0.], [0., -5., 0.]])
    rx_velocities = np.array([[-2., 1., 0.], [4., 0., 0.]])
    st_velocities = np.array([[0., 6., 0.], [-1., -3., 0.]])

    scene = _iso_scene()
    for i, (position, velocity) in enumerate(zip(tx_positions,
                                                 tx_velocities)):
        scene.add(Transmitter(f"tx{i}",
                              position=mi.Point3f(*position.tolist()),
                              velocity=mi.Vector3f(*velocity.tolist())))
    for i, (position, velocity) in enumerate(zip(rx_positions,
                                                 rx_velocities)):
        scene.add(Receiver(f"rx{i}", position=mi.Point3f(*position.tolist()),
                           velocity=mi.Vector3f(*velocity.tolist())))
    for i, (position, velocity) in enumerate(zip(st_positions,
                                                 st_velocities)):
        target = _add_sensing_target(scene, f"st{i}",
                                     mi.Point3f(*position.tolist()))
        target.velocity = mi.Vector3f(*velocity.tolist())

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    # One path per pair of devices and per target
    assert paths.tau.shape == (2, 2, 2)
    assert paths.valid.numpy().all()

    wavelength = scene.wavelength.numpy()[0]
    doppler = paths.doppler.numpy()
    st_indices = paths.objects.numpy()[0]
    for rx in range(2):
        for tx in range(2):
            assert sorted(st_indices[rx, tx]) == [0, 1]
            for p in range(2):
                st = st_indices[rx, tx, p]
                # Both legs are unobstructed, so the wave leaves the
                # transmitter towards the target and reaches the receiver from
                # it
                k_i = st_positions[st] - tx_positions[tx]
                k_i /= np.linalg.norm(k_i)
                k_r = rx_positions[rx] - st_positions[st]
                k_r /= np.linalg.norm(k_r)
                expected = (np.dot(k_i, tx_velocities[tx])
                            - np.dot(k_r, rx_velocities[rx])
                            + np.dot(k_r - k_i, st_velocities[st]))/wavelength
                assert np.allclose(doppler[rx, tx, p], expected, rtol=1e-5)

    # The shifts do tell the links and the targets apart, i.e., the check
    # above is not satisfied by a single shift shared by all the paths
    assert len(np.unique(np.round(doppler, 3))) == doppler.size


def test_solver_pairs_legs_of_the_same_scattering_point():
    # Two transmitters, three receivers, and two targets holding three
    # scattering points in total. The numbers of transmitters and of receivers
    # differ, so a transposition of the two dimensions cannot go unnoticed.
    scene = _iso_scene()
    scene.add(Transmitter("tx0", position=mi.Point3f(0, 0, 0)))
    scene.add(Transmitter("tx1", position=mi.Point3f(0, 5, 0)))
    scene.add(Receiver("rx0", position=mi.Point3f(20, 0, 0)))
    scene.add(Receiver("rx1", position=mi.Point3f(20, 5, 0)))
    scene.add(Receiver("rx2", position=mi.Point3f(20, -6, 0)))
    _add_sensing_target(scene, "st0", mi.Point3f(10, 0, 0),
                        lcs_positions=mi.Point3f([0, 1], [0, 2], [0, 0]))
    _add_sensing_target(scene, "st1", mi.Point3f(10, 8, 0))

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    # One path per pair of devices and per scattering point
    tau = paths.tau.numpy()
    assert tau.shape == (3, 2, 3)

    valid = paths.valid.numpy()
    assert valid.all()

    # Every path scatters once, off a scattering point of one of the two
    # targets
    interactions = paths.interactions.numpy()
    assert interactions.shape == (1, 3, 2, 3)
    assert (interactions == InteractionType.SENSING).all()

    st_indices = paths.objects.numpy()[0]
    spst_indices = paths.primitives.numpy()[0]
    # The paths of a given pair of devices scatter off all three points, i.e.,
    # points 0 and 1 of target 0 and point 0 of target 1
    for rx in range(3):
        for tx in range(2):
            assert sorted(zip(st_indices[rx, tx], spst_indices[rx, tx])) \
                   == [(0, 0), (0, 1), (1, 0)]

    # Delays and coefficients are consistent with the geometry of the two legs
    tx_positions = np.array([[0., 0., 0.], [0., 5., 0.]])
    rx_positions = np.array([[20., 0., 0.], [20., 5., 0.], [20., -6., 0.]])
    # Positions of the scattering points of every target
    spst_positions = [np.array([[10., 0., 0.], [11., 2., 0.]]),
                      np.array([[10., 8., 0.]])]
    vertices = paths.vertices.numpy()[0]
    a_real, a_imag = paths.a
    a = np.abs(a_real.numpy() + 1j*a_imag.numpy())
    wavelength = scene.wavelength.numpy()[0]
    for rx in range(3):
        for tx in range(2):
            for p in range(3):
                spst = spst_positions[st_indices[rx, tx, p]][
                    spst_indices[rx, tx, p]]
                assert np.allclose(vertices[rx, tx, p], spst, atol=1e-5)
                r_1 = np.linalg.norm(spst - tx_positions[tx])
                r_2 = np.linalg.norm(rx_positions[rx] - spst)
                assert np.allclose(tau[rx, tx, p],
                                   (r_1 + r_2)/speed_of_light, rtol=1e-5)
                # Every link sees the path of its own two legs. The amplitudes
                # are small, so the comparison is made purely relative rather
                # than leaving `np.allclose` fall back on its absolute
                # tolerance.
                expected_a = _expected_free_space_a(wavelength, _UNIT_RCS,
                                                    r_1, r_2)
                assert np.allclose(a[rx, 0, tx, 0, p], expected_a, rtol=1e-4,
                                   atol=0.)


def test_solver_synthetic_and_non_synthetic_arrays_agree():
    # A single-antenna array leaves no room for synthetic phase shifts, so both
    # modes must produce the same paths
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 3, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, -4, 2))

    solver = RCSSolver()
    synthetic = solver(scene, max_depth=1, samples_per_sp=10**4,
                       synthetic_array=True)
    non_synthetic = solver(scene, max_depth=1, samples_per_sp=10**4,
                           synthetic_array=False)

    for i in range(2):
        assert np.allclose(synthetic.a[i].numpy(),
                           non_synthetic.a[i].numpy(), rtol=1e-5)
    assert np.allclose(synthetic.tau.numpy().squeeze(),
                       non_synthetic.tau.numpy().squeeze(), rtol=1e-6)


def test_solver_multi_antenna_arrays():
    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1, num_cols=2, polarization="V",
                                 pattern="iso")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=4, polarization="V",
                                 pattern="iso")
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0))

    solver = RCSSolver()
    for synthetic_array in (True, False):
        paths = solver(scene, max_depth=1, samples_per_sp=10**4,
                       synthetic_array=synthetic_array)
        a_real, a_imag = paths.a
        assert a_real.shape == (1, 4, 1, 2, 1)
        assert a_imag.shape == (1, 4, 1, 2, 1)
        # Every antenna pair sees the scattered path
        assert (np.abs(a_real.numpy()) > 0.).all()


def test_solver_multi_device_arrays_use_per_antenna_positions():
    # Three transmitters carrying a four-antenna array and two receivers
    # carrying a five-antenna array, so that the four dimensions of the
    # coefficients have distinct sizes and no permutation of them can go
    # unnoticed.
    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1, num_cols=4, polarization="V",
                                 pattern="iso")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=5, polarization="V",
                                 pattern="iso")

    # The antennas of a one-row array are spread along the y-axis of their
    # device, and the target lies in the +y direction, so the antennas of a
    # device are at measurably different distances from it. Everything stays
    # in the z = 0 plane, in which the vertical polarizations are aligned.
    tx_positions = np.array([[0., 0., 0.], [4., 0., 0.], [-5., 0., 0.]])
    rx_positions = np.array([[1., 2., 0.], [-2., -1., 0.]])
    for i, position in enumerate(tx_positions):
        scene.add(Transmitter(f"tx{i}",
                              position=mi.Point3f(*position.tolist())))
    for i, position in enumerate(rx_positions):
        scene.add(Receiver(f"rx{i}", position=mi.Point3f(*position.tolist())))
    _add_sensing_target(scene, "st", mi.Point3f(0, 30, 0))

    wavelength = scene.wavelength.numpy()[0]
    solver = RCSSolver()

    def amplitudes(paths):
        a_real, a_imag = paths.a
        assert a_real.shape == (2, 5, 3, 4, 1)
        return np.abs(a_real.numpy() + 1j*a_imag.numpy()).squeeze(-1)

    # Without synthetic arrays, every antenna is a source or a target of the
    # traced paths in its own right, and the two legs of a path are measured
    # from the antennas rather than from the devices
    paths = solver(scene, max_depth=1, samples_per_sp=10**4,
                   synthetic_array=False)
    assert paths.tau.shape == (2, 5, 3, 4, 1)
    assert paths.valid.numpy().all()

    # Antennas are ordered antenna-first within their device
    tx_ant_positions = paths.sources.numpy().T.reshape(3, 4, 3)
    rx_ant_positions = paths.targets.numpy().T.reshape(2, 5, 3)
    spst = paths.vertices.numpy()[0, 0, 0, 0, 0]
    assert np.allclose(spst, [0., 30., 0.], atol=1e-5)

    # The amplitudes are of the order of 1e-6, so the comparisons below are
    # made purely relative rather than leaving `np.allclose` fall back on its
    # absolute tolerance
    a = amplitudes(paths)
    r_1 = np.linalg.norm(spst - tx_ant_positions, axis=-1)
    r_2 = np.linalg.norm(rx_ant_positions - spst, axis=-1)
    expected_a = _expected_free_space_a(wavelength, _UNIT_RCS,
                                        r_1[None, None, :, :],
                                        r_2[:, :, None, None])
    assert np.allclose(a, expected_a, rtol=1e-4, atol=0.)

    # The antennas of a device do see the target from distances that differ by
    # more than the tolerance above, on both sides of the link, so that check
    # is not satisfied by amplitudes computed from the device positions
    for legs in (r_1, r_2):
        assert (legs.max(axis=-1)/legs.min(axis=-1) - 1. > 1e-3).all()

    # With synthetic arrays, the devices are modelled as single-antenna ones
    # and the array is accounted for by phase shifts, which leave the
    # amplitudes of a device pair identical
    paths = solver(scene, max_depth=1, samples_per_sp=10**4,
                   synthetic_array=True)
    assert paths.tau.shape == (2, 3, 1)
    assert paths.valid.numpy().all()

    a = amplitudes(paths)
    r_1 = np.linalg.norm(spst - tx_positions, axis=-1)
    r_2 = np.linalg.norm(rx_positions - spst, axis=-1)
    expected_a = _expected_free_space_a(wavelength, _UNIT_RCS,
                                        r_1[None, None, :, None],
                                        r_2[:, None, None, None])
    assert np.allclose(a, expected_a, rtol=1e-4, atol=0.)


def test_solver_max_depth_counts_the_sensing_interaction():
    # Transmitter, scattering point, and receiver above a small flat
    # reflector, so that each leg is either unobstructed or reflected once and
    # the paths have a depth of one, two, or three
    scene = _iso_scene(sionna.rt.scene.simple_reflector)
    scene.add(Transmitter("tx", position=mi.Point3f(-0.1, -0.08, 1)))
    scene.add(Receiver("rx", position=mi.Point3f(-0.1, 0.36, 1)))
    # The scene is small, so the target is small too
    _add_sensing_target(scene, "st", mi.Point3f(0.3, -0.2, 1), size=0.01)

    solver = RCSSolver()
    for max_depth, expected_depths in ((1, [1]),
                                       (2, [1, 2, 2]),
                                       (3, [1, 2, 2, 3])):
        paths = solver(scene, max_depth=max_depth, samples_per_sp=10**5,
                       refraction=False)

        interactions = paths.interactions.numpy()
        assert interactions.shape[0] == max_depth

        depths = (interactions != InteractionType.NONE).sum(axis=0)
        assert sorted(depths.flatten()) == expected_depths


def test_deterministic_solver_finds_the_same_paths():
    # The same geometry, solved by the deterministic candidate generator. The
    # scattering point is held within its target, whose shape must be excluded
    # from the legs of the paths by both generators alike.
    scene = _iso_scene(sionna.rt.scene.simple_reflector)
    scene.add(Transmitter("tx", position=mi.Point3f(-0.1, -0.08, 1)))
    scene.add(Receiver("rx", position=mi.Point3f(-0.1, 0.36, 1)))
    _add_sensing_target(scene, "st", mi.Point3f(0.3, -0.2, 1), size=0.01)

    kwargs = {"max_depth": 3, "samples_per_sp": 10**5, "refraction": False}
    expected = RCSSolver()(scene, **kwargs)
    paths = RCSSolver(deterministic=True)(scene, **kwargs)

    # The same paths are found, up to their order which depends on the order in
    # which the samples reach the paths buffer
    assert paths.tau.shape == expected.tau.shape
    assert np.allclose(np.sort(paths.tau.numpy(), axis=-1),
                       np.sort(expected.tau.numpy(), axis=-1), rtol=1e-6)
    power = sum(np.sum(a.numpy()**2) for a in paths.a)
    expected_power = sum(np.sum(a.numpy()**2) for a in expected.a)
    assert np.allclose(power, expected_power, rtol=1e-5)


def test_deterministic_solver_sensing_targets_shadow_each_other():
    # The deterministic generator shares the occlusion semantics of the random
    # one: each of the two targets lies on a leg of the other one
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "near", mi.Point3f(5, 0, 0))
    _add_sensing_target(scene, "far", mi.Point3f(10, 0, 0))

    paths = RCSSolver(deterministic=True)(scene, max_depth=1,
                                          samples_per_sp=10**4)

    assert paths.tau.shape == (1, 1, 0)

    # Holding the targets off each other legs brings both paths back
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "left", mi.Point3f(10, 3, 0))
    _add_sensing_target(scene, "right", mi.Point3f(10, -3, 0))

    paths = RCSSolver(deterministic=True)(scene, max_depth=1,
                                          samples_per_sp=10**4)

    assert paths.tau.shape == (1, 1, 2)
    assert paths.valid.numpy().all()


def test_solver_rejects_zero_max_depth():
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0))

    # The scattering event on the sensing target is always one interaction
    with pytest.raises(ValueError, match="`max_depth` must be greater"):
        RCSSolver()(scene, max_depth=0)


def test_solver_returns_no_paths_when_no_leg_can_be_traced():
    # The line-of-sight legs are disabled, and the only object of the scene is
    # the target itself, which is excluded from the traced geometry. No leg can
    # therefore be traced.
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0))

    paths = RCSSolver()(scene, max_depth=2, los=False,
                        samples_per_sp=10**4)

    a_real, a_imag = paths.a
    assert a_real.shape == (1, 1, 1, 1, 0)
    assert a_imag.shape == (1, 1, 1, 1, 0)
    assert paths.tau.shape == (1, 1, 0)
    # The depth dimension matches `max_depth` even without any path
    assert paths.interactions.shape == (2, 1, 1, 0)


@pytest.mark.parametrize("hidden", ("tx", "rx"))
def test_solver_returns_no_paths_when_a_device_is_unreachable(hidden):
    # The legs of a path reach one of the two devices only. The scattering
    # point is held outside of a closed absorbing box in which the unreachable
    # device is locked, while the other device is left outside of the box.
    scene = _iso_scene(sionna.rt.scene.box)
    box = list(scene.objects.values())[0]
    box.radio_material = AbsorberRadioMaterial("absorber")
    inside = mi.Point3f(0, 0, 2.5)
    outside = mi.Point3f(0, 0, 20)

    scene.add(Transmitter("tx", position=inside if hidden == "tx" else outside))
    scene.add(Receiver("rx", position=inside if hidden == "rx" else outside))
    _add_sensing_target(scene, "st", mi.Point3f(0, 0, 30))

    paths = RCSSolver()(scene, max_depth=2, samples_per_sp=10**4)

    assert paths.tau.shape == (1, 1, 0)
    assert paths.a[0].shape == (1, 1, 1, 1, 0)


def test_solver_occlusion_of_a_device_leaves_the_other_links_intact():
    # Two receivers, one locked in the closed absorbing box and one left
    # outside of it with the transmitter. The target is held outside of the
    # box as well, so only the receiver outside of it is reached, and the
    # blocked link must be reported as such rather than dropping the path of
    # the other link.
    scene = _iso_scene(sionna.rt.scene.box)
    box = list(scene.objects.values())[0]
    box.radio_material = AbsorberRadioMaterial("absorber")

    # The box spans [-5, 5] x [-5, 5] x [0, 5], so the devices above it and
    # the target see each other along directions that are neither vertical nor
    # obstructed
    tx_position = np.array([0., 0., 20.])
    rx_position = np.array([10., 0., 20.])
    st_position = np.array([5., 0., 30.])
    scene.add(Transmitter("tx", position=mi.Point3f(*tx_position.tolist())))
    scene.add(Receiver("rx-blocked", position=mi.Point3f(0, 0, 2.5)))
    scene.add(Receiver("rx-open", position=mi.Point3f(*rx_position.tolist())))
    _add_sensing_target(scene, "st", mi.Point3f(*st_position.tolist()))

    paths = RCSSolver()(scene, max_depth=2, samples_per_sp=10**4)

    # The buffer holds the single path of the reachable link, and the blocked
    # link is padded with an invalid one
    assert paths.tau.shape == (2, 1, 1)
    assert list(paths.valid.numpy().squeeze()) == [False, True]

    # The surviving path is the one of the open link, i.e., it was not shifted
    # into the row of the blocked receiver
    expected_tau = (np.linalg.norm(st_position - tx_position)
                    + np.linalg.norm(rx_position - st_position)) \
                   /speed_of_light
    assert np.allclose(paths.tau.numpy()[1].squeeze(), expected_tau, rtol=1e-5)
    assert np.abs(paths.a[0].numpy()[1]).squeeze() > 0.
    assert np.allclose(paths.a[0].numpy()[0], 0.)
    assert np.allclose(paths.a[1].numpy()[0], 0.)


def test_solver_ignores_the_shape_of_a_sensing_target():
    # The scattering point sits at the center of the cuboid of its target,
    # which would occlude it if the target occluded the points it contains
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0), size=2.)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    # Same free-space path as if the target had no shape at all
    a_real, a_imag = paths.a
    wavelength = scene.wavelength.numpy()[0]
    expected_a = _expected_free_space_a(wavelength, _UNIT_RCS, 10., 10.)
    assert np.allclose(a_real.numpy().squeeze(), expected_a, rtol=1e-4)
    assert np.allclose(a_imag.numpy().squeeze(), 0.)
    assert np.allclose(paths.tau.numpy().squeeze(), 20./speed_of_light,
                       rtol=1e-5)


def test_solver_sensing_targets_shadow_each_other():
    # Two targets between the transmitter and the receiver, both holding a
    # scattering point at their center. Each target lies on a leg of the other
    # one, so neither of the two paths survives.
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "near", mi.Point3f(5, 0, 0))
    _add_sensing_target(scene, "far", mi.Point3f(10, 0, 0))

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    assert paths.tau.shape == (1, 1, 0)


def test_solver_sensing_targets_off_each_other_legs_both_scatter():
    # The same two targets, now held on either side of the axis joining the
    # devices so that neither of them lies on a leg of the other one
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    _add_sensing_target(scene, "left", mi.Point3f(10, 3, 0))
    _add_sensing_target(scene, "right", mi.Point3f(10, -3, 0))

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    # One path per target, both scattering off the point of their target
    assert paths.tau.shape == (1, 1, 2)
    assert paths.valid.numpy().all()
    assert sorted(paths.objects.numpy().squeeze()) == [0, 1]

    # The two scattering points are symmetric about the axis joining the
    # devices, so the two paths have the same length
    expected_tau = 2.*np.sqrt(10.**2 + 3.**2)/speed_of_light
    assert np.allclose(paths.tau.numpy().squeeze(), expected_tau, rtol=1e-5)


def test_solver_sensing_target_shadows_a_scattering_point_outside_of_it():
    # A scattering point 1m below the cuboid of its target, i.e., outside of
    # it, with both devices held above the target. The cuboid stands between
    # the point and the devices, and is not excluded from their legs.
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(5, 0, 20)))
    scene.add(Receiver("rx", position=mi.Point3f(15, 0, 20)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0), elevation=2.,
                        size=2.)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    assert paths.tau.shape == (1, 1, 0)

    # The very same geometry, with the scattering point moved inside the
    # cuboid, does produce the path
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(5, 0, 20)))
    scene.add(Receiver("rx", position=mi.Point3f(15, 0, 20)))
    _add_sensing_target(scene, "st", mi.Point3f(10, 0, 2), size=2.)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    assert paths.tau.shape == (1, 1, 1)
    assert paths.valid.numpy().all()


def test_solver_scattering_point_on_a_face_keeps_its_outward_legs():
    # A scattering point at the center of the roof of its target, i.e., exactly
    # on one of the faces of its cuboid. The legs which leave that face must
    # survive, including when they graze it, as the scattering points of the
    # TR38901 targets are placed this way.
    size = 2.
    for device_height in (10., 2.001):
        scene = _iso_scene()
        scene.add(Transmitter("tx", position=mi.Point3f(0, 0, device_height)))
        scene.add(Receiver("rx", position=mi.Point3f(20, 0, device_height)))
        _add_sensing_target(scene, "st", mi.Point3f(10, 0, 1),
                            lcs_positions=mi.Point3f(0, 0, 0.5*size),
                            size=size)

        paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

        # The scattering point sits at the center of the roof, 2m above the
        # ground, and the devices are 20m apart
        assert paths.tau.shape == (1, 1, 1), device_height
        assert paths.valid.numpy().all()
        leg = np.sqrt(10.**2 + (device_height - 2.)**2)
        assert np.allclose(paths.tau.numpy().squeeze(),
                           2.*leg/speed_of_light, rtol=1e-5)


def test_solver_reaches_a_device_within_the_box_of_a_target():
    # A receiver inside the cuboid of a target holding a scattering point at
    # its center. The leg towards that receiver never leaves the box, so its
    # origin is not advanced, and nothing occludes it.
    rx_offset = np.array([0.5, 0.2, 0.2])
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(10.5, 0.2, 0.2)))
    target = _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0), size=2.)

    # The receiver is within the box of the target
    assert np.all(np.abs(rx_offset)
                  <= target.lcs_half_extents.numpy().squeeze())

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    assert paths.tau.shape == (1, 1, 1)
    assert paths.valid.numpy().all()

    # The leg to the receiver is as long as the distance to it, i.e., the ray
    # was not reversed by an advance of its origin out of the box
    expected_tau = (10. + np.linalg.norm(rx_offset))/speed_of_light
    assert np.allclose(paths.tau.numpy().squeeze(), expected_tau, rtol=1e-5)

    # The wave reaches the receiver from the direction of the scattering
    # point, and not from the opposite one as it would if the ray had been
    # reversed. Angles of arrival point from the device back towards the
    # scattering point.
    theta_r = paths.theta_r.numpy().squeeze()
    phi_r = paths.phi_r.numpy().squeeze()
    k_r = np.array([np.sin(theta_r)*np.cos(phi_r),
                    np.sin(theta_r)*np.sin(phi_r),
                    np.cos(theta_r)])
    assert np.allclose(k_r, -rx_offset/np.linalg.norm(rx_offset), atol=1e-5)


def test_solver_shape_of_a_target_occludes_within_its_own_box():
    # The same configuration, with a target shaped as a sphere, whose mesh is
    # far from filling its bounding box. The receiver sits within the box and
    # outside of the sphere, so the leg towards it has to cross the sphere:
    # the box only decides where the leg starts, not whether it is occluded.
    rx_offset = np.array([0.7, 0.7, 0.7])
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(10.7, 0.7, 0.7)))

    model = ScatteringModel(mi.Point3f(0, 0, 0), rcs=_rcs_unit,
                            cpm=ConstantCPM())
    target = SensingTarget("st", scattering_model=model,
                           fname=sionna.rt.scene.sphere)
    scene.edit(add=target)
    target.position = mi.Point3f(10, 0, 0)

    # The receiver is within the box of the target, and outside of its sphere
    assert np.all(np.abs(rx_offset)
                  <= target.lcs_half_extents.numpy().squeeze())
    assert np.linalg.norm(rx_offset) > 1.

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    assert paths.tau.shape == (1, 1, 0)


def test_solver_shape_of_a_target_occludes_a_vertex_within_its_own_box():
    # A sphere target straddling the reflector, so that the reflector runs
    # through the bounding box of the target. The specular reflection point
    # sits within that box, below the sphere, so the leg which joins it to the
    # scattering point has to cross the sphere and is occluded by it. The
    # candidate generator only has to sample the right triangle: the image
    # method then moves the vertex to the specular point, which is how that
    # point can lie within the box at all.
    scene = _iso_scene(sionna.rt.scene.simple_reflector)
    scene.add(Transmitter("tx", position=mi.Point3f(-1.5, 0, 0.5)))
    scene.add(Receiver("rx", position=mi.Point3f(0.77, 0, 0.5)))

    model = ScatteringModel(mi.Point3f(0, 0, 0), rcs=_rcs_unit,
                            cpm=ConstantCPM())
    target = SensingTarget("st", scattering_model=model,
                           fname=sionna.rt.scene.sphere)
    scene.edit(add=target)
    target.scaling = mi.Vector3f(0.3)
    target.position = mi.Point3f(0, 0, 0.27)

    # The specular reflection point of the leg towards the receiver does lie
    # within the box of the target, and below its sphere
    sp = np.array([0., 0., 0.27])
    reflection = np.array([0.27, 0., 0.])
    assert np.all(np.abs(reflection - sp)
                  <= target.lcs_half_extents.numpy().squeeze())

    paths = RCSSolver()(scene, max_depth=2, samples_per_sp=10**6,
                        refraction=False)

    # Only the direct path remains: the reflected one would have to leak
    # through the sphere
    direct = (np.linalg.norm(sp - np.array([-1.5, 0., 0.5]))
              + np.linalg.norm(np.array([0.77, 0., 0.5]) - sp))
    assert paths.tau.shape == (1, 1, 1)
    assert paths.valid.numpy().all()
    assert np.allclose(paths.tau.numpy().squeeze(), direct/speed_of_light,
                       rtol=1e-4)


def test_solver_does_not_edit_the_scene():
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(20, 0, 0)))
    target = _add_sensing_target(scene, "st", mi.Point3f(10, 0, 0))

    mi_scene = scene.mi_scene
    RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    # The scene is left untouched, and is traced as is
    assert scene.mi_scene is mi_scene
    assert scene.objects == {"st": target}
    assert scene.sensing_targets == {"st": target}
    assert any(s.id() == target.mi_mesh.id() for s in scene.mi_scene.shapes())
