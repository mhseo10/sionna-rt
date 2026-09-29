#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the RCS models of 3GPP TR 38.901, clause 7.9.2."""

import pytest
import drjit as dr
import mitsuba as mi
import numpy as np

import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.rt.rcs import (RCSSolver, TR38901RCS, TR38901CPM,
                           TR38901ScatteringModel, TR38901SensingTarget)
from sionna.rt.rcs.tr38901 import (OBJECT_TYPES, TARGET_PARAMETERS,
                                   get_target_parameters)
from sionna.rt.utils import r_hat

# Object types and RCS models the specifications define, as
# `(object_type, model_type)`
_COMBINATIONS = sorted(TARGET_PARAMETERS)
# Those of them using RCS model 2, i.e., having lobes
_MODEL_2_OBJECT_TYPES = [object_type for object_type, model in _COMBINATIONS
                         if model == 2]
# Single and multiple scattering point variants of the same target
_VARIANT_PAIRS = [("vehicle-single-sp", "vehicle-multi-sp"),
                  ("agv-single-sp", "agv-multi-sp")]


def _target(object_type, model_type=2, **kwargs):
    """Builds a TR 38.901 target with a unique name."""
    return TR38901SensingTarget("st", object_type, model_type, **kwargs)


def _model(object_type, model_type=2, **kwargs):
    """Returns the internally-created model of a TR 38.901 target."""
    return _target(object_type, model_type, **kwargs).scattering_model


def _standalone_model(object_type, model_type=2, **kwargs):
    """Builds a model directly, rather than through a target, so that its
    flags can be set at construction.
    """
    positions = _model(object_type, model_type).spst.lcs_positions
    return TR38901ScatteringModel(object_type, model_type, positions, **kwargs)


def _direction(theta_deg, phi_deg):
    """Unit vector pointing away from a scattering point."""
    return r_hat(mi.Float(np.radians(theta_deg)),
                 mi.Float(np.radians(phi_deg)))


def _propagation_directions(theta_deg, phi_deg, beta_deg=0.):
    r"""Directions of propagation `(k_i, k_s)` for a bisector at
    `(theta_deg, phi_deg)` and a bistatic angle of `beta_deg`.

    The incident and scattered rays are tilted away from the bisector by half
    the bistatic angle, within the plane containing the bisector and the
    z-axis, or the x-axis for a bisector along the z-axis.
    """
    bisector = _direction(theta_deg, phi_deg)
    axis = mi.Vector3f(1., 0., 0.) if abs(np.sin(np.radians(theta_deg))) < 1e-6 \
        else mi.Vector3f(0., 0., 1.)
    tangent = dr.normalize(dr.cross(dr.cross(bisector, axis), bisector))

    half_beta = 0.5*np.radians(beta_deg)
    d_i = np.cos(half_beta)*bisector + np.sin(half_beta)*tangent
    d_s = np.cos(half_beta)*bisector - np.sin(half_beta)*tangent
    # The solver passes directions of propagation, which are the opposite of
    # the incident direction of the specifications
    return -d_i, d_s


def _sigma_md_db(rcs, theta_deg, phi_deg, beta_deg=0.):
    r"""Evaluates `10*lg(sigma_M*sigma_D)` [dBsm] for a bisector at
    `(theta_deg, phi_deg)` and a bistatic angle of `beta_deg`.
    """
    k_i, k_s = _propagation_directions(theta_deg, phi_deg, beta_deg)
    return np.array(rcs.sigma_md_db(k_i, k_s))[0]


def _single_lobe_rcs(object_type, lobe):
    """RCS callable holding a single lobe of a target, as tabulated."""
    parameters = get_target_parameters(object_type, 2)
    return TR38901RCS([lobe], parameters.k1, parameters.k2,
                      parameters.sigma_m_db)


def _whole_table_rcs(object_type, model_type=2, **kwargs):
    """RCS callable holding every lobe of a target, as tabulated."""
    parameters = get_target_parameters(object_type, model_type)
    return TR38901RCS(parameters.lobes, parameters.k1, parameters.k2,
                      parameters.sigma_m_db,
                      sigma_s_std_db=parameters.sigma_s_db, **kwargs)


def _table_cpm(object_type, model_type=2, **kwargs):
    """CPM callable holding the XPR of a target, as tabulated."""
    parameters = get_target_parameters(object_type, model_type)
    return TR38901CPM(parameters.xpr_mean_db,
                      xpr_std_db=parameters.xpr_std_db, **kwargs)


def _random_directions(num_samples, seed=1234):
    """Pairs of independent random directions of propagation, drawn over the
    sphere.
    """
    rng = np.random.default_rng(seed)

    def directions():
        vectors = rng.normal(size=(3, num_samples))
        vectors /= np.linalg.norm(vectors, axis=0)
        return mi.Vector3f(*[mi.Float(row) for row in vectors])

    return directions(), directions()


def _matrices(cpm, k_i, k_s, seed=1):
    """Evaluates a CPM and returns its matrices as a complex numpy array of
    shape `(2, 2, num_samples)`.
    """
    real, imag = cpm(k_i, k_s, seed)
    return np.array(real).reshape(2, 2, -1) \
        + 1j*np.array(imag).reshape(2, 2, -1)


def _lobe_boresight(lobe):
    """Bisector angles of the boresight of a lobe, as `(theta, phi)` [deg]."""
    return (lobe.theta_center,
            0. if lobe.phi_center is None else lobe.phi_center)


def _iso_scene():
    """Empty scene with single-antenna, V-polarized isotropic arrays."""
    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, polarization="V",
                                 pattern="iso")
    scene.rx_array = scene.tx_array
    return scene


# ------------------------------------------------------------------------
# Values of the RCS
# ------------------------------------------------------------------------

@pytest.mark.parametrize("object_type", _MODEL_2_OBJECT_TYPES)
def test_rcs_at_boresight_equals_g_max(object_type):
    # On the boresight of a lobe and for monostatic backscatter, the angular
    # and bistatic terms of eq. 7.9.2-3 all vanish
    for lobe in get_target_parameters(object_type, 2).lobes:
        theta, phi = _lobe_boresight(lobe)
        got = _sigma_md_db(_single_lobe_rcs(object_type, lobe), theta, phi)
        assert got == pytest.approx(lobe.g_max, abs=1e-4)


@pytest.mark.parametrize("object_type", _MODEL_2_OBJECT_TYPES)
def test_rcs_reaches_the_floor_for_forward_scattering(object_type):
    # The `5*log10(cos(beta/2))` term of eq. 7.9.2-3 diverges at
    # `beta = 180` degrees, where the `G_max - sigma_max` floor governs
    for lobe in get_target_parameters(object_type, 2).lobes:
        theta, phi = _lobe_boresight(lobe)
        got = _sigma_md_db(_single_lobe_rcs(object_type, lobe), theta, phi,
                           beta_deg=180.)
        assert got == pytest.approx(lobe.floor, abs=1e-4)


@pytest.mark.parametrize("object_type", _MODEL_2_OBJECT_TYPES)
def test_rcs_reaches_the_floor_far_off_boresight(object_type):
    # `sigma_V_dB` and `sigma_H_dB` are both clipped at `sigma_max`, so a
    # direction far enough from the boresight of a lobe reaches the floor
    for lobe in get_target_parameters(object_type, 2).lobes:
        theta, phi = _lobe_boresight(lobe)
        if lobe.azimuth_dependent:
            # Half a turn away in azimuth
            phi += 180.
        else:
            # The roof and bottom lobes have no azimuth dependence, so the
            # zenith angle is taken to the opposite pole instead
            theta = 180. - theta
        got = _sigma_md_db(_single_lobe_rcs(object_type, lobe), theta, phi)
        assert got == pytest.approx(lobe.floor, abs=1e-3)


@pytest.mark.parametrize("object_type", _MODEL_2_OBJECT_TYPES)
def test_rcs_never_falls_below_the_floor(object_type):
    rcs = _whole_table_rcs(object_type)
    lobes = get_target_parameters(object_type, 2).lobes
    floor = min(lobe.floor for lobe in lobes)

    rng = np.random.default_rng(42)
    theta = np.degrees(np.arccos(rng.uniform(-1., 1., 256)))
    phi = rng.uniform(-180., 540., 256)
    for beta in (0., 45., 90., 135., 179.):
        got = np.array([_sigma_md_db(rcs, t, p, beta)
                        for t, p in zip(theta, phi)])
        assert np.all(got >= floor - 1e-4)
        assert np.all(got <= max(lobe.g_max for lobe in lobes) + 1e-4)


def test_model_1_has_no_aspect_dependence():
    # RCS model 1 fixes the angular component `sigma_D` to 1, so eq. 7.9.2-2
    # only keeps the dependence on the bistatic angle
    for object_type in ("uav-small-size", "human"):
        parameters = get_target_parameters(object_type, 1)
        rcs = _whole_table_rcs(object_type, model_type=1)
        assert parameters.lobes == ()

        # Rotating the target changes nothing
        for theta, phi in ((90., 0.), (90., 123.), (12., 250.), (180., 0.)):
            got = _sigma_md_db(rcs, theta, phi)
            assert got == pytest.approx(parameters.sigma_m_db, abs=1e-4)

        # The bistatic falloff is 3dB over the whole range of `beta`
        got = _sigma_md_db(rcs, 90., 0., beta_deg=180.)
        assert got == pytest.approx(parameters.sigma_m_db - 3., abs=1e-4)
        got = _sigma_md_db(rcs, 90., 0., beta_deg=60.)
        assert got == pytest.approx(parameters.sigma_m_db - 1.5, abs=1e-4)


# ------------------------------------------------------------------------
# Lobe selection
# ------------------------------------------------------------------------

@pytest.mark.parametrize("object_type", ["uav-large-size", "human",
                                         "vehicle-single-sp",
                                         "agv-single-sp"])
def test_lobe_selection_picks_the_tabulated_row(object_type):
    # A model with a single scattering point holds every row of its table and
    # the bisector angle indexes one of them, as given by the `Range of theta`
    # and `Range of phi` columns
    lobes = get_target_parameters(object_type, 2).lobes
    whole_table = _whole_table_rcs(object_type)

    for lobe in lobes:
        theta_lo, theta_hi = lobe.theta_range
        phi_lo, phi_hi = lobe.phi_range
        # Sample the interior of the range of the lobe
        for theta_weight in (0.1, 0.5, 0.9):
            for phi_weight in (0.1, 0.5, 0.9):
                theta = theta_lo + theta_weight*(theta_hi - theta_lo)
                phi = phi_lo + phi_weight*(phi_hi - phi_lo)
                assert _sigma_md_db(whole_table, theta, phi) \
                    == pytest.approx(
                        _sigma_md_db(_single_lobe_rcs(object_type, lobe),
                                     theta, phi), abs=1e-4)


def test_lobe_selection_handles_wrapped_azimuth_ranges():
    # The front lobes of the vehicle and the AGV span [-45, 45), and the front
    # lobe of the human spans [-90, 90), so their ranges wrap around
    for object_type, phi_hi in (("vehicle-single-sp", 45.),
                                ("agv-single-sp", 45.), ("human", 90.)):
        lobes = get_target_parameters(object_type, 2).lobes
        front = next(lobe for lobe in lobes if lobe.name == "front")
        assert front.phi_range == (-phi_hi, phi_hi)

        whole_table = _whole_table_rcs(object_type)
        single = _single_lobe_rcs(object_type, front)
        theta = front.theta_center
        # Just inside either end of the range, given in [0, 360) as an
        # azimuth computed from a direction is
        for phi in (0., phi_hi - 1., 360. - phi_hi + 1., 359.):
            assert _sigma_md_db(whole_table, theta, phi) \
                == pytest.approx(_sigma_md_db(single, theta, phi), abs=1e-4)


def test_human_model_2_is_front_back_symmetric():
    # The two rows of Table 7.9.2.1-3 only differ by their azimuth center and
    # range, so the model is symmetric under a half turn in azimuth
    lobes = get_target_parameters("human", 2).lobes
    assert len(lobes) == 2
    front, back = lobes
    assert (front.name, back.name) == ("front", "back")
    assert front.phi_range == (-90., 90.)
    assert back.phi_range == (90., 270.)

    rcs = _whole_table_rcs("human")
    rng = np.random.default_rng(7)
    for theta in np.degrees(np.arccos(rng.uniform(-1., 1., 16))):
        for phi in rng.uniform(0., 360., 16):
            assert _sigma_md_db(rcs, theta, phi) \
                == pytest.approx(_sigma_md_db(rcs, theta, phi + 180.),
                                 abs=1e-4)


# ------------------------------------------------------------------------
# Consistency of the transcribed tables
# ------------------------------------------------------------------------

@pytest.mark.parametrize("object_type,model_type", _COMBINATIONS)
def test_floor_is_constant_within_a_table(object_type, model_type):
    # Every row of a table shares the same `G_max - sigma_max`
    lobes = get_target_parameters(object_type, model_type).lobes
    if not lobes:
        pytest.skip("RCS model 1 has no lobe")
    floors = [lobe.floor for lobe in lobes]
    assert max(floors) - min(floors) < 0.02


@pytest.mark.parametrize("single,multi", _VARIANT_PAIRS)
def test_multi_point_floor_is_the_single_point_floor_split(single, multi):
    # The floor of a multiple scattering point table is that of the matching
    # single scattering point table divided by the number of points, so that
    # the points summing at their floor reproduce the single-point floor
    single_lobes = get_target_parameters(single, 2).lobes
    multi_parameters = get_target_parameters(multi, 2)
    num_points = multi_parameters.num_scattering_points
    assert num_points == 5

    gap = single_lobes[0].floor - multi_parameters.lobes[0].floor
    assert gap == pytest.approx(10.*np.log10(num_points), abs=0.01)


@pytest.mark.parametrize("single,multi", _VARIANT_PAIRS)
def test_multi_point_boresights_reconstruct_the_single_point_values(single,
                                                                   multi):
    # At the boresight of one face, the other four points sit at their floor,
    # and the cross-sections of the five points sum to the `G_max` of the
    # single scattering point table for that face
    single_lobes = {lobe.name: lobe
                    for lobe in get_target_parameters(single, 2).lobes}
    multi_lobes = get_target_parameters(multi, 2).lobes

    for lobe in multi_lobes:
        theta, phi = _lobe_boresight(lobe)
        total = sum(10.**(0.1*_sigma_md_db(_single_lobe_rcs(multi, other),
                                           theta, phi))
                    for other in multi_lobes)
        assert 10.*np.log10(total) \
            == pytest.approx(single_lobes[lobe.name].g_max, abs=0.01)


# ------------------------------------------------------------------------
# Cross-section and cross-polarization matrix
# ------------------------------------------------------------------------

@pytest.mark.parametrize("object_type,model_type", _COMBINATIONS)
def test_rcs_returns_the_linear_cross_section(object_type, model_type):
    rcs = _whole_table_rcs(object_type, model_type)

    theta, phi, beta = 63., 217., 37.
    sigma = 10.**(0.1*_sigma_md_db(rcs, theta, phi, beta))

    got = np.array(rcs(*_propagation_directions(theta, phi, beta)))
    assert got.squeeze() == pytest.approx(sigma, rel=1e-5)


@pytest.mark.parametrize("object_type,model_type", _COMBINATIONS)
def test_cpm_carries_the_xpr_of_the_table(object_type, model_type):
    parameters = get_target_parameters(object_type, model_type)
    cpm = _table_cpm(object_type, model_type)
    assert cpm.xpr_db == parameters.xpr_mean_db

    real, imag = cpm(*_propagation_directions(63., 217., 37.))
    real = np.array(real).squeeze()

    # The cross-polarization matrix of eq. 7.9.2-5 is symmetric, and dropping
    # its phases leaves it real
    assert real == pytest.approx(real.T, rel=1e-5)
    assert np.allclose(np.array(imag).squeeze(), 0.)

    # The co-polarized entries of eq. 7.9.2-5 have unit modulus, so the
    # matrix is not normalized and the depolarized power adds to the
    # cross-section
    beta = 10.**(-0.05*parameters.xpr_mean_db)
    assert np.diag(real) == pytest.approx(1., rel=1e-5)
    assert np.sum(real**2) == pytest.approx(2.*(1. + beta**2), rel=1e-5)

    # The ratio of the co- and cross-polarized entries is the XPR of
    # Table 7.9.2.2-1
    xpd_db = 10.*np.log10(real[0, 0]**2/real[0, 1]**2)
    assert xpd_db == pytest.approx(parameters.xpr_mean_db, rel=1e-5)


def test_cpm_is_the_identity_without_an_xpr():
    cpm = TR38901CPM()
    assert cpm.xpr_db is None

    real, imag = cpm(*_propagation_directions(63., 217., 37.))

    # Without an XPR, the scattering point preserves the polarization of the
    # incident field
    assert np.array(real).squeeze() == pytest.approx(np.eye(2))
    assert np.allclose(np.array(imag).squeeze(), 0.)


# ------------------------------------------------------------------------
# Random components
# ------------------------------------------------------------------------

# Number of pairs of directions the draws are checked on
_NUM_DRAWS = 100000


@pytest.mark.parametrize("object_type,model_type", _COMBINATIONS)
def test_random_components_are_disabled_by_default(object_type, model_type):
    parameters = get_target_parameters(object_type, model_type)
    rcs = _whole_table_rcs(object_type, model_type)
    cpm = _table_cpm(object_type, model_type)
    model = _model(object_type, model_type)

    assert not rcs.random_sigma_s
    assert not cpm.random_phases
    assert not cpm.random_xpr
    assert not model.random_sigma_s
    assert not model.random_phases
    assert not model.random_xpr

    # The tabulated standard deviations are carried, ready to be drawn from
    assert rcs.sigma_s_std_db == parameters.sigma_s_db
    assert cpm.xpr_std_db == parameters.xpr_std_db

    # `sigma_S` is fixed to 1, i.e. to 0dB, and the matrix stays real
    k_i, k_s = _random_directions(256)
    assert np.allclose(np.array(rcs.sigma_s_db(k_i, k_s, 1)), 0.)
    assert np.allclose(np.array(cpm(k_i, k_s, 1)[1]), 0.)


@pytest.mark.parametrize("flags", [{}, {"random_sigma_s": True},
                                   {"random_phases": True, "random_xpr": True},
                                   {"random_sigma_s": True, "random_phases": False}])
def test_target_forwards_the_random_flags_to_its_model(flags):
    # Flags which are not given keep the defaults of the scattering model
    target = TR38901SensingTarget("st", "vehicle-multi-sp", **flags)
    model = target.scattering_model
    expected = {name: flags.get(name, False)
                for name in ("random_sigma_s", "random_phases", "random_xpr")}

    assert {name: getattr(model, name) for name in expected} == expected
    assert all(rcs.random_sigma_s == expected["random_sigma_s"]
               for rcs in model._rcs)
    assert model._cpm.random_phases == expected["random_phases"]
    assert model._cpm.random_xpr == expected["random_xpr"]


@pytest.mark.parametrize("object_type,model_type", _COMBINATIONS)
def test_sigma_s_follows_the_distribution_of_the_specifications(object_type,
                                                                model_type):
    std = get_target_parameters(object_type, model_type).sigma_s_db
    rcs = _whole_table_rcs(object_type, model_type, random_sigma_s=True)

    k_i, k_s = _random_directions(_NUM_DRAWS)
    sigma_s_db = np.array(rcs.sigma_s_db(k_i, k_s, 1))

    # Mean of eq. 7.9.2-1, which makes the linear mean of `sigma_S` exactly 1
    mean = -np.log(10.)/20.*std**2
    assert sigma_s_db.mean() == pytest.approx(mean, abs=0.1)
    assert sigma_s_db.std() == pytest.approx(std, rel=0.05)

    # The truncation of eq. 7.9.4-1 caps the draw three standard deviations
    # above its mean, which enough samples reach to make it the maximum
    cap = mean + 3.*std
    assert sigma_s_db.max() == pytest.approx(cap, rel=1e-5)
    assert np.mean(sigma_s_db >= cap - 1e-4) == pytest.approx(0.00135,
                                                              abs=5e-4)

    # Truncating a distribution of unit linear mean lowers that mean a little
    linear_mean = np.mean(10.**(0.1*sigma_s_db))
    assert 0.9 < linear_mean <= 1.01


@pytest.mark.parametrize("object_type,model_type", _COMBINATIONS)
def test_rcs_scales_the_cross_section_by_sigma_s(object_type, model_type):
    rcs = _whole_table_rcs(object_type, model_type, random_sigma_s=True)

    k_i, k_s = _random_directions(256)
    sigma_md_db = np.array(rcs.sigma_md_db(k_i, k_s))
    sigma_s_db = np.array(rcs.sigma_s_db(k_i, k_s, 3))

    # The cross-section is the product of the three RCS components, and
    # `sigma_S` does vary from one pair of directions to the next
    assert np.allclose(np.array(rcs(k_i, k_s, 3)),
                       10.**(0.1*(sigma_md_db + sigma_s_db)), rtol=1e-5)
    assert sigma_s_db.std() > 0.


@pytest.mark.parametrize("object_type,model_type", _COMBINATIONS)
def test_xpr_follows_the_distribution_of_the_table(object_type, model_type):
    parameters = get_target_parameters(object_type, model_type)
    cpm = _table_cpm(object_type, model_type, random_xpr=True)

    k_i, k_s = _random_directions(_NUM_DRAWS)
    matrices = _matrices(cpm, k_i, k_s)

    # The XPR is the ratio of the co- to the cross-polarized power, and is
    # drawn from the log-normal distribution of Table 7.9.2.2-1
    xpr_db = 20.*np.log10(np.abs(matrices[0, 0]/matrices[0, 1]))
    assert xpr_db.mean() == pytest.approx(parameters.xpr_mean_db, abs=0.1)
    assert xpr_db.std() == pytest.approx(parameters.xpr_std_db, rel=0.05)

    # Both cross-polarized entries carry the same draw, as eq. 7.9.2-5 gives
    # one XPR per pair of directions
    assert np.allclose(matrices[0, 1], matrices[1, 0])


def test_random_phases_are_uniform():
    cpm = _table_cpm("vehicle-single-sp", random_phases=True)

    k_i, k_s = _random_directions(_NUM_DRAWS)
    phases = np.angle(_matrices(cpm, k_i, k_s)).reshape(4, -1)

    # The four phases of eq. 7.9.2-5 are uniform over `(-pi, pi)`
    for phase in phases:
        assert phase.min() == pytest.approx(-np.pi, abs=1e-3)
        assert phase.max() == pytest.approx(np.pi, abs=1e-3)
        assert phase.mean() == pytest.approx(0., abs=0.05)
        assert phase.std() == pytest.approx(np.pi/np.sqrt(3.), rel=0.02)

    # They are drawn independently of each other
    correlations = np.corrcoef(phases) - np.eye(4)
    assert np.abs(correlations).max() < 0.02


@pytest.mark.parametrize("random_phases", [False, True])
@pytest.mark.parametrize("random_xpr", [False, True])
@pytest.mark.parametrize("xpr_db", [None, 9.6])
def test_cpm_keeps_the_entries_of_the_specifications_when_drawn(xpr_db,
                                                                random_xpr,
                                                                random_phases):
    cpm = TR38901CPM(xpr_db, xpr_std_db=6.85, random_phases=random_phases,
                     random_xpr=random_xpr)

    k_i, k_s = _random_directions(4096)
    matrices = _matrices(cpm, k_i, k_s)

    # Whatever the draws, the co-polarized entries of eq. 7.9.2-5 have unit
    # modulus, which makes the norm of the matrix `2(1+beta^2)`
    assert np.allclose(np.abs(matrices[0, 0]), 1., rtol=1e-5)
    assert np.allclose(np.abs(matrices[1, 1]), 1., rtol=1e-5)

    beta = np.abs(matrices[0, 1])
    norms = np.sum(np.abs(matrices)**2, axis=(0, 1))
    assert np.allclose(norms, 2.*(1. + beta**2), rtol=1e-5)

    # A scattering point with an infinite XPR does not depolarize, even when
    # the XPR is drawn, so its matrix is normalized
    if xpr_db is None:
        assert np.allclose(matrices[0, 1], 0.)
        assert np.allclose(matrices[1, 0], 0.)
        assert np.allclose(norms, 2., rtol=1e-5)


def test_draws_are_reciprocal():
    rcs = _whole_table_rcs("agv-single-sp", random_sigma_s=True)
    cpm = _table_cpm("agv-single-sp", random_phases=True, random_xpr=True)

    k_i, k_s = _random_directions(4096)

    # Clause 7.9.4 requires the same `sigma_S` in both directions of
    # propagation, and the cross-polarization matrix to be transposed
    assert np.allclose(np.array(rcs.sigma_s_db(k_s, k_i, 1)),
                       np.array(rcs.sigma_s_db(k_i, k_s, 1)))
    assert np.allclose(_matrices(cpm, k_s, k_i),
                       _matrices(cpm, k_i, k_s).transpose(1, 0, 2))


def test_draws_are_reproducible_and_keyed_by_the_seed():
    rcs = _whole_table_rcs("human", random_sigma_s=True)
    cpm = _table_cpm("human", random_phases=True, random_xpr=True)

    k_i, k_s = _random_directions(4096)
    sigma_s_db = np.array(rcs.sigma_s_db(k_i, k_s, 42))
    matrices = _matrices(cpm, k_i, k_s, 42)

    # The draws only depend on the directions and on the seed
    assert np.array_equal(np.array(rcs.sigma_s_db(k_i, k_s, 42)), sigma_s_db)
    assert np.array_equal(_matrices(cpm, k_i, k_s, 42), matrices)

    # Another seed gives unrelated draws
    other_sigma_s_db = np.array(rcs.sigma_s_db(k_i, k_s, 43))
    assert not np.allclose(other_sigma_s_db, sigma_s_db)
    assert not np.allclose(_matrices(cpm, k_i, k_s, 43), matrices)

    # Which are drawn from the same distribution
    assert other_sigma_s_db.std() == pytest.approx(sigma_s_db.std(), rel=0.1)


@pytest.mark.parametrize("object_type", ["human", "vehicle-multi-sp"])
def test_model_flags_reach_the_callables(object_type):
    # Enabled through the constructor
    model = _standalone_model(object_type, random_sigma_s=True,
                              random_phases=True, random_xpr=True)
    assert model.random_sigma_s
    assert model.random_phases
    assert model.random_xpr
    assert all(rcs.random_sigma_s for rcs in model.spst.rcs_callables)
    assert all(cpm.random_phases and cpm.random_xpr
               for cpm in model.spst.cpm_callables)

    # And through the properties, which write through to every callable
    model.random_sigma_s = False
    model.random_phases = False
    model.random_xpr = False
    assert not any(rcs.random_sigma_s for rcs in model.spst.rcs_callables)
    assert not any(cpm.random_phases or cpm.random_xpr
                   for cpm in model.spst.cpm_callables)

    # The callables also carry the standard deviations of the tables
    parameters = get_target_parameters(object_type, 2)
    assert all(rcs.sigma_s_std_db == parameters.sigma_s_db
               for rcs in model.spst.rcs_callables)
    assert all(cpm.xpr_std_db == parameters.xpr_std_db
               for cpm in model.spst.cpm_callables)


def test_flags_reject_non_boolean_values():
    model = _model("human")

    for flag in ("random_sigma_s", "random_phases", "random_xpr"):
        with pytest.raises(TypeError, match=flag):
            setattr(model, flag, 1)

    with pytest.raises(ValueError, match="sigma_s_std_db"):
        TR38901RCS((), sigma_s_std_db=-1.)
    with pytest.raises(ValueError, match="xpr_std_db"):
        TR38901CPM(9.6, xpr_std_db=-1.)


# ------------------------------------------------------------------------
# Scattering points of the model
# ------------------------------------------------------------------------

@pytest.mark.parametrize("object_type,model_type", _COMBINATIONS)
def test_model_has_the_number_of_points_of_its_object_type(object_type,
                                                           model_type):
    model = _model(object_type, model_type)
    parameters = get_target_parameters(object_type, model_type)

    assert dr.width(model.spst.lcs_positions) \
        == parameters.num_scattering_points
    assert len(model.spst.rcs_callables) == parameters.num_scattering_points
    assert model.object_type == object_type
    assert model.model_type == model_type

    if parameters.num_scattering_points == 1:
        # The single point holds every row of its table as a lobe
        assert model.spst.rcs_callables[0].lobes == parameters.lobes
    else:
        # Every point holds one row
        assert all(len(rcs.lobes) == 1
                   for rcs in model.spst.rcs_callables)

    # The specifications give one XPR per target, so every point shares the
    # same CPM
    cpm_callables = model.spst.cpm_callables
    assert len(cpm_callables) == parameters.num_scattering_points
    assert all(cpm is cpm_callables[0] for cpm in cpm_callables)
    assert cpm_callables[0].xpr_db == parameters.xpr_mean_db


@pytest.mark.parametrize("object_type,model_type", _COMBINATIONS)
def test_single_point_models_place_their_point_at_the_center(object_type,
                                                             model_type):
    parameters = get_target_parameters(object_type, model_type)
    if parameters.num_scattering_points != 1:
        pytest.skip("Not a single scattering point model")

    target = _target(object_type, model_type)
    model = target.scattering_model
    height = target.height

    if object_type == "human":
        # Table 7.9.6.1-2 places the scattering point of an adult pedestrian
        # of 1.75m at a height of 1.5m, i.e., at 5*H/14 above mid-body
        expected = mi.Point3f(0., 0., 5.*height/14.)
    else:
        expected = mi.Point3f(0., 0., 0.)

    assert dr.allclose(model.spst.lcs_positions, expected, atol=1e-6)
    assert dr.allclose(model.spst.lcs_orientations, mi.Point3f(0., 0., 0.))


@pytest.mark.parametrize("height", [1.75, 2.1])
def test_human_scattering_point_scales_with_the_height(height):
    model = _model("human", 2, height=height)
    z = np.array(model.spst.lcs_positions.z)[0]
    # The point is at 1.5/1.75 of the height, in a frame centered at mid-body
    assert z == pytest.approx((1.5/1.75 - 0.5)*height, abs=1e-6)
    if height == 1.75:
        assert z == pytest.approx(0.625, abs=1e-6)


@pytest.mark.parametrize("object_type", ["vehicle-multi-sp", "agv-multi-sp"])
@pytest.mark.parametrize("scale", [1., 2.5])
def test_multi_point_models_place_their_points_on_the_faces(object_type,
                                                            scale):
    default = get_target_parameters(object_type, 2).dimensions
    length, width, height = (scale*value for value in default)
    model = _model(object_type, 2, length=length, width=width, height=height)

    # NOTE 2 of Table 7.9.6.1-3 places the points at the centers of the faces,
    # the side ones at mid-height and the roof one at the top, in a frame
    # whose origin lies on the ground. The frame of a `SensingTarget` is
    # centered instead, which shifts every height down by `H/2`.
    expected = {
        "front": (0.5*length, 0., 0.),
        "left": (0., 0.5*width, 0.),
        "back": (-0.5*length, 0., 0.),
        "right": (0., -0.5*width, 0.),
        "roof": (0., 0., 0.5*height),
    }
    positions = np.array(model.spst.lcs_positions).T
    orientations = np.array(model.spst.lcs_orientations).T

    lobes = get_target_parameters(object_type, 2).lobes
    assert len(lobes) == len(positions) == 5
    for lobe, position, orientation in zip(lobes, positions, orientations):
        assert position == pytest.approx(expected[lobe.name], abs=1e-6)
        # Orientations carry the azimuth center of the lobe only
        azimuth = 0. if lobe.phi_center is None \
            else np.radians(lobe.phi_center)
        assert orientation == pytest.approx((azimuth, 0., 0.), abs=1e-6)


# ------------------------------------------------------------------------
# Target geometry and dimensions
# ------------------------------------------------------------------------

@pytest.mark.parametrize("object_type,model_type", _COMBINATIONS)
def test_dimensions_default_to_the_specification_values(object_type,
                                                        model_type):
    target = _target(object_type, model_type)
    expected = get_target_parameters(object_type, model_type).dimensions

    assert (target.length, target.width, target.height) == expected
    assert target.dimensions == {"length": expected[0], "width": expected[1],
                                 "height": expected[2]}
    assert np.array(target.mi_mesh.bbox().max - target.mi_mesh.bbox().min) \
        == pytest.approx(expected)


def test_default_dimensions_match_the_tables():
    # Length x Width x Height, the ordering the specifications use
    assert get_target_parameters("uav-small-size", 1).dimensions \
        == (0.3, 0.4, 0.2)
    assert get_target_parameters("uav-large-size", 2).dimensions \
        == (1.6, 1.5, 0.7)
    assert get_target_parameters("human", 1).dimensions == (0.5, 0.5, 1.75)
    assert get_target_parameters("human", 2).dimensions == (0.5, 0.5, 1.75)
    assert get_target_parameters("vehicle-single-sp", 2).dimensions \
        == (5.0, 2.0, 1.6)
    assert get_target_parameters("vehicle-multi-sp", 2).dimensions \
        == (5.0, 2.0, 1.6)

    # The AGV is wider than it is long, its front facing the short edge in
    # the horizontal direction, as clause 7.9.2.1 states
    for object_type in ("agv-single-sp", "agv-multi-sp"):
        assert get_target_parameters(object_type, 2).dimensions \
            == (0.5, 1.0, 0.5)


def test_dimensions_default_independently_of_each_other():
    # Overriding one dimension keeps the values of the specifications for the
    # other two
    target = _target("vehicle-multi-sp", height=2.4)
    assert (target.length, target.width, target.height) == (5.0, 2.0, 2.4)

    target = _target("agv-single-sp", length=0.8, width=1.2)
    assert (target.length, target.width, target.height) == (0.8, 1.2, 0.5)


@pytest.mark.parametrize("dimension", ["length", "width", "height"])
@pytest.mark.parametrize("value", [0., -1.])
def test_dimensions_must_be_strictly_positive(dimension, value):
    with pytest.raises(ValueError, match=f"`{dimension}` must be strictly "
                                         "positive"):
        _target("human", 2, **{dimension: value})


def test_mesh_aabb_sets_dimensions_and_scattering_point_positions():
    mesh = TR38901SensingTarget._cuboid(8., 3., 2.)
    target = _target("vehicle-multi-sp", mi_mesh=mesh)

    assert target.dimensions == {"length": 8., "width": 3., "height": 2.}
    positions = np.array(target.scattering_model.spst.lcs_positions).T
    names = [lobe.name for lobe in
             get_target_parameters("vehicle-multi-sp", 2).lobes]
    expected = {
        "front": (4., 0., 0.),
        "left": (0., 1.5, 0.),
        "back": (-4., 0., 0.),
        "right": (0., -1.5, 0.),
        "roof": (0., 0., 1.),
    }
    for name, position in zip(names, positions):
        assert position == pytest.approx(expected[name])


def test_mesh_file_aabb_sets_dimensions():
    target = _target("human", fname=sionna.rt.scene.sphere)
    extents = np.array(target.mi_mesh.bbox().max - target.mi_mesh.bbox().min)

    assert (target.length, target.width, target.height) \
        == pytest.approx(extents)
    z = np.array(target.scattering_model.spst.lcs_positions.z)[0]
    assert z == pytest.approx((1.5/1.75 - 0.5)*target.height)


def test_mesh_and_dimensions_are_mutually_exclusive():
    mesh = TR38901SensingTarget._cuboid(1., 1., 1.)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _target("human", mi_mesh=mesh, height=2.)


def test_target_and_model_share_the_model_parameters():
    target = _target("human", model_type=1)
    assert isinstance(target.scattering_model, TR38901ScatteringModel)
    assert target.object_type == target.scattering_model.object_type == "human"
    assert target.model_type == target.scattering_model.model_type == 1


def test_target_clone_preserves_specialized_type_and_parameters():
    target = _target("vehicle-multi-sp", length=8., width=3., height=2.)
    clone = target.clone()

    assert isinstance(clone, TR38901SensingTarget)
    assert clone.name == "st-clone"
    assert clone.object_type == target.object_type
    assert clone.model_type == target.model_type
    assert clone.dimensions == target.dimensions


def test_target_dimensions_follow_the_scaling():
    target = _target("vehicle-multi-sp", length=8., width=3., height=2.)
    target.scaling = mi.Vector3f(0.5, 2., 4.)

    assert target.dimensions == pytest.approx({"length": 4., "width": 6.,
                                               "height": 8.})
    assert (target.length, target.width, target.height) \
        == pytest.approx((4., 6., 8.))
    # The reported dimensions are those of the mesh the solver sees
    assert np.allclose(2.*target.lcs_half_extents.numpy().squeeze(),
                       [4., 6., 8.], atol=1e-5)

    # A clone is scaled like the target it was built from
    assert target.clone().dimensions == pytest.approx(target.dimensions)

    # The scattering points are still given in the unscaled frame, and the
    # solver is what scales them
    positions = np.array(target.scattering_model.spst.lcs_positions).T
    assert np.abs(positions).max(axis=0) == pytest.approx([4., 1.5, 1.])


def test_target_can_be_posed_at_construction():
    target = _target("vehicle-multi-sp", length=8., width=3., height=2.,
                     position=mi.Point3f(10., 5., 1.),
                     orientation=mi.Point3f(np.pi/2, 0., 0.),
                     velocity=mi.Vector3f(20., 0., 0.))

    assert np.allclose(target.position.numpy()[:, 0], [10., 5., 1.], atol=1e-5)
    assert np.allclose(target.orientation.numpy()[:, 0], [np.pi/2, 0., 0.])
    assert np.allclose(target.velocity.numpy()[:, 0], [20., 0., 0.])

    # The dimensions are those of the LCS, and so are unaffected by the
    # rotation, even though the AABB of the mesh now spans 3m along the x-axis
    assert target.dimensions == {"length": 8., "width": 3., "height": 2.}
    extents = target.mi_mesh.bbox().max - target.mi_mesh.bbox().min
    assert np.allclose(np.array(extents), [3., 8., 2.], atol=1e-5)

    # A look-at target orients the object without a scene being involved
    looking = TR38901SensingTarget("looking", "human",
                                   position=mi.Point3f(0., 0., 0.),
                                   look_at=mi.Point3f(0., 5., 0.))
    assert np.allclose(looking.orientation.numpy()[:, 0],
                       [np.pi/2, 0., 0.], atol=1e-5)

    # The pose survives being added to a scene
    scene = load_scene()
    scene.add([target, looking])
    assert set(scene.sensing_targets) == {"st", "looking"}
    assert np.allclose(target.position.numpy()[:, 0], [10., 5., 1.], atol=1e-5)


def test_target_clone_preserves_pose_and_lcs_dimensions():
    target = _target("vehicle-multi-sp", length=8., width=3., height=2.,
                     position=mi.Point3f(10., 5., 1.),
                     orientation=mi.Point3f(np.pi/2, 0., 0.),
                     velocity=mi.Vector3f(20., 0., 0.))
    clone = target.clone()

    # The dimensions are carried over rather than re-derived from the AABB of
    # the cloned mesh, which is rotated and would give 3m x 8m x 2m
    assert clone.dimensions == target.dimensions
    assert np.allclose(clone.position.numpy(), target.position.numpy(),
                       atol=1e-5)
    assert np.allclose(clone.orientation.numpy(), target.orientation.numpy())
    assert np.allclose(clone.velocity.numpy(), target.velocity.numpy())

    # As for any sensing target, the scattering model is shared
    assert clone.scattering_model is target.scattering_model


# ------------------------------------------------------------------------
# Rejected combinations
# ------------------------------------------------------------------------

@pytest.mark.parametrize("object_type", ["vehicle", "agv"])
def test_ambiguous_object_types_name_both_variants(object_type):
    # The specifications provide both variants for the vehicle and the AGV
    # only, and they are different parameterizations, so neither is a default
    with pytest.raises(ValueError, match="must be selected") as excinfo:
        _target(object_type)
    message = str(excinfo.value)
    assert f"{object_type}-single-sp" in message
    assert f"{object_type}-multi-sp" in message


def test_unknown_object_types_are_rejected():
    with pytest.raises(ValueError, match="Unknown object type 'truck'"):
        _target("truck")


@pytest.mark.parametrize("model_type", [0, 3, "2"])
def test_invalid_model_types_are_rejected(model_type):
    with pytest.raises(ValueError, match="Invalid model type"):
        _target("human", model_type)


@pytest.mark.parametrize("object_type,model_type",
                         [("uav-small-size", 2), ("uav-large-size", 1),
                          ("vehicle-single-sp", 1), ("vehicle-multi-sp", 1),
                          ("agv-single-sp", 1), ("agv-multi-sp", 1)])
def test_undefined_combinations_are_rejected(object_type, model_type):
    with pytest.raises(ValueError, match="do not define RCS model"):
        _target(object_type, model_type)


def test_every_object_type_is_reachable():
    # Every object type of the lookup has at least one model type
    for object_type in OBJECT_TYPES:
        models = [model for object_type_, model in _COMBINATIONS
                  if object_type_ == object_type]
        assert models
        for model_type in models:
            _target(object_type, model_type)


# ------------------------------------------------------------------------
# End-to-end
# ------------------------------------------------------------------------

@pytest.mark.parametrize("object_type", ["vehicle-single-sp",
                                         "uav-large-size", "human"])
def test_solver_reproduces_the_cross_section_of_the_model(object_type):
    # Monostatic geometry along the x-axis, so both legs have the same length
    # and the polarizations are aligned. The single scattering point of the
    # target sits at its center, except for the human.
    distance = 30.
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(0, 0, 0)))

    target = _target(object_type)
    model = target.scattering_model
    scene.edit(add=target)
    st = scene.get("st")
    st.position = mi.Point3f(distance, 0, 0)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)
    a_real, a_imag = paths.a
    assert a_real.shape == (1, 1, 1, 1, 1)

    # The transmitter is seen from the back of the target, which faces the
    # +x axis, and the two legs reach the scattering point of the model
    point = np.array(model.spst.lcs_positions).T[0]
    leg = np.linalg.norm(np.array([distance, 0., 0.]) + point)
    k_i = mi.Vector3f(1., 0., 0.)
    sigma = np.array(model.spst.rcs_callables[0](k_i, -k_i)).squeeze()

    # Both devices are V-polarized, so the link only sees the co-polarized
    # entry of the Jones matrix `sqrt(sigma)*W`
    cpm_real, _ = model.spst.cpm_callables[0](k_i, -k_i)
    co = np.array(cpm_real).squeeze()[0, 0]

    wavelength = scene.wavelength.numpy()[0]
    expected_a = wavelength/(4.*np.pi)*co*np.sqrt(sigma/(4.*np.pi))/leg**2
    assert np.allclose(a_real.numpy().squeeze(), expected_a, rtol=1e-4)
    assert np.allclose(a_imag.numpy().squeeze(), 0.)


def test_solver_sums_the_points_of_a_multi_point_model():
    # The five points of a multiple scattering point model each give a path
    distance = 50.
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(0, 0, 0)))

    target = _target("vehicle-multi-sp")
    scene.edit(add=target)
    scene.get("st").position = mi.Point3f(distance, 0, 0)

    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4)

    assert paths.tau.shape == (1, 1, 5)
    assert paths.valid.numpy().all()
    # One path per scattering point of the target, which the interaction
    # identifies by its index within the target
    primitives = paths.primitives.numpy().squeeze()
    assert sorted(primitives) == [0, 1, 2, 3, 4]

    # The front of the target faces the +x axis, so its back point is the
    # closest to the transmitter and has the shortest delay, and its front
    # point the longest. The points keep the row order of Table 7.9.2.1-5.
    names = [lobe.name for lobe in
             get_target_parameters("vehicle-multi-sp", 2).lobes]
    names_per_path = [names[primitive] for primitive in primitives]
    tau = paths.tau.numpy().squeeze()
    assert names_per_path[np.argmin(tau)] == "back"
    assert names_per_path[np.argmax(tau)] == "front"


def _solver_coefficients(scene, seed):
    """Channel coefficients of the paths of a scene, ordered by the scattering
    point they interact with, so that two calls can be compared.
    """
    paths = RCSSolver()(scene, max_depth=1, samples_per_sp=10**4, seed=seed)
    a_real, a_imag = paths.a
    a = a_real.numpy().squeeze() + 1j*a_imag.numpy().squeeze()
    return a[np.argsort(paths.primitives.numpy().squeeze())]


def test_solver_draws_are_reproducible_and_keyed_by_its_seed():
    # Monostatic geometry, in which every scattering point of the target
    # gives exactly one path
    scene = _iso_scene()
    scene.add(Transmitter("tx", position=mi.Point3f(0, 0, 0)))
    scene.add(Receiver("rx", position=mi.Point3f(0, 0, 0)))

    target = _target("vehicle-multi-sp")
    scene.edit(add=target)
    scene.get("st").position = mi.Point3f(40, 0, 0)

    deterministic = _solver_coefficients(scene, seed=1)

    model = target.scattering_model
    model.random_sigma_s = True
    model.random_phases = True
    model.random_xpr = True

    # Drawing the random components changes the coefficients, and gives
    # complex ones as the phases are no longer zero
    drawn = _solver_coefficients(scene, seed=1)
    assert not np.allclose(drawn, deterministic)
    assert not np.allclose(drawn.imag, 0.)

    # Two runs with the same seed give the same draws, whereas another seed
    # gives unrelated ones
    assert np.allclose(_solver_coefficients(scene, seed=1), drawn)
    assert not np.allclose(_solver_coefficients(scene, seed=2), drawn)
