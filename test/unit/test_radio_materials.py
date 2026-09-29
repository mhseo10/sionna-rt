#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the radio materials."""

import pytest
import numpy as np
import drjit as dr
import mitsuba as mi

import sionna
from sionna.rt import (load_scene, AbsorberRadioMaterial, PathSolver,
                       PlanarArray, Transmitter, Receiver)
from sionna.rt.constants import InteractionType


def _surface_interaction(num_samples):
    """Surface interaction with the incident wave hitting a surface whose
    normal is the z-axis."""
    si = dr.zeros(mi.SurfaceInteraction3f, num_samples)
    si.wi = dr.normalize(mi.Vector3f(np.linspace(0.1, 0.9, num_samples), 0.,
                                     -1.))
    si.sh_frame = mi.Frame3f(mi.Vector3f(1, 0, 0), mi.Vector3f(0, 1, 0),
                             mi.Vector3f(0, 0, 1))
    return si


def test_absorber_requires_a_name():
    with pytest.raises(ValueError, match="`name` is required"):
        AbsorberRadioMaterial()


def test_absorber_is_loaded_from_a_mitsuba_dict():
    mat = mi.load_dict({"type": "absorber-radio-material",
                        "id": "mat-absorber",
                        "color": mi.ScalarColor3f(1., 0., 0.)})

    assert isinstance(mat, AbsorberRadioMaterial)
    # The `mat-` prefix is stripped from the name
    assert mat.name == "absorber"
    assert np.allclose(mat.color, (1., 0., 0.))


def test_absorber_sample_reports_no_interaction():
    mat = AbsorberRadioMaterial(name="absorber")

    num_samples = 8
    si = _surface_interaction(num_samples)
    bs, jones_mat = mat.sample(mi.BSDFContext(), si,
                               dr.zeros(mi.Float, num_samples),
                               dr.zeros(mi.Point2f, num_samples))

    assert np.all(bs.sampled_component.numpy() == InteractionType.NONE)
    assert np.allclose(bs.pdf.numpy(), 0.)
    # No energy is scattered
    assert np.allclose(jones_mat.numpy(), 0.)


def test_absorber_eval_and_pdf_are_zero():
    mat = AbsorberRadioMaterial(name="absorber")

    num_samples = 8
    si = _surface_interaction(num_samples)
    wo = dr.normalize(mi.Vector3f(0., 0.5, 1.))

    assert np.allclose(mat.eval(mi.BSDFContext(), si, wo).numpy(), 0.)
    assert np.allclose(mat.pdf(mi.BSDFContext(), si, wo).numpy(), 0.)


def test_absorber_scatters_no_path():
    # Transmitter and receiver above a small flat reflector, so that the only
    # path with a single interaction is the one reflected off the reflector
    scene = load_scene(sionna.rt.scene.simple_reflector)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, polarization="V",
                                 pattern="iso")
    scene.rx_array = scene.tx_array
    scene.add(Transmitter("tx", position=mi.Point3f(-0.1, -0.08, 1)))
    scene.add(Receiver("rx", position=mi.Point3f(-0.1, 0.36, 1)))

    reflector = scene.objects["merged-shapes"]
    solver = PathSolver()

    assert solver(scene, max_depth=2, los=False).tau.numpy().size == 1

    # The very same path vanishes once the reflector absorbs all the energy
    reflector.radio_material = AbsorberRadioMaterial(name="absorber")
    assert solver(scene, max_depth=2, los=False).tau.numpy().size == 0
