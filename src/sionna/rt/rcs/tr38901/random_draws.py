#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Random draws of the models of 3GPP TR 38.901, clause 7.9.2"""

import math
from typing import Tuple
import drjit as dr
import mitsuba as mi

# Index of the stream of every random quantity, which keeps their draws
# mutually independent
STREAM_SIGMA_S = 0
STREAM_XPR = 1
STREAM_PHASE_THETA_THETA = 2
STREAM_PHASE_PHI_PHI = 3
STREAM_PHASE_CROSS = 4

# Odd constant mixed into the symmetric key, so that it does not collide with
# the keys of the individual directions. It is the 32-bit golden ratio, as used
# by most integer hashes.
_KEY_SALT = 0x9E3779B9

# Bounds applied to a uniform sample before it is fed to the inverse Gaussian
# cumulative distribution function, which diverges at both ends of [0, 1]
_MIN_UNIFORM = 1e-7
_MAX_UNIFORM = 1. - 1e-7

_SQRT_2 = math.sqrt(2.)


def direction_keys(k_i: mi.Vector3f,
                   k_s: mi.Vector3f,
                   seed: int) -> Tuple[mi.UInt32, mi.UInt32, mi.UInt32]:
    r"""Hashes a pair of directions into the keys of its random draws

    The draws of the models of clause 7.9.2 are made per pair of incident and
    scattered angles. A callable only sees the directions and a seed, and may
    be evaluated through :py:func:`drjit.switch`, where no per-sample index is
    available. The draws are therefore keyed by a hash of the directions
    themselves, which also makes them independent of the ordering of the
    samples and reproducible across evaluations.

    Three keys are returned. The first one is unchanged when the two
    directions are exchanged, and keys the quantities which clause 7.9.4
    requires to be the same in both directions of propagation, i.e., the RCS
    component :math:`\sigma_S`, the XPR, and the co-polarized phases. The last
    two are exchanged with the directions, and key the two cross-polarized
    phases, so that exchanging the directions transposes the
    cross-polarization matrix.

    The directions are hashed bit by bit, without any quantization, so two
    directions which differ in their last bit give unrelated draws.

    :param k_i: Incident directions of propagation
    :param k_s: Scattered directions of propagation
    :param seed: Seed of the draws
    :return: Key which is symmetric in the two directions
    :return: Key of the pair in the given order
    :return: Key of the pair in the reversed order
    """
    hash_i = _hash_direction(k_i, seed)
    hash_s = _hash_direction(k_s, seed)

    # The sum is commutative, so hashing it gives a key which does not depend
    # on the ordering of the two directions, whereas hashing the pair gives
    # one which does
    return (_hash_pair(hash_i + hash_s, mi.UInt32(_KEY_SALT)),
            _hash_pair(hash_i, hash_s),
            _hash_pair(hash_s, hash_i))


def uniform(key: mi.UInt32, stream: int) -> mi.Float:
    """Draws samples uniform over [0, 1)

    :param key: Key of the draw
    :param stream: Index of the stream of the drawn quantity
    :return: Samples uniform over [0, 1)
    """
    return mi.sample_tea_float32(key, mi.UInt32(stream))


def gaussian(key: mi.UInt32,
             stream: int,
             mean: float,
             std: float) -> mi.Float:
    """Draws Gaussian samples

    The samples are obtained by applying the inverse cumulative distribution
    function to uniform ones, which draws one sample per uniform.

    :param key: Key of the draw
    :param stream: Index of the stream of the drawn quantity
    :param mean: Mean of the distribution
    :param std: Standard deviation of the distribution
    :return: Gaussian samples
    """
    u = dr.clip(uniform(key, stream), _MIN_UNIFORM, _MAX_UNIFORM)
    return dr.fma(std*_SQRT_2, dr.erfinv(dr.fma(2., u, -1.)), mean)


def phase(key: mi.UInt32, stream: int) -> mi.Float:
    r"""Draws phases uniform over :math:`(-\pi, \pi)`

    :param key: Key of the draw
    :param stream: Index of the stream of the drawn quantity
    :return: Phases uniform over :math:`(-\pi, \pi)` [rad]
    """
    return dr.fma(dr.two_pi, uniform(key, stream), -dr.pi)


def _hash_direction(k: mi.Vector3f, seed: int) -> mi.UInt32:
    """Hashes a direction and a seed

    :param k: Directions
    :param seed: Seed of the draws
    :return: Hash of every direction
    """
    hash_ = mi.UInt32(seed)
    for component in (k.x, k.y, k.z):
        # A component is hashed through its bit pattern, which requires a
        # single-precision float
        hash_ = _hash_pair(hash_,
                           dr.reinterpret_array(mi.UInt32,
                                                mi.Float32(component)))
    return hash_


def _hash_pair(v_0: mi.UInt32, v_1: mi.UInt32) -> mi.UInt32:
    """Hashes a pair of unsigned integers

    :param v_0: First integers
    :param v_1: Second integers
    :return: Hash of every pair
    """
    return mi.sample_tea_32(v_0, v_1)[0]
