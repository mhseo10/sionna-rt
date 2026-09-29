#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utilities for ray tracing"""

import drjit as dr
import mitsuba as mi
from typing import Callable, Tuple
from dataclasses import dataclass

from sionna.rt.constants import EPSILON_FLOAT
from sionna.rt.utils.geometry import rotate_vector_around_axis,\
    rotation_matrix, theta_phi_from_unit_vec
from sionna.rt.utils.wedges import wedge_interior_angle


def spawn_ray_from_sources(
    lattice: Callable[[int], mi.Point2f],
    samples_per_src: int,
    src_positions: mi.Point3f
) -> mi.Ray3f:
    r"""
    Spawns ``samples_per_src`` rays for each source at the positions specified
    by ``src_positions``, oriented in the directions defined by the ``lattice``

    The spawned rays are ordered samples-first.

    :param lattice: Callable that generates the lattice used as directions for
        the rays
    :param samples_per_src: Number of rays per source to spawn
    :param src_positions: Positions of the sources
    """

    num_sources = dr.shape(src_positions)[1]

    # Ray directions
    samples_on_square = lattice(samples_per_src)
    k_world = mi.warp.square_to_uniform_sphere(samples_on_square)

    # Samples-first ordering is used, i.e., the samples are ordered as
    # follows:
    # [source_0_samples..., source_1_samples..., ...]
    # Each source has its own lattice
    k_world = dr.tile(k_world, num_sources)
    # Rays origins are the source locations
    origins = dr.repeat(src_positions, samples_per_src)

    # Spawn rays from the sources
    ray = mi.Ray3f(o=origins, d=k_world)
    # Minor workaround to avoid loop retracing due to size mismatch.
    ray.time = dr.zeros(mi.Float, dr.width(k_world))

    return ray

def offset_p(p: mi.Point3f, d: mi.Vector3f, n: mi.Vector3f) -> mi.Point3f:
    # pylint: disable=line-too-long
    r"""
    Adds a small offset to :math:`\mathbf{p}` along :math:`\mathbf{n}` such that
    :math:`\mathbf{n}^{\textsf{T}} \mathbf{d} \gt 0`

    More precisely, this function returns :math:`\mathbf{o}` such that:

    .. math::
        \mathbf{o} = \mathbf{p} + \epsilon\left(1 + \max{\left\{|p_x|,|p_y|,|p_z|\right\}}\right)\texttt{sign}(\mathbf{d} \cdot \mathbf{n})\mathbf{n}

    where :math:`\epsilon` depends on the numerical precision and :math:`\mathbf{p} = (p_x,p_y,p_z)`.

    :param p: Point to offset
    :param d: Direction toward which to offset along ``n``
    :param n: Direction along which to offset
    """

    a = (1. + dr.max(dr.abs(p), axis=0)) * EPSILON_FLOAT
    # Detach this operation to ensure there is no gradient computation
    a = dr.detach(dr.mulsign(a, dr.dot(d,n)))
    po = dr.fma(a, n, p)
    return po

def _safe_unit_direction(v: mi.Vector3f,
                         fallback: mi.Vector3f = mi.Vector3f(0, 0, 1)
                         ) -> Tuple[mi.Vector3f, mi.Float]:
    r"""
    Normalizes ``v`` when its length is positive; otherwise returns ``fallback``

    :param v: Vector to normalize
    :param fallback: Unit direction used when ``\|v\| = 0``

    :return: Unit direction and ``\|v\|``
    """
    length = dr.norm(v)
    direction = dr.select(length > 0, v * dr.rcp(length), fallback)
    return direction, length


def spawn_ray_towards(
    p: mi.Point3f,
    t: mi.Point3f,
    n: mi.Vector3f | None = None
) -> mi.Ray3f:
    r"""
    Spawns a ray with infinite length from :math:`\mathbf{p}` toward
    :math:`\mathbf{t}`

    If :math:`\mathbf{n}` is not :py:class:`None`, then a small offset is added
    to :math:`\mathbf{p}` along :math:`\mathbf{n}` in the direction of
    :math:`\mathbf{t}`.

    If :math:`\mathbf{p}` and :math:`\mathbf{t}` coincide (after the optional
    offset), the ray direction falls back to :math:`(0, 0, 1)` so that
    vectorized evaluation never produces NaN directions. Callers that need a
    meaningful direction must ensure a non-zero displacement.

    :param p: Origin of the ray
    :param t: Point towards which to spawn the ray
    :param n: (Optional) Direction along which to offset :math:`\mathbf{p}`
    """

    # Adds a small offset to `p` to avoid self-intersection
    if n is None:
        po = p
    else:
        po = offset_p(p, t - p, n)
    # Ray direction towards `t`
    d, _ = _safe_unit_direction(t - po)
    #
    ray = mi.Ray3f(po, d)
    # Small optimization to avoid loop retracing due to size mismatch.
    # This is still a literal.
    ray.time = dr.zeros(mi.Float, dr.width(ray.d))
    return ray

def spawn_ray_to(
    p: mi.Point3f,
    t: mi.Point3f,
    n: mi.Vector3f | None = None
) -> mi.Ray3f:
    r"""
    Spawns a finite ray from :math:`\mathbf{p}` to :math:`\mathbf{t}`

    The length of the ray is set to :math:`\|\mathbf{p} - \mathbf{t}\|`.

    If :math:`\mathbf{n}` is not :py:class:`None`, then a small offset is added
    to :math:`\mathbf{p}` along :math:`\mathbf{n}` in the direction of
    :math:`\mathbf{t}`.

    If :math:`\mathbf{p}` and :math:`\mathbf{t}` coincide (after the optional
    offset), the ray uses direction :math:`(0, 0, 1)` and ``maxt = 0`` so that
    vectorized evaluation never produces NaN directions.

    :param p: Origin of the ray
    :param t: Point towards which to spawn the ray
    :param n: (Optional) Direction along which to offset :math:`\mathbf{p}`
    """

    # Adds a small offset to `p`
    if n is None:
        po = p
    else:
        po = offset_p(p, t - p, n)
    # Ray direction towards `t`
    d, maxt = _safe_unit_direction(t - po)
    maxt = dr.select(maxt > 0, maxt * (1. - EPSILON_FLOAT), mi.Float(0))
    #
    ray = mi.Ray3f(po, d,  maxt=maxt, time=0., wavelengths=mi.Color0f())
    # Small optimization to avoid loop retracing due to size mismatch.
    # This is still a literal.
    ray.time = dr.zeros(mi.Float, dr.width(ray.d))
    return ray


def box_contains(p: mi.Point3f, half_extents: mi.Vector3f) -> mi.Bool:
    r"""
    Tests whether :math:`\mathbf{p}` lies within the box centered on the origin
    and spanning ``half_extents`` along every axis

    The test is inclusive up to a numerical tolerance, so that the points lying
    exactly on a face of the box, which is a common way to place them, are
    reported as contained.

    :param p: Points to test, in the frame of the box
    :param half_extents: Half extents of the box along its three axes
    """

    tolerance = half_extents*EPSILON_FLOAT + EPSILON_FLOAT
    return dr.all(dr.abs(p) <= (half_extents + tolerance), axis=0)


@dataclass
class SourceExclusionBoxes:
    # pylint: disable=line-too-long
    r"""
    Collection of boxes which do not occlude the rays spawned from their source

    Every source is paired with one box, given by its center, its half extents,
    and its orientation, and with a flag enabling it. The rays spawned from a
    source with an enabled box are not tested for occlusion within that box:
    :meth:`~sionna.rt.utils.SourceExclusionBoxes.advance` moves their origin to
    the point at which they leave it.

    This is used by :class:`~sionna.rt.rcs.RCSSolver` to keep a sensing
    target from occluding the scattering points it contains, as the scattering
    response of a target is entirely described by its scattering model. The
    box of a scattering point is then the bounding box of its sensing target,
    and is only enabled if the point lies within it.

    :param centers: Centers of the boxes, one per source
    :param half_extents: Half extents of the boxes along their three axes, one per source
    :param orientations: Orientations of the boxes, one per source, specified through three angles :math:`(\alpha, \beta, \gamma)` corresponding to a 3D rotation as defined in :eq:`rotation`
    :param enabled: Flags enabling the box of every source
    """

    centers: mi.Point3f
    half_extents: mi.Vector3f
    orientations: mi.Point3f
    enabled: mi.Bool

    def advance(self,
                o: mi.Point3f,
                d: mi.Vector3f,
                indices: mi.UInt) -> Tuple[mi.Point3f, mi.Float]:
        r"""
        Moves the origins ``o`` of the rays travelling along ``d`` to the
        points at which they leave the box of their source

        The returned origins are offset from the face through which the rays
        leave their box, along the normal to that face, so that the faces of
        the boxes do not intersect the rays they spawn. The origins of the rays
        whose box is disabled are returned unchanged.

        The rays are assumed to travel along unit directions, so that the
        returned distances are lengths and can be compared to the length of a
        segment. A box being convex, a distance which exceeds the length of a
        segment means that both of its endpoints lie within the box. The caller
        is then expected to leave the origin untouched, as the advanced origin
        would lie beyond the endpoint of the segment, and to test that segment
        for occlusion over its whole length.

        :param o: Origins of the rays
        :param d: Unit directions of propagation of the rays
        :param indices: Index of the source of every ray

        :return: Advanced origins of the rays, and distances by which they were
            advanced
        """

        center = dr.gather(mi.Point3f, self.centers, indices)
        half_extents = dr.gather(mi.Vector3f, self.half_extents, indices)
        orientation = dr.gather(mi.Point3f, self.orientations, indices)
        enabled = dr.gather(mi.Bool, self.enabled, indices)

        # The boxes are axis-aligned in the local frame of their source
        to_world = rotation_matrix(orientation)
        to_local = to_world.T
        o_local = to_local@(mi.Point3f(o) - center)
        d_local = to_local@mi.Vector3f(d)

        # Distance at which the ray reaches the face of the box it travels
        # towards along each of the three axes. The rays which do not travel
        # along an axis never reach the faces perpendicular to it.
        t_axis = dr.select(dr.abs(d_local) > 0.,
                           (dr.mulsign(half_extents, d_local) - o_local)
                           /d_local,
                           dr.inf)
        # The box is left through the face which is reached first
        t = dr.minimum(t_axis.x, dr.minimum(t_axis.y, t_axis.z))
        t = dr.maximum(t, 0.)

        # Outward normal to the face through which the box is left, i.e., the
        # axis reached first oriented along the direction of propagation
        exit_x = t_axis.x <= dr.minimum(t_axis.y, t_axis.z)
        exit_y = ~exit_x & (t_axis.y <= t_axis.z)
        exit_z = ~exit_x & ~exit_y
        n_local = mi.Vector3f(
            dr.select(exit_x, dr.mulsign(mi.Float(1.), d_local.x), 0.),
            dr.select(exit_y, dr.mulsign(mi.Float(1.), d_local.y), 0.),
            dr.select(exit_z, dr.mulsign(mi.Float(1.), d_local.z), 0.))

        # Offsetting along the normal to the face, rather than along the
        # direction of propagation, ensures that the rays which leave their box
        # at a grazing angle are moved off that face
        advanced = offset_p(dr.fma(mi.Point3f(d), t, mi.Point3f(o)), d,
                            to_world@n_local)

        return (dr.select(enabled, advanced, mi.Point3f(o)),
                dr.select(enabled, t, mi.Float(0.)))


def first_order_diffraction_point(s: mi.Point3f,
                                  t: mi.Point3f,
                                  o: mi.Point3f,
                                  zeta: mi.Vector3f) -> mi.Float:
    # pylint: disable=line-too-long
    r"""
    Computes the scalar parameter of the diffraction point on an edge for a
    single-order diffracted path connecting a source and a target

    This function determines the parameter :math:`x` that defines the position
    on an edge where diffraction occurs for a ray path from source
    :math:`\mathbf{s}` to target :math:`\mathbf{t}`. The edge is parameterized
    as :math:`\mathbf{o} + x \boldsymbol{\zeta}`, where :math:`\mathbf{o}` is
    the edge origin and :math:`\boldsymbol{\zeta}` is the normalized direction
    vector.

    Degenerate inputs (zero edge direction, source or target lying on the edge,
    or a vanishing diffraction construction) return ``0``. Downstream solvers
    reject that value because a valid diffraction point must satisfy
    :math:`0 < x < L` for edge length :math:`L`.


    :param s: Source position :math:`\mathbf{s}`
    :param t: Target position :math:`\mathbf{t}`
    :param o: Origin point of the edge :math:`\mathbf{o}`
    :param zeta: Normalized direction vector of the edge :math:`\boldsymbol{\zeta}`

    :return: Parameter :math:`x` defining the diffraction point position
        :math:`\mathbf{o} + x \boldsymbol{\zeta}` on the edge
    """

    ## Use edge origin as reference point
    to = t - o
    so = s - o

    ## Projection of the source and target on the edge
    sp = dr.dot(so, zeta)*zeta
    tp = dr.dot(to, zeta)*zeta

    ## Rotate the target around the edge to be coplanar
    ## with the source and the edge

    # Vectors from the edge to the source/target (perpendicular components)
    so_perp = so - sp
    to_perp = to - tp
    so_perp_norm = dr.norm(so_perp)
    to_perp_norm = dr.norm(to_perp)
    zeta_norm = dr.norm(zeta)

    # Degenerate if the edge direction vanishes or either endpoint lies on
    # the edge (no unique Keller cone / diffraction construction)
    valid = (zeta_norm > 0) & (so_perp_norm > 0) & (to_perp_norm > 0)

    # Angle of rotation
    v1 = dr.select(to_perp_norm > 0, to_perp * dr.rcp(to_perp_norm),
                   mi.Vector3f(0, 0, 1))
    v2 = dr.select(so_perp_norm > 0, so_perp * dr.rcp(so_perp_norm),
                   mi.Vector3f(0, 0, 1))
    theta = dr.pi - dr.safe_acos(dr.dot(v1, v2))

    # Axis of rotation
    s_ax = dr.dot(dr.cross(so_perp, to_perp), zeta)
    rot_axis = dr.select(s_ax < 0, -zeta, zeta)

    # Apply the rotation
    tc = rotate_vector_around_axis(to, rot_axis, theta)

    ## Compute the diffraction point

    u0_vec = tc - so
    u0, u0_norm = _safe_unit_direction(u0_vec)
    valid &= u0_norm > 0

    u1 = dr.cross(so, u0)
    u2 = dr.cross(zeta, u0)
    u2_norm = dr.norm(u2)
    valid &= u2_norm > 0
    sign = dr.sign(dr.dot(u1, u2))

    # Position of the diffraction point on the edge
    # relative to the edge origin
    x = sign * dr.norm(u1) * dr.rcp(u2_norm)
    x = dr.select(valid, x, mi.Float(0))

    return x

def sample_keller_cone(e_hat: mi.Vector3f,
                       n0: mi.Vector3f,
                       nn: mi.Vector3f,
                       sample: mi.Float,
                       ki: mi.Vector3f,
                       lit_region: bool) -> mi.Vector3f:
    r"""
    Samples a direction on the Keller cone for diffraction

    This function samples a diffracted ray direction on the Keller cone.
    The Keller cone is such that the angle between the diffracted rays and
    the edge direction is equal to the angle between the incident ray and
    the edge direction.

    :param e_hat: Normalized edge direction vector
    :param n0: Normal to the 0-face
    :param nn: Normal to the n-face
    :param sample: Random sample in [0, 1) used for sampling the Keller cone
    :param ki: Incident ray direction
    :param lit_region: If set to `True`, then the lit region is sampled.
        If set to `False`, then only the shadow region is sampled.

    :return: Normalized direction vector of the diffracted ray in the
        local coordinate system
    """

    k_diffr, _ = sample_keller_cone_with_measure(
        e_hat, n0, nn, sample, ki, lit_region)
    return k_diffr


def sample_keller_cone_with_measure(
        e_hat: mi.Vector3f,
        n0: mi.Vector3f,
        nn: mi.Vector3f,
        sample: mi.Float,
        ki: mi.Vector3f,
        lit_region: bool) -> Tuple[mi.Vector3f, mi.Float]:
    r"""
    Samples a direction on the Keller cone and returns the angular measure of
    the sampled interval.

    The angular measure is the inverse density associated with the uniform
    angular sample. It is the full exterior angle when the lit region is
    enabled and the shadow-region angle otherwise.

    :param e_hat: Normalized edge direction vector
    :param n0: Normal to the 0-face
    :param nn: Normal to the n-face
    :param sample: Random sample in [0, 1) used for sampling the Keller cone
    :param ki: Incident ray direction
    :param lit_region: If set to `True`, then the lit region is sampled.
        If set to `False`, then only the shadow region is sampled.

    :return: Normalized direction vector of the diffracted ray and angular
        measure of the sampled interval
    """

    # Local coordinate system (t0, n0, e_fwd), where
    # e_fwd is the edge direction oriented in the direction
    # of the incident ray
    t0 = dr.normalize(dr.cross(n0, e_hat))
    e_fwd = dr.select(dr.dot(e_hat, ki) > 0,
                      e_hat, -e_hat)

    # k_i in local coordinate system
    ki_local = mi.Vector3f(dr.dot(ki, t0),
                           dr.dot(ki, n0),
                           dr.dot(ki, e_fwd))

    beta0, phi_i = theta_phi_from_unit_vec(ki_local)

    # Read the wedge opening angle
    exterior_angle = dr.two_pi - wedge_interior_angle(n0, nn)

    # Sample an angle on the Keller cone
    if lit_region:
        angular_measure = exterior_angle
        phi = sample * angular_measure
    else:
        # Ensures that phi_i is in [0, 2*pi]
        phi_i = dr.select(phi_i < 0, phi_i + dr.two_pi, phi_i)
        angular_measure = exterior_angle - phi_i
        phi = phi_i + sample * angular_measure


    # Direction of propagation of diffracted ray
    sin_beta0, cos_beta0 = dr.sincos(beta0)
    sin_phi, cos_phi = dr.sincos(phi)
    k_diffr = sin_phi*sin_beta0*n0 \
        + cos_phi*sin_beta0*t0 \
        + cos_beta0*e_fwd

    return k_diffr, angular_measure
