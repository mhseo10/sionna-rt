#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Computes the paths channel coefficients and delays"""

import mitsuba as mi
import drjit as dr
from typing import Callable, Tuple, List

from scipy.constants import speed_of_light

from sionna.rt.utils import rotation_matrix, jones_vec_dot, \
    spectrum_to_matrix_4f
from sionna.rt.antenna_pattern import antenna_pattern_to_world_implicit
from sionna.rt.constants import InteractionType
from .paths_buffer import PathsBuffer


class FieldCalculator:
    r"""
    Computes the channel coefficients and delays corresponding to traced paths

    The computation of the electric field is performed in three steps, each
    implemented by a dedicated method:

        1. The electric field is initialized for every source from the antenna
    patterns (:meth:`~evaluate_transmitter_antenna_patterns`).
        2. The electric field is "transported" by replaying the traced paths
    (:meth:`~transport_electric_field`). At every interaction with the scene,
    the transfer matrix for this interaction is computed by evaluating the
    corresponding radio material, and the transfer matrix is applied to the
    transported electric field to update it.
        3. Once paths replay is over, the paths complex-valued coefficients are
    computed by taking the dot product between the transported electric field
    and the target antenna patterns
    (:meth:`~compute_channel_coefficients`).

    Note that the electric field is transported in the world implicit frame.
    """

    def __init__(self):

        # Dr.Jit mode for running the loop that performs the transportation of
        # the electric field.
        # Symbolic mode is the fastest mode but does not currently support
        # backpropagation of gradients
        self._loop_mode = "symbolic"

    # pylint: disable=line-too-long
    def __call__(self,
                 wavelength: mi.Float | float,
                 paths: PathsBuffer,
                 samples_per_src: int,
                 diffraction: bool,
                 src_positions: mi.Point3f,
                 tgt_positions: mi.Point3f,
                 src_orientations: mi.Point3f,
                 tgt_orientations: mi.Point3f,
                 src_antenna_patterns: List[Callable[[mi.Float,mi.Float], Tuple[mi.Complex2f,mi.Complex2f]]],
                 tgt_antenna_patterns: List[Callable[[mi.Float,mi.Float], Tuple[mi.Complex2f,mi.Complex2f]]],
                 ):
        r"""
        Computes the channel coefficients and delays for the given paths

        :param wavelength: Wavelength [m]
        :param paths: Traced paths
        :param samples_per_src: Number of samples per source
        :param diffraction: If set to `True`, then the diffraction is computed
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param src_orientations: Sources orientations specified through three angles [rad] corresponding to a 3D rotation as defined in :eq:`rotation`
        :param tgt_orientations: Targets orientations specified through three angles [rad] corresponding to a 3D rotation as defined in :eq:`rotation`
        :param src_antenna_patterns: Antenna pattern of the sources
        :param tgt_antenna_patterns: Antenna pattern of the targets

        :return: Paths buffer with channel coefficients, delays, and Doppler shifts set
        """

        # Return immediately if there are no paths
        if paths.buffer_size == 0:
            paths.a = dr.zeros(mi.Complex2f, 0)
            paths.tau = dr.zeros(mi.Float, 0)
            return paths

        # Compute the channel impulse response
        with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, False):

            # Electric fields radiated by the sources
            e_fields_tx = self.evaluate_transmitter_antenna_patterns(
                paths, src_positions, tgt_positions, src_orientations,
                src_antenna_patterns)

            # Transport the electric fields along the paths
            e_fields, tau, doppler, ki_world = self.transport_electric_field(
                e_fields_tx, wavelength, paths, samples_per_src, diffraction,
                src_positions, tgt_positions)

            # Project the incident fields on the receiver antenna patterns
            a, valid_a = self.compute_channel_coefficients(
                e_fields, wavelength, ki_world, paths, tgt_orientations,
                tgt_antenna_patterns)

            # Disable paths with 0 contribution
            paths.valid &= valid_a

            paths.a = a
            paths.tau = tau
            paths.doppler = doppler

        return paths

    @property
    def loop_mode(self):
        r"""Get/set the Dr.Jit mode used to evaluate the loops that implement
        the solver. Should be one of "evaluated" or "symbolic". Symbolic mode
        (default) is the fastest one but does not support automatic
        differentiation.

        :type: "evaluated" | "symbolic"
        """
        return self._loop_mode

    @loop_mode.setter
    def loop_mode(self, mode):
        if mode not in ("evaluated", "symbolic"):
            raise ValueError("Invalid loop mode. Must be either 'evaluated'"
                             " or 'symbolic'")
        self._loop_mode = mode

    def evaluate_transmitter_antenna_patterns(
        self,
        paths: PathsBuffer,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f,
        src_orientations: mi.Point3f,
        src_antenna_patterns: List[Callable[[mi.Float,mi.Float],
                                   Tuple[mi.Complex2f,mi.Complex2f]]]
    ) -> List[mi.Vector4f]:
        r"""
        Evaluates the transmitter antenna patterns in the direction in which
        every path leaves its source

        :param paths: Traced paths
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets
        :param src_orientations: Sources orientations specified through three
                                 angles [rad] corresponding to a 3D rotation as
                                 defined in :eq:`rotation`
        :param src_antenna_patterns: Antenna pattern of the sources

        :return: For every source antenna pattern, the radiated electric field
            in the world implicit frame, represented as a Jones vector
        """

        # Source and target position for each path
        path_src_pos = dr.gather(mi.Point3f, src_positions,
                                 paths.source_indices)
        path_tgt_pos = dr.gather(mi.Point3f, tgt_positions,
                                 paths.target_indices)

        # Orientation of the sources and corresponding to-world transformation
        path_src_ort = dr.gather(mi.Point3f, src_orientations,
                                 paths.source_indices)
        src_to_world = rotation_matrix(path_src_ort)

        # Direction of propagation of the wave leaving the source
        ki_world, *_ = self.first_segment(paths, path_src_pos, path_tgt_pos)

        # Evaluate the transmitter antenna patterns in the world frame
        return [antenna_pattern_to_world_implicit(src_antenna_pattern,
                                                 src_to_world,
                                                 ki_world,
                                                 direction="out")
                for src_antenna_pattern in src_antenna_patterns]

    @dr.syntax
    def transport_electric_field(
        self,
        e_fields_tx: List[mi.Vector4f],
        wavelength: mi.Float | float,
        paths: PathsBuffer,
        samples_per_src: int,
        diffraction_enabled: bool,
        src_positions: mi.Point3f,
        tgt_positions: mi.Point3f
    ) -> Tuple[List[mi.Vector4f], mi.Float, mi.Float, mi.Vector3f]:
        # pylint: disable=line-too-long
        r"""
        Transports the electric fields radiated by the sources to the targets by
        replaying the traced paths

        At every interaction with the scene, the radio material of the
        intersected shape is evaluated to obtain the transfer matrix which is
        then applied to the transported field. The free-space propagation loss
        is applied once the replay is over, together with the scaling
        :math:`1/\sqrt{4\pi}` accounting for the power radiated by the source
        of the transported field being spread over the sphere.

        :param e_fields_tx: For every source antenna pattern, the radiated electric field in the world implicit frame, represented as a Jones vector
        :param wavelength: Wavelength [m]
        :param paths: Traced paths
        :param samples_per_src: Number of samples per source
        :param diffraction_enabled: If set to `True`, then the diffraction is computed
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets

        :return: Electric fields incident on the targets, propagation delays [s], Doppler shifts [Hz], and directions of propagation of the waves incident on the targets
        """
        num_paths = paths.buffer_size
        max_depth = paths.max_depth

        # Source and target position for each path
        path_src_pos = dr.gather(mi.Point3f, src_positions,
                                 paths.source_indices)
        path_tgt_pos = dr.gather(mi.Point3f, tgt_positions,
                                 paths.target_indices)

        # Current depth
        depth = dr.full(mi.UInt, 1, num_paths)

        # Geometry of the first path segment. Paths without any interaction,
        # i.e., LoS paths, are inactive from the start as they only consist of
        # this segment.
        # The loop below keeps track of the current and next path vertices to
        # compute the incident and scattered wave directions of propagation
        # required to evaluate the radio materials.
        ki_world, active, interaction_type, vertex, path_length = \
            self.first_segment(paths, path_src_pos, path_tgt_pos)

        # Next vertex is initialized to 0
        next_vertex = dr.zeros(mi.Point3f, num_paths)

        # Direction of the scattered wave is initialized to 0
        ko_world = dr.zeros(mi.Vector3f, num_paths)

        # The transported fields are stored in a new list to leave the fields
        # radiated by the sources unchanged
        e_fields = list(e_fields_tx)

        # Solid angle of the ray tube.
        # It is required to compute the diffusely reflected field.
        # Initialized assuming that all the rays initially spawned from the source
        # share the unit sphere equally, i.e., initialized to
        # 4*PI/samples_per_src.
        # This quantity is also used to account for the fact that paths
        # including a diffuse reflection are sampled during shoot-and-bounce
        # with a probability defined by the radio material, and that this
        # probability should be canceled to avoid an undesired weighting that
        # arises from this sampling.
        # This canceling is implemented by scaling the solid-angle by the
        # inverse square-root of the probability of sampling the path.
        solid_angle = dr.full(mi.Float, 4.*dr.pi*dr.rcp(samples_per_src),
                              num_paths)

        # Length of the ray tube.
        # Note that this is different from the length of the path, as every
        # diffuse interaction generates a new ray tube, and therefore
        # effectively "resets" the ray tube length.
        ray_tube_length = dr.copy(path_length)

        # Doppler due to moving objects [Hz]
        doppler = dr.zeros(mi.Float, num_paths)

        # For diffraction, the path length from the diffraction point to the
        # source (s') and to the target (s)
        if diffraction_enabled:
            s, s_prime = self._diffraction_compute_s_s_prime(paths,
                                                             path_src_pos,
                                                             path_tgt_pos)
        else:
            s, s_prime = 0., 0.

        # Mask indicating which paths involve a diffraction
        has_diffraction = dr.full(mi.Bool, False, num_paths)

        # Exclude the non-loop variable `wavelength` to avoid having to
        # trace the loop twice, which is expensive.
        while dr.hint(active, mode=self.loop_mode, exclude=[wavelength]):

            # Flag set to True if this is the last depth
            last_depth = depth == max_depth

            # Flag indicating if data about the next depth can be gathered
            gather_next = active & ~last_depth

            # Mask indicating if this interaction is a diffuse reflection
            diffuse = active & (interaction_type == InteractionType.DIFFUSE)

            # Flag paths involving a diffraction, as they require a specific
            # spreading factor
            if diffraction_enabled:
                has_diffraction |= active & \
                    (interaction_type == InteractionType.DIFFRACTION)

            # Next interaction type
            next_interaction_type = paths.get_interaction_type(depth+1,
                                                               gather_next)
            # If the next interaction type is None, then this is the last
            # interaction
            next_is_none = next_interaction_type == InteractionType.NONE

            # Flag indicating if this interaction is the last one
            last_interaction = active & (next_is_none | last_depth)

            # Next vertex
            # Set to the target position if this is the last interaction
            next_vertex = dr.select(last_interaction,
                                    path_tgt_pos,
                                    paths.get_vertex(depth+1, gather_next))

            # Direction of the scattered wave.
            # Only updated if the path is still active, as we need a valid
            # value after the loop ends to evaluate the receive pattern
            ko_world = dr.select(active,
                                 mi.Vector3f(next_vertex - vertex),
                                 ko_world)
            length = dr.norm(ko_world)
            ko_world *= dr.rcp(length)

            # Intersected shape
            shape = paths.get_shape(depth, active)
            shape = dr.reinterpret_array(mi.ShapePtr, shape)

            # Update the fields
            # The solid angle of the ray tube is also updated based on the
            # probability of the interaction event
            e_fields, solid_angle = self._update_field(
                shape, paths, depth, diffraction_enabled, interaction_type,
                ki_world, ko_world, e_fields, solid_angle, s, s_prime, active)

            # Update the Doppler
            self._update_doppler_shift(shape, doppler, wavelength, ki_world,
                                       ko_world, active)

            # Update the path length
            path_length += dr.select(active, length, 0.)

            # Updates the ray tube length
            ray_tube_length = dr.select(diffuse, 0.0, ray_tube_length)
            ray_tube_length += dr.select(active, length, 0.)

            # Update the solid angle
            # If a diffuse reflection is sampled, then it is set to 2PI.
            # Otherwise it is left unchanged
            solid_angle = dr.select(diffuse, dr.two_pi, solid_angle)

            # Prepare for next iteration
            depth += 1
            ki_world = dr.select(active, ko_world, ki_world)
            active &= (depth <= max_depth) & ~next_is_none
            vertex = dr.copy(next_vertex)
            interaction_type = dr.copy(next_interaction_type)

        # Scaling to apply the free-space propagation loss.
        # Paths involving a diffraction require a specific spreading factor.
        if diffraction_enabled:
            spreading_factor = dr.select(has_diffraction,
                                         dr.rcp(dr.sqrt(s*s_prime*(s+s_prime))),
                                         dr.rcp(ray_tube_length))
        else:
            spreading_factor = dr.rcp(ray_tube_length)

        # Apply the free-space propagation loss, together with the scaling
        # accounting for the power radiated by the source of the transported
        # field being spread over the sphere.
        scaling = spreading_factor*dr.rsqrt(4.*dr.pi)
        e_fields = [e_field*scaling for e_field in e_fields]

        # Delay
        tau = path_length*dr.rcp(speed_of_light)

        return e_fields, tau, doppler, ki_world

    def compute_channel_coefficients(
        self,
        e_fields: List[mi.Vector4f],
        wavelength: mi.Float | float,
        ki_world: mi.Vector3f,
        paths: PathsBuffer,
        tgt_orientations: mi.Point3f,
        tgt_antenna_patterns: List[Callable[[mi.Float,mi.Float],
                                   Tuple[mi.Complex2f,mi.Complex2f]]],
    ) -> Tuple[List[List[mi.Complex2f]], mi.Bool]:
        # pylint: disable=line-too-long
        r"""
        Computes the channel coefficients by projecting the electric fields
        incident on the targets onto the target antenna patterns

        The scaling :math:`\lambda/\sqrt{4\pi}` that accounts for the
        effective area :math:`\lambda^2/(4\pi)` of an isotropic receive
        antenna is applied by this method, as it applies once per path
        whereas the free-space propagation loss applies to every segment of a
        path.

        :param e_fields: For every source antenna pattern, the electric field incident on the target in the world implicit frame, represented as a Jones vector
        :param wavelength: Wavelength [m]
        :param ki_world: Directions of propagation of the waves incident on the targets
        :param paths: Traced paths
        :param tgt_orientations: Targets orientations specified through three angles [rad] corresponding to a 3D rotation as defined in :eq:`rotation`
        :param tgt_antenna_patterns: Antenna pattern of the targets

        :return: Channel coefficients, where ``a[n][m]`` corresponds to the
            :math:`n` th target antenna pattern and :math:`m` th source antenna
            pattern, and mask set to `False` for paths with a zero coefficient
            for all pairs of antenna patterns
        """

        # Evaluate the receiver antenna patterns in the world frame
        path_tgt_ort = dr.gather(mi.Point3f, tgt_orientations,
                                 paths.target_indices)
        tgt_to_world = rotation_matrix(path_tgt_ort)

        tgt_patterns = [antenna_pattern_to_world_implicit(tgt_antenna_pattern,
                                                          tgt_to_world,
                                                          -ki_world,
                                                          direction="in")
                        for tgt_antenna_pattern in tgt_antenna_patterns]

        # Scaling accounting for the effective area of the receive antenna,
        # which is `wavelength**2/(4*pi)` for an isotropic antenna and enters
        # the field as its square-root.
        # It is applied once per path, whereas the free-space propagation loss
        # is applied to every segment of a path.
        scaling = wavelength*dr.rsqrt(4.*dr.pi)

        # Compute the channel coefficients by projecting the incident fields
        # onto the receiver antenna patterns
        a = []
        valid_a = mi.Bool(False)
        for tgt_pattern in tgt_patterns:
            a.append([])
            for e_field in e_fields:
                a_ = jones_vec_dot(tgt_pattern, e_field)*scaling
                valid_a |= (dr.abs(a_) > 0.)
                a[-1].append(a_)

        return a, valid_a

    def first_segment(self,
                      paths: PathsBuffer,
                      path_src_pos: mi.Point3f,
                      path_tgt_pos: mi.Point3f
                      ) -> Tuple[mi.Vector3f, mi.Bool, mi.UInt, mi.Point3f,
                                 mi.Float]:
        # pylint: disable=line-too-long
        r"""
        Computes the geometry of the first segment of every path, i.e., the one
        that starts at the source

        Paths without any interaction, i.e., line-of-sight paths, only consist
        of this segment and therefore end at the target.

        :param paths: Traced paths
        :param path_src_pos: Position of the source of every path
        :param path_tgt_pos: Position of the target of every path

        :return: Direction of propagation of the wave leaving the source, mask set to `True` for paths with at least one interaction, type of the first interaction, end point of the first segment, and length of the first segment [m]
        """

        interaction_type = paths.get_interaction_type(1, True)

        has_interaction = ( (paths.max_depth >= 1) &
                            (interaction_type != InteractionType.NONE) &
                            paths.valid )

        # End point of the segment: first vertex, or target for LoS paths
        vertex = dr.select(has_interaction, paths.get_vertex(1, True),
                           path_tgt_pos)

        # Direction of propagation of the wave leaving the source
        k_world = mi.Vector3f(vertex - path_src_pos)
        length = dr.norm(k_world)
        k_world *= dr.rcp(length)

        return k_world, has_interaction, interaction_type, vertex, length

    ##################################################
    # Internal methods
    ##################################################

    def _update_field(self,
                      shape: mi.ShapePtr,
                      paths: PathsBuffer,
                      depth: mi.UInt,
                      diffraction_enabled: bool,
                      interaction_type: mi.UInt,
                      ki_world: mi.Vector3f,
                      ko_world: mi.Vector3f,
                      e_fields: List[mi.Vector4f],
                      solid_angle: mi.Float,
                      s: mi.Float,
                      s_prime: mi.Float,
                      active: mi.Bool
                      ) -> Tuple[List[mi.Vector4f], mi.Float]:
        # pylint: disable=line-too-long
        r"""
        Evaluates the radio material and updates the electric fields accordingly

        :param shape: Intersected shape
        :param paths: Buffer containing path information
        :param depth: Current depth/interaction index in the path
        :param diffraction_enabled: If set to `True`, then the diffraction is computed
        :param interaction_type: Interaction type to evaluate, represented using :data:`~sionna.rt.constants.InteractionType`
        :param ki_world: Directions of propagation of the incident waves in the world frame
        :param ko_world: Directions of propagation of the scattered waves in the world frame
        :param e_fields: Incident electric fields, each represented by a Jones vector stored as a 4D real-valued vector
        :param solid_angle: Ray tube solid angles [sr]
        :param s: Distance from the diffraction point to the target
        :param s_prime: Distance from the source to the diffraction point
        :param active: Mask to specify active rays

        :return: Updated electric fields and updated ray tube solid angle [sr]
        """

        # Radio material of the intersected shape
        rm = shape.bsdf()

        num_paths = paths.buffer_size

        # Normal to the intersected surface in the world frame
        normal_world = paths.get_primitive_props(depth, return_normal=True,
                                                 return_vertices=False, active=active)

        # Build a surface interaction object and context object to call the
        # radio material
        si = dr.zeros(mi.SurfaceInteraction3f, num_paths)
        ctx = mi.BSDFContext(mode=mi.TransportMode.Importance,
                             type_mask=0, component=0)
        # If diffraction is globally disabled, we can avoid running the related code to
        # speed up the computation
        if diffraction_enabled:
            ctx.component |= InteractionType.DIFFRACTION

        # Ensure the normal is oriented in the opposite of the direction of
        # propagation of the incident wave
        normal_world *= dr.sign(dr.dot(normal_world, -ki_world))
        si.n = normal_world
        si.sh_frame.n = normal_world
        si.initialize_sh_frame()

        # Set `si.wi` to the local direction of propagation of the incident wave
        si.wi = si.to_local(ki_world)
        # Interaction point
        si.p = paths.get_vertex(depth, active)
        # Intersected shape
        si.shape = shape
        # Intersected primitive
        si.prim_index = paths.get_primitive(depth, active)

        if diffraction_enabled:
            wedges = paths.diffracting_wedges
            # `si.dn_du` stores the edge vector in the local frame
            si.dn_du = si.to_local(wedges.e_hat)
            # `si.dn_dv` stores the normal to the n-face in the local frame
            si.dn_dv = si.to_local(wedges.nn)
        # `si.dp_du` stores the path length from the diffraction point to the
        # source, target, and the interaction type.
        si.dp_du = mi.Vector3f(s,
                               s_prime,
                               dr.reinterpret_array(mi.Float,
                                                    interaction_type))

        # Probability of the event to be sampled
        probs = paths.get_prob(depth, active)
        # Scale the solid angle accordingly. A zero probability (no available
        # interaction) must not produce an infinite solid angle; it contributes
        # nothing and is handled by the NONE event path.
        solid_angle[active] *= dr.select(probs > 0., dr.rcp(probs), mi.Float(0.))

        # `si.t` stores the solid angle
        si.t = solid_angle

        # Update the fields
        updated_e_fields = []
        for e_field in e_fields:
            # `si.duv_dx` and `si.duv_dy` stores the incident field
            si.duv_dx = mi.Vector2f(e_field.x, # S
                                    e_field.y) # P
            si.duv_dy = mi.Vector2f(e_field.z, # S
                                    e_field.w) # P

            # Evaluate the radio material
            jones_mat = rm.eval(ctx, si, ko_world, active)
            jones_mat = spectrum_to_matrix_4f(jones_mat)
            # Update the field by applying the Jones matrix
            updated_e_fields.append(dr.select(active, jones_mat@e_field,
                                              e_field))

        return updated_e_fields, solid_angle

    @dr.syntax
    def _diffraction_compute_s_s_prime(self,
                                       paths: PathsBuffer,
                                       src_positions: mi.Point3f,
                                       tgt_positions: mi.Point3f
                                       ) -> Tuple[mi.Float, mi.Float]:
        r"""
        Computes the distances s and s' for diffraction calculations

        For each path, this function computes:
        - s: Total distance from the diffraction point to the target
        - s': Total distance from the source to the diffraction point

        These distances are used in the diffraction field calculations to
        determine the appropriate Fresnel diffraction coefficients.

        :param paths: Buffer containing the paths
        :param src_positions: Positions of the sources
        :param tgt_positions: Positions of the targets

        :return: s, s_prime -- Distance from diffraction point to target for each
            path, Distance from source to diffraction point for each path
        """

        num_paths = paths.buffer_size

        s = dr.zeros(mi.Float, num_paths)
        s_prime = dr.zeros(mi.Float, num_paths)

        active = dr.full(mi.Bool, True, num_paths)
        depth = dr.full(mi.UInt, 1, num_paths)
        max_depth = paths.max_depth
        prev_vertex = dr.copy(src_positions)
        post_diffraction = dr.full(mi.Bool, False, num_paths)
        while dr.hint(active, mode=self.loop_mode):

            interaction_type = paths.get_interaction_type(depth, active)

            diffraction = active & (interaction_type == InteractionType.DIFFRACTION)
            none = active & (interaction_type == InteractionType.NONE)

            vertex = paths.get_vertex(depth, active)
            length = dr.norm(vertex - prev_vertex)

            s[post_diffraction & active] += length
            s_prime[~post_diffraction & active] += length

            post_diffraction |= diffraction
            prev_vertex = dr.select(active, vertex, prev_vertex)
            depth += 1
            active &= (depth <= max_depth) & ~none

        last_segment_length = dr.norm(tgt_positions - prev_vertex)
        s[post_diffraction] += last_segment_length
        s_prime[~post_diffraction] += last_segment_length

        return s, s_prime

    def _update_doppler_shift(self,
                              shape: mi.ShapePtr,
                              doppler: mi.Float,
                              wavelength: mi.Float,
                              ki_world: mi.Vector3f,
                              ko_world: mi.Vector3f,
                              active: mi.Bool) -> None:
        r"""
        Updates the Doppler shifts [Hz] of paths by adding the shift due to
        the interaction

        The ``doppler`` array is updated in-place.

        :param shape: Intersected shape
        :param doppler: Array of Doppler shifts to update
        :param wavelength: Wavelength [m]
        :param ki_world: Direction of propagation of the incident wave
        :param ko_world: Direction of propagation of the scattered wave
        :param active: Flag indicating the active paths
        """

        num_paths = dr.shape(active)[0]

        # Velocity vector of the intersected object
        v_world = shape.eval_attribute_3("velocity",
                                         dr.zeros(mi.SurfaceInteraction3f, num_paths),
                                         active)

        # Effective velocity [m/s]
        v_effective = dr.dot(ko_world - ki_world, v_world)

        # Doppler shift due to this interaction [Hz]
        doppler_interaction = v_effective/wavelength
        doppler_interaction = dr.select(active, doppler_interaction, 0.)

        # Add contribution
        doppler += doppler_interaction
