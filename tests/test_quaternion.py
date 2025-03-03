import taichi as ti
import numpy as np
import pytest
import logging

import rt_gaussian_splat_renderer.utils.quaternion as quat


logger = logging.getLogger(__name__)

# Init taichi runtime.
ti.init(arch=ti.gpu, random_seed=42)
logger.info(f"Current Taichi backend: {ti.cfg.arch}")  # pyright: ignore


def test_mul():
    """
    Test quaternion multiplication.
    """

    @ti.kernel
    def ker_mul(p: ti.math.vec4, q: ti.math.vec4) -> ti.math.vec4:
        return quat.mul(p, q)

    p = ti.math.vec4(1, -2, 1, 3)
    q = ti.math.vec4(-1, 2, 3, 2)
    pq = ker_mul(p, q)

    assert pq == ti.math.vec4(-9, -2, 11, 8)


def test_conj():
    """
    Test quaternion complex conjugate.
    """
    @ti.kernel
    def ker_conj(q: ti.math.vec4) -> ti.math.vec4:
        return quat.conj(q)

    q = ti.math.vec4(1, 2, 3, 4)
    qc = ker_conj(q)

    assert qc == ti.math.vec4(-1, -2, -3, 4)


def test_inv():
    """
    Test quaternion inverse.
    """
    @ti.kernel
    def ker_mul(p: ti.math.vec4, q: ti.math.vec4) -> ti.math.vec4:
        return quat.mul(p, q)

    @ti.kernel
    def ker_inv(q: ti.math.vec4) -> ti.math.vec4:
        return quat.inv(q)

    q = ti.math.vec4(1, 2, 3, 4)
    qi = ker_inv(q)
    qiq = ker_mul(q, qi)

    assert qiq == ti.math.vec4(0, 0, 0, 1)


def test_axis_angle():
    """
    Test conversion between axis-angle and quaternion.
    """
    @ti.kernel
    def ker_from_axis_angle(v: ti.math.vec3) -> ti.math.vec4:
        return quat.from_axis_angle(v)

    @ti.kernel
    def ker_as_axis_angle(q: ti.math.vec4) -> ti.math.vec3:
        return quat.as_axis_angle(q)

    @ti.kernel
    def ker_length(v: ti.template()) -> ti.f32:
        return ti.math.length(v)

    v = ti.math.vec3(0, np.pi, 0)
    logger.info(f"Original axis-angle vector: {v}")
    q_from = ker_from_axis_angle(v)
    logger.info(f"Convert to quaternion: {q_from}")
    assert ker_length(
        q_from -
        ti.math.vec4(0, 1, 0, 0)) == pytest.approx(0, abs=1e-6)
    q_as = ker_as_axis_angle(q_from)
    logger.info(f"Recovered axis-angle vector: {q_as}")
    assert ker_length(q_as - v) == pytest.approx(0)


def test_rot_vec3():
    """Test vector rotation using quaternion."""
    @ti.kernel
    def ker_from_axis_angle(v: ti.math.vec3) -> ti.math.vec4:
        return quat.from_axis_angle(v)

    @ti.kernel
    def ker_rot_vec(q: ti.math.vec4, v: ti.math.vec3) -> ti.math.vec3:
        return quat.rot_vec3(q, v)

    @ti.kernel
    def ker_length(v: ti.template()) -> ti.f32:
        return ti.math.length(v)

    # Rotate (1, 0, 0) by z-axis by pi/2.
    v = ti.math.vec3(1, 0, 0)
    r1 = ti.math.vec3(0, 0, 1) * (np.pi / 2)
    q1 = ker_from_axis_angle(r1)
    v1 = ker_rot_vec(q1, v)
    assert ker_length(v1 - ti.math.vec3(0, 1, 0)) == pytest.approx(0, abs=1e-6)

    # Rotate (1, 0, 0) by y-axis by pi/2.
    r2 = ti.math.vec3(0, 1, 0) * (np.pi / 2)
    q2 = ker_from_axis_angle(r2)
    v2 = ker_rot_vec(q2, v)
    assert ker_length(v2 - ti.math.vec3(0, 0, -1)) == pytest.approx(0, abs=1e-6)


def test_as_rotation_mat3():
    """Test quaternion to 4x4 rotation matrix"""
    @ti.kernel
    def ker_rand_quat() -> ti.math.vec4:
        x = ti.random(ti.f32)
        y = ti.random(ti.f32)
        z = ti.random(ti.f32)
        w = ti.random(ti.f32)
        return ti.math.normalize(ti.math.vec4(x, y, z, w))

    @ti.kernel
    def ker_quat_to_mat3(q: ti.math.vec4) -> ti.math.mat3:
        return quat.as_rotation_mat3(q)

    @ti.kernel
    def ker_rot_with_mat3(m: ti.math.mat3, v: ti.math.vec3) -> ti.math.vec3:
        return m @ v

    @ti.kernel
    def ker_rot_vec(q: ti.math.vec4, v: ti.math.vec3) -> ti.math.vec3:
        return quat.rot_vec3(q, v)

    @ti.kernel
    def ker_length(v: ti.template()) -> ti.f32:
        return ti.math.length(v)

    # Rotate (1, 0, 0) by a random quaternion.
    v = ti.math.vec3(1, 0, 0)
    q = ker_rand_quat()
    v_ref = ker_rot_vec(q, v)
    m = ker_quat_to_mat3(q)
    v = ker_rot_with_mat3(m, v)
    assert ker_length(v - v_ref) == pytest.approx(0, abs=1e-6)


def test_as_rotation_mat4():
    """Test quaternion to 4x4 rotation matrix"""
    @ti.kernel
    def ker_rand_quat() -> ti.math.vec4:
        x = ti.random(ti.f32)
        y = ti.random(ti.f32)
        z = ti.random(ti.f32)
        w = ti.random(ti.f32)
        return ti.math.normalize(ti.math.vec4(x, y, z, w))

    @ti.kernel
    def ker_quat_to_mat4(q: ti.math.vec4) -> ti.math.mat4:
        return quat.as_rotation_mat4(q)

    @ti.kernel
    def ker_rot_with_mat4(m: ti.math.mat4, v: ti.math.vec3) -> ti.math.vec3:
        return (m @ ti.math.vec4(v.x, v.y, v.z, 1)).xyz

    @ti.kernel
    def ker_rot_vec(q: ti.math.vec4, v: ti.math.vec3) -> ti.math.vec3:
        return quat.rot_vec3(q, v)

    @ti.kernel
    def ker_length(v: ti.template()) -> ti.f32:
        return ti.math.length(v)

    # Rotate (1, 0, 0) by a random quaternion.
    v = ti.math.vec3(1, 0, 0)
    q = ker_rand_quat()
    v_ref = ker_rot_vec(q, v)
    m = ker_quat_to_mat4(q)
    v = ker_rot_with_mat4(m, v)
    assert ker_length(v - v_ref) == pytest.approx(0, abs=1e-6)
