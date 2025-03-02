"""
Quaternion utils functions in Taichi.
"""

import taichi as ti


@ti.func
def mul(p, q):
    """Quaternion multiplication.

    :param p: left quaternion.
    :type p: ti.math.vec4
    :param q: right quaternion.
    :type q: ti.math.vec4
    :return: multiplication result pq.
    :rtype: ti.math.vec4
    """
    # Scalar.
    w = p.w * q.w - ti.math.dot(p.xyz, q.xyz)
    # Vector.
    v = p.w * q.xyz + q.w * p.xyz + ti.math.cross(p.xyz, q.xyz)
    return ti.math.vec4(v.x, v.y, v.z, w)


@ti.func
def conj(q):
    """The complex conjugate of a quaternion.

    :param q: input quaternion.
    :type q: ti.math.vec4
    :return: the complex conjugate of q.
    :rtype: ti.math.vec4
    """
    return ti.math.vec4(-q.x, -q.y, -q.z, q.w)


@ti.func
def inv(q):
    """The inverse of a quaternion.

    :param q: input quaternion.
    :type q: ti.math.vec4
    :return: the inverse of q.
    :rtype: ti.math.vec4
    """
    return conj(q) / ti.math.length(q)


@ti.func
def from_axis_angle(v):
    """Convert a vector in axis-angle representation to quaternion.

    :param v: axis-angle represented vector with v pointing to the direction
        of the rotation and length of v equals to the angle to rotate.
    :type v: ti.math.vec3
    :return: rotation quaternion.
    :rtype: ti.math.vec4
    """
    theta = ti.math.length(v)
    if theta > 0:
        v = ti.math.normalize(v) * ti.math.sin(theta / 2)
    w = ti.math.cos(theta / 2)
    return ti.math.vec4(v.x, v.y, v.z, w)


@ti.func
def as_axis_angle(q):
    """Convert a quaternion into axis-angle representation.

    :param q: a unit quaternion.
    :type q: ti.math.vec4
    :return: axis-angle representation of q
    :rtype: ti.math.vec3
    """
    theta = ti.math.acos(q.w) * 2
    norm = ti.math.length(q.xyz)
    res = ti.math.vec3(0)
    if norm > 0:
        res = q.xyz / norm * theta
    return res


@ti.func
def rot_vec3(q, v):
    """Rotate a vector v using a quaternion q.

    :param q: quaternion q.
    :type q: ti.math.vec4
    :param v: vector v.
    :type v: ti.math.vec3
    :return: the rotated vector v using quaternion q (as $q\\vec{v}q^{*}$).
    :rtype: ti.math.vec3
    """
    qv = ti.math.vec4(v.x, v.y, v.z, 0)
    return mul(q, mul(qv, conj(q))).xyz


@ti.func
def as_rotation_mat4(q):
    """Convert a vec4 quaternion to a 4x4 rotation matrix in homogeneous
    coordinate.

    :param q: a normalized quaternion.
    :type q: ti.math.vec4
    :return: a 4x4 rotation matrix in homogeneous coordinate.
    :rtype: ti.math.mat4
    """
    mx = mul(q, mul(ti.math.vec4(1, 0, 0, 0), conj(q)))
    my = mul(q, mul(ti.math.vec4(0, 1, 0, 0), conj(q)))
    mz = mul(q, mul(ti.math.vec4(0, 0, 1, 0), conj(q)))
    m = ti.math.eye(4)
    m[0, 0] = mx.x
    m[1, 0] = mx.y
    m[2, 0] = mx.z
    m[0, 1] = my.x
    m[1, 1] = my.y
    m[2, 1] = my.z
    m[0, 2] = mz.x
    m[1, 2] = mz.y
    m[2, 2] = mz.z
    return m
