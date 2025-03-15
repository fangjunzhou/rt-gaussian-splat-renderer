# Quaternion

## Quaternion Representation

All the quaternion in this project are represented using scalar-last `vec4` ordered as `(x, y, z, w)`. This is different from the order in [numpy-quaternion](https://quaternion.readthedocs.io/en/latest/) library as we are following the order defined in [Taichi](https://docs.taichi-lang.org/docs/math_module#small-vector-and-matrix-types).

## Quaternion Math

The quaternion math implemented in this project follows the [CS 348A Lecture Notes](https://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf).

We implemented following quaternion math in Taichi:

- [`mul`](#rtgs.utils.quaternion.mul)
- [`conj`](#rtgs.utils.quaternion.conj)
- [`inv`](#rtgs.utils.quaternion.inv)
- [`from_axis_angle`](#rtgs.utils.quaternion.from_axis_angle)
- [`as_axis_angle`](#rtgs.utils.quaternion.as_axis_angle)
- [`rot_vec3`](#rtgs.utils.quaternion.rot_vec3)
- [`as_rotation_mat3`](#rtgs.utils.quaternion.as_rotation_mat3)
- [`as_rotation_mat4`](#rtgs.utils.quaternion.as_rotation_mat4)
