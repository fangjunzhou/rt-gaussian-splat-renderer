# Quaternion

## Quaternion Representation

All the quaternion in this project are represented using scalar-last `vec4` ordered as `(x, y, z, w)`. This is different from the order in [numpy-quaternion](https://quaternion.readthedocs.io/en/latest/) library as we are following the order defined in [Taichi](https://docs.taichi-lang.org/docs/math_module#small-vector-and-matrix-types).

## Quaternion Math

The quaternion math implemented in this project follows the [CS 348A Lecture Notes](https://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf).

We implemented following quaternion math in Taichi:

- [`mul`](#rt_gaussian_splat_renderer.utils.quaternion.mul)
- [`conj`](#rt_gaussian_splat_renderer.utils.quaternion.conj)
- [`inv`](#rt_gaussian_splat_renderer.utils.quaternion.inv)
- [`from_axis_angle`](#rt_gaussian_splat_renderer.utils.quaternion.from_axis_angle)
- [`as_axis_angle`](#rt_gaussian_splat_renderer.utils.quaternion.as_axis_angle)
- [`as_rotation_matrix`](#rt_gaussian_splat_renderer.utils.quaternion.as_rotation_matrix)
