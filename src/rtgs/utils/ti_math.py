"""
Taichi math utils.
"""

import taichi as ti


@ti.func
def random_vec3():
    """Randomize vec3.

    :return: a random vec3 from (0, 0, 0) to (1, 1, 1)
    :rtype: ti.math.vec3
    """
    x = ti.random(ti.f32)
    y = ti.random(ti.f32)
    z = ti.random(ti.f32)
    return ti.math.vec3(x, y, z)
