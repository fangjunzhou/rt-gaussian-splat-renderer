import taichi as ti


@ti.dataclass
class Gaussian:
    """Gaussian struct.

    :param position: gaussian center (mean).
    :param rotation: gaussian rotation quaternion.
    :param scale: gaussian scale.
    """
    position: ti.math.vec3
    rotation: ti.math.vec4
    scale: ti.math.vec3


def new_gaussian(position: ti.math.vec3 = ti.math.vec3(0, 0, 0),
                 rotation: ti.math.vec4 = ti.math.vec4(0, 0, 0, 1),
                 scale: ti.math.vec3 = ti.math.vec3(1, 1, 1)) -> Gaussian:  # pyright: ignore
    """Python scope Gaussian constructor to create a new Gaussian.

    :param position: gaussian center (mean).
    :param rotation: gaussian rotation quaternion.
    :param scale: gaussian scale.
    :return: a Gaussian dataclass.
    """
    return Gaussian(position, rotation, scale)
