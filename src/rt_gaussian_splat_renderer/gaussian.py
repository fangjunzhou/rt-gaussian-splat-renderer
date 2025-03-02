"""
Gaussian struct in Taichi.
"""

import taichi as ti
import rt_gaussian_splat_renderer.utils.quaternion as quat


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

    @ti.func
    def init(self,
             position=ti.math.vec3(0, 0, 0),
             rotation=ti.math.vec4(0, 0, 0, 1),
             scale=ti.math.vec3(1, 1, 1)):
        """Taichi scope Gaussian initialization method.

        :param position: gaussian center (mean).
        :type position: ti.math.vec3
        :param rotation: gaussian rotation quaternion.
        :type rotation: ti.math.vec4
        :param scale: gaussian scale.
        :type scale: ti.math.vec3
        """
        self.position = position
        self.rotation = rotation
        self.scale = scale

    @ti.func
    def cov(self):
        """Get the covariance matrix of the gaussian using the rotation
        quaternion and scale vector.

        :return: a 3x3 covariance matrix.
        :rtype: ti.math.mat3
        """
        rotation_mat = quat.as_rotation_mat3(self.rotation)
        scale_mat = ti.math.mat3([
            [self.scale.x, 0, 0],
            [0, self.scale.y, 0],
            [0, 0, self.scale.z]
        ])
        cov_mat = rotation_mat @ scale_mat \
            @ scale_mat.transpose() @ rotation_mat.transpose()
        return cov_mat


def new_gaussian(position: ti.math.vec3 = ti.math.vec3(0, 0, 0),
                 rotation: ti.math.vec4 = ti.math.vec4(0, 0, 0, 1),
                 scale: ti.math.vec3 = ti.math.vec3(1, 1, 1)) -> Gaussian:
    """Python scope Gaussian constructor to create a new Gaussian.

    :param position: gaussian center (mean).
    :param rotation: gaussian rotation quaternion.
    :param scale: gaussian scale.
    :return: a Gaussian dataclass.
    """
    return Gaussian(position, rotation, scale)
