"""
Gaussian struct in Taichi.
"""

from math import sqrt
import taichi as ti
import numpy as np
from rtgs.bounding_box import Bound
from rtgs.ray import Ray
import rtgs.utils.quaternion as quat


BOUNDING_THRESHOLD = 1


# Spherical harmonics coefficient.
c_0 = sqrt(3 / np.pi)
c_1 = sqrt(15 / np.pi)
c_2 = sqrt(5 / np.pi)
c_3 = sqrt(35 / (2 * np.pi))
c_4 = sqrt(105 / np.pi)
c_5 = sqrt(21 / (2 * np.pi))
c_6 = sqrt(7 / np.pi)


@ti.dataclass
class Gaussian:
    """Gaussian struct in Taichi.

    :param position: gaussian center (mean).
    :param rotation: gaussian rotation quaternion.
    :param scale: gaussian scale.
    """
    position: ti.math.vec3
    rotation: ti.math.vec4
    scale: ti.math.vec3

    color: ti.math.vec3
    opacity: ti.f32

    sh_10: ti.math.vec3
    sh_11: ti.math.vec3
    sh_12: ti.math.vec3
    sh_20: ti.math.vec3
    sh_21: ti.math.vec3
    sh_22: ti.math.vec3
    sh_23: ti.math.vec3
    sh_24: ti.math.vec3
    sh_30: ti.math.vec3
    sh_31: ti.math.vec3
    sh_32: ti.math.vec3
    sh_33: ti.math.vec3
    sh_34: ti.math.vec3
    sh_35: ti.math.vec3
    sh_36: ti.math.vec3

    # TODO: Implement spherical harmonics.

    @ti.func
    def init(self,
             position=ti.math.vec3(0, 0, 0),
             rotation=ti.math.vec4(0, 0, 0, 1),
             scale=ti.math.vec3(1, 1, 1),
             color=ti.math.vec3(1, 0, 1),
             opacity=1,
             ):
        """Taichi scope Gaussian initialization method.

        :param position: gaussian center (mean).
        :type position: ti.math.vec3
        :param rotation: gaussian rotation quaternion.
        :type rotation: ti.math.vec4
        :param scale: gaussian scale.
        :type scale: ti.math.vec3
        :param color: gaussian base color.
        :type color: ti.math.vec3
        :param opacity: gaussian opacity.
        :type opacity: ti.f32
        """
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.color = color
        self.opacity = opacity

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

    @ti.func
    def bounding_box(self):
        """Compute the axis-aligned bounding box (AABB) for the Gaussian
        considering rotation.

        :return: bounding box for the Gaussian.
        :rtype: Bound
        """
        # TODO: Implement more accurate ellipsoid bounding box.
        rotation_mat = quat.as_rotation_mat3(self.rotation)
        scale_mat = ti.math.mat3([
            [self.scale.x, 0, 0],
            [0, self.scale.y, 0],
            [0, 0, self.scale.z]
        ])

        # six endpoints of the gaussian
        p1 = self.position + rotation_mat @ (scale_mat @ ti.math.vec3(1, 0, 0))
        p2 = self.position - rotation_mat @ (scale_mat @ ti.math.vec3(1, 0, 0))
        p3 = self.position + rotation_mat @ (scale_mat @ ti.math.vec3(0, 1, 0))
        p4 = self.position - rotation_mat @ (scale_mat @ ti.math.vec3(0, 1, 0))
        p5 = self.position + rotation_mat @ (scale_mat @ ti.math.vec3(0, 0, 1))
        p6 = self.position - rotation_mat @ (scale_mat @ ti.math.vec3(0, 0, 1))

        # compute the bounding box for the six points
        min_bound = ti.min(p1, p2, p3, p4, p5, p6)
        max_bound = ti.max(p1, p2, p3, p4, p5, p6)

        return Bound(min_bound, max_bound)

    @ti.func
    def eval_sh(self, dir):
        """Evaluate spherical harmonics.

        :param dir: normalized view direction.
        :type dir: ti.math.vec3
        :return: spherical harmonics radiance.
        :rtype: ti.math.vec3
        """
        y_10 = 0.5 * c_0 * dir.y
        y_11 = 0.5 * c_0 * dir.z
        y_12 = 0.5 * c_0 * dir.x
        y_20 = 0.5 * c_1 * dir.x * dir.y
        y_21 = 0.5 * c_1 * dir.y * dir.z
        y_22 = 0.25 * c_2 * (3 * dir.z ** 2 - 1)
        y_23 = 0.5 * c_1 * dir.x * dir.z
        y_24 = 0.25 * c_1 * (dir.x ** 2 - dir.y ** 2)
        y_30 = 0.25 * c_3 * dir.y * (3 * dir.x ** 2 - dir.y ** 2)
        y_31 = 0.5 * c_4 * dir.x * dir.y * dir.z
        y_32 = 0.25 * c_5 * dir.y * (5 * dir.z**2 - 1)
        y_33 = 0.25 * c_6 * (5 * dir.z ** 2 - 3 * dir.z)
        y_34 = 0.25 * c_5 * dir.x * (5 * dir.z ** 2 - 1)
        y_35 = 0.25 * c_4 * (dir.x ** 2 - dir.y ** 2) * dir.z
        y_36 = 0.25 * c_3 * dir.x * (dir.x ** 2 - 3 * dir.y ** 2)

        return (
            y_10 * self.sh_10 +
            y_11 * self.sh_11 +
            y_12 * self.sh_12 +
            y_20 * self.sh_20 +
            y_21 * self.sh_21 +
            y_22 * self.sh_22 +
            y_23 * self.sh_23 +
            y_24 * self.sh_24 +
            y_30 * self.sh_30 +
            y_31 * self.sh_31 +
            y_32 * self.sh_32 +
            y_33 * self.sh_33 +
            y_34 * self.sh_34 +
            y_35 * self.sh_35 +
            y_36 * self.sh_36
        )

    @ti.func
    def eval(self, pos, dir):
        """Evaluate gaussian color.

        :param pos: evaluate position.
        :type pos: ti.math.vec3
        :param dir: ray direction for SH color encoding.
        :type dir: ti.math.vec3
        :return: gaussian color and alpha at pos from direction dir.
        :rtype: ti.math.vec4
        """
        # Evaluate gaussian
        d = pos - self.position
        cov_inv = ti.math.inverse(self.cov())
        rho = ti.math.exp(- d.dot(cov_inv @ d))
        alpha = self.opacity * rho
        color = self.color
        color += self.eval_sh(ti.math.normalize(dir))
        return ti.math.vec4(color.x, color.y, color.z, alpha)

    @ti.func
    def hit(self, ray):
        """Ray-Gaussian intersection test. The algorithm will only test the
        intersection for t between ray.start and ray.end.

        :param ray: camera ray.
        :type ray: Ray
        :return: two ray Gaussian intersections in increasing order. If
            there's no solution, the result will be both ti.math.inf.
        :rtype: ti.math.vec2
        """
        # Ray-Gaussian intersection.
        cov_inv = ti.math.inverse(self.cov())
        A = ray.direction.dot(cov_inv @ ray.direction)
        B = 2 * ray.direction.dot(cov_inv @ (ray.origin - self.position))
        C = (ray.origin - self.position).dot(cov_inv @
                                             (ray.origin - self.position)) - BOUNDING_THRESHOLD
        delta = B**2 - 4 * A * C
        result = ti.math.vec2(ti.math.inf, ti.math.inf)

        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2 * A)
            t2 = (-B + ti.sqrt(delta)) / (2 * A)
            result = ti.math.vec2(t1, t2)
        elif delta == 0:
            t = -B / (2 * A)
            result = ti.math.vec2(t, ti.math.inf)
        return result


def new_gaussian(position: ti.math.vec3 = ti.math.vec3(0, 0, 0),
                 rotation: ti.math.vec4 = ti.math.vec4(0, 0, 0, 1),
                 scale: ti.math.vec3 = ti.math.vec3(1, 1, 1),
                 color: ti.math.vec3 = ti.math.vec3(1, 0, 1),
                 opacity: ti.f32 = 1) -> Gaussian:
    """Python scope Gaussian constructor to create a new Gaussian.

    :param position: gaussian center (mean).
    :param rotation: gaussian rotation quaternion.
    :param scale: gaussian scale.
    :param color: gaussian base color.
    :param opacity: gaussian opacity.
    :return: a Gaussian dataclass.
    """
    return Gaussian(position, rotation, scale, color, opacity)
