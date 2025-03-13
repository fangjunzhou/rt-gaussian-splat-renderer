"""
Gaussian struct in Taichi.
"""

import taichi as ti
from rtgs.bounding_box import Bound
from rtgs.ray import Ray
import rtgs.utils.quaternion as quat


BOUNDING_THRESHOLD = 1


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
        
        axis_x = rotation_mat @ (scale_mat @ ti.math.vec3(1, 0, 0))
        axis_y = rotation_mat @ (scale_mat @ ti.math.vec3(0, 1, 0))
        axis_z = rotation_mat @ (scale_mat @ ti.math.vec3(0, 0, 1))

        abs_extents = ti.abs(axis_x) + ti.abs(axis_y) + ti.abs(axis_z)

        min_bound = self.position - abs_extents
        max_bound = self.position + abs_extents

        return Bound(min_bound, max_bound)

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
