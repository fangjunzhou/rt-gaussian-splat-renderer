import taichi as ti
import math
from typing import Tuple

from rtgs.ray import Ray, new_ray
from rtgs.utils.quaternion import rot_vec3


@ti.data_oriented
class Camera:
    position: ti.math.vec3
    rotation: ti.math.vec4
    censor_size: ti.math.vec2
    focal_length: ti.math.vec2

    def __init__(
            self,
            position: ti.math.vec3,
            rotation: ti.math.vec4,
            censor_size: ti.math.vec2,
            focal_length: ti.math.vec2):
        self.position = position
        self.rotation = rotation
        self.censor_size = censor_size
        self.focal_length = focal_length

    @ti.func
    def generate_ray(self, u, v):
        """Generate ray based on screen space coordinate. (u, v) is from (0, 0)
        to (1, 1), with (0, 0) being the bottom left of the screen and (1, 1)
        being the top right of the screen.

        :param u: horizontal u coordinate.
        :type u: ti.f32
        :param v: vertical v coordinate.
        :type v: ti.f32
        :return: camera ray.
        :rtype: Ray
        """
        px = (self.censor_size.x * u - 0.5 *
              self.censor_size.x) / self.focal_length.x
        py = (self.censor_size.y * v - 0.5 *
              self.censor_size.y) / self.focal_length.y
        pz = -1

        dir_camera = ti.math.vec3(px, py, pz)
        dir_camera = ti.math.normalize(dir_camera)

        dir_world = rot_vec3(self.rotation, dir_camera)
        return new_ray(self.position, dir_world)
