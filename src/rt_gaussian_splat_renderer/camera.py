import taichi as ti
import math
from typing import Tuple

from rt_gaussian_splat_renderer.ray import Ray


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
        # Did not implement rotation yet!
        # h = math.tan(math.radians(vert_fov) / 2.0)
        # v_height = 2.0 * h
        # v_width = v_height * aspect_ratio

        # direction = (looking_at - position).normalized()
        # dir=rot.rotate(direction)
        # looking_at = position + dir

        # w = (position - looking_at).normalized()
        # self.u = up.cross(w).normalized()
        # self.v = w.cross(self.u)
        #
        # self.origin = position
        # self.horizontal = v_width * self.u
        # self.vertical = v_height * self.v
        # self.lower_left_corner = self.origin - \
        #     (self.horizontal / 2.0) - (self.vertical / 2.0) - w
        pass

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
        # return self.origin, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin
        # TODO: Implement camera ray.
        return Ray()
