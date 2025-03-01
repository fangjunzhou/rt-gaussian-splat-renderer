import taichi as ti
import math
from typing import Tuple

from ray import Ray



@ti.data_oriented
class Camera:
    def __init__(self, position, looking_at,up, vert_fov, aspect_ratio):
        # Did not implement rotation yet!
        h = math.tan(math.radians(vert_fov) / 2.0)
        v_height = 2.0 * h
        v_width = v_height * aspect_ratio

        # direction = (looking_at - position).normalized()
        # dir=rot.rotate(direction)
        # looking_at = position + dir

        w = (position - looking_at).normalized()
        self.u = up.cross(w).normalized()
        self.v = w.cross(self.u)

        self.origin = position
        self.horizontal =  v_width * self.u
        self.vertical = v_height * self.v
        self.lower_left_corner = self.origin - (self.horizontal / 2.0) - (self.vertical / 2.0) - w

    @ti.func
    def generate_ray(self, u, v):
        return self.origin, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin
    
# @ti.kernel
# def cam_ray_gen(extrinsic: ti.math.mat4, intrinsic: ti.math.mat3, res: Tuple[int, int]) -> ti.field(Ray):
#     """Generate camera ray based on camera extrinsic and intrinsic.

#     :param extrinsic: a 4x4 camera extrinsic matrix that maps from world space 
#         to view space.
#     :param intrinsic: a 3x3 camera intrinsic matrix that maps from view space 
#         to screen space.
#     :param res: the resolution of the field.
#     :return: a Ray field of resolution res.
#     """
#     inv_intrinsic = intrinsic.inverse()  
#     inv_extrinsic = extrinsic.inverse()

#     width, height = res
#     rays = ti.field(dtype=Ray, shape=(width, height))
#     for i, j in ti.ndrange((width, height)):
#         x = (i + 0.5) / width * 2.0 - 1.0
#         y = (j + 0.5) / height * 2.0 - 1.0

#         pixel_view = inv_intrinsic @ ti.math.vec3(x, y, 1.0)  # Backproject to view space
#         pixel_world = inv_extrinsic @ ti.math.vec4(pixel_view, 1.0)
#         cam_pos = inv_extrinsic @ ti.math.vec4(0, 0, 0, 1)

#         direction = (pixel_world.xyz - cam_pos.xyz).normalized()
#         rays[i, j] = ti.Vector([cam_pos.x, cam_pos.y, cam_pos.z, direction.x, direction.y, direction.z])

#     return Ray.field(shape=(res))
