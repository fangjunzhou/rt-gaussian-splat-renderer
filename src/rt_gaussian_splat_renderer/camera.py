import taichi as ti
import math


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
    



from typing import Tuple
import taichi as ti

from rt_gaussian_splat_renderer.ray import Ray

def cam_ray_gen(extrinsic: ti.math.mat4, intrinsic: ti.math.mat3, res: Tuple[int, int]) -> ti.field(Ray):
    """Generate camera ray based on camera extrinsic and intrinsic.

    :param extrinsic: a 4x4 camera extrinsic matrix that maps from world space 
        to view space.
    :param intrinsic: a 3x3 camera intrinsic matrix that maps from view space 
        to screen space.
    :param res: the resolution of the field.
    :return: a Ray field of resolution res.
    """
    # TODO: Implement camera ray generation.
    return Ray.field(shape=(res))
