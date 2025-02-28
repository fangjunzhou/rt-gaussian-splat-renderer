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
