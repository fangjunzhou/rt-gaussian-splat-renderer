import taichi as ti

@ti.dataclass
class Ray:
    """A ray representation.

    :param origin: the origin of the ray.
    :param dir: the ray direction.
    :param start: the minimum t for raycast.
    :param end: the maximun t for raycast, this should be set to ti.math.inf 
        for camera rays.
    """
    origin: ti.math.vec3
    dir: ti.math.vec3
    start: ti.f32
    end: ti.f32
