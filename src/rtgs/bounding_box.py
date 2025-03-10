import taichi as ti
import math

@ti.dataclass
class Bounds:
    """Bounding box struct in Taichi.

    :param p_min: minimum point of the bounding box.
    :param p_max: maximum point of the bounding box.
    """
    p_min: ti.math.vec3
    p_max: ti.math.vec3

    @ti.func
    def __init__(self, p_min=ti.math.vec3(0), p_max=ti.math.vec3(0)):
        self.p_min = p_min
        self.p_max = p_max

    @ti.func
    def get_centroid(self):
        return 0.5 * (self.p_min + self.p_max)
    
    @ti.func
    def bounds_union(self, box):
        pp_min = min(self.p_min, box.p_min)
        pp_max = max(self.p_max, box.p_max)
        return Bounds(p_min=pp_min, p_max=pp_max)
    
    @ti.func
    def hit(self, ray):
        """Ray cast hit the bounding box.

        :param ray: a camera ray.
        :type ray: Ray
        :return: hit information.
        :rtype: bool
        """
        t_min = (self.p_min - ray.origin) / ray.direction
        t_max = (self.p_max - ray.origin) / ray.direction
        t_min = ti.math.vec3(
            min(t_min.x, t_max.x),
            min(t_min.y, t_max.y),
            min(t_min.z, t_max.z)
        )
        t_max = ti.math.vec3(
            max(t_min.x, t_max.x),
            max(t_min.y, t_max.y),
            max(t_min.z, t_max.z)
        )
        return (t_min.x <= 0 and t_max.x >= 0) \
            or (t_min.y <= 0 and t_max.y >= 0) \
            or (t_min.z <= 0 and t_max.z >= 0)
