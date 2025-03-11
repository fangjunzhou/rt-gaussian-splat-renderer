import taichi as ti
import math


@ti.dataclass
class Bound:
    """Bounding box struct in Taichi.

    :param p_min: minimum point of the bounding box.
    :param p_max: maximum point of the bounding box.
    """
    p_min: ti.math.vec3
    p_max: ti.math.vec3

    @ti.func
    def init(self, p_min=ti.math.vec3(ti.math.inf),
             p_max=ti.math.vec3(-ti.math.inf)):
        self.p_min = p_min
        self.p_max = p_max

    @ti.func
    def get_centroid(self):
        return 0.5 * (self.p_min + self.p_max)

    @ti.func
    def size(self):
        return self.p_max - self.p_min

    @ti.func
    def area(self):
        size = self.size()
        return 2 * (size[0] * size[1] +
                    size[1] * size[2] +
                    size[2] * size[0])

    @ti.func
    def bounds_union(self, box):
        pp_min = ti.min(self.p_min, box.p_min)
        pp_max = ti.max(self.p_max, box.p_max)
        result = Bound()
        result.init(pp_min, pp_max)
        return result

    @ti.func
    def hit(self, ray):
        """Ray cast hit the bounding box.

        :param ray: a camera ray.
        :type ray: Ray
        :return: hit interval. Hit if x < y. Otherwise, no hit.
        :rtype: ti.math.vec2
        """
        pxmin = self.p_min.x
        if (ray.direction.x < 0):
            pxmin = self.p_max.x
        pxmax = self.p_max.x
        if (ray.direction.x < 0):
            pxmax = self.p_min.x
        t_minx = (pxmin - ray.origin.x) / ray.direction.x
        t_maxx = (pxmax - ray.origin.x) / ray.direction.x

        pymin = self.p_min.y
        if (ray.direction.y < 0):
            pymin = self.p_max.y
        pymax = self.p_max.y
        if (ray.direction.y < 0):
            pymax = self.p_min.y
        t_miny = (pymin - ray.origin.y) / ray.direction.y
        t_maxy = (pymax - ray.origin.y) / ray.direction.y

        pzmin = self.p_min.z
        if (ray.direction.z < 0):
            pzmin = self.p_max.z
        pzmax = self.p_max.z
        if (ray.direction.z < 0):
            pzmax = self.p_min.z
        t_minz = (pzmin - ray.origin.z) / ray.direction.z
        t_maxz = (pzmax - ray.origin.z) / ray.direction.z

        t_min = ti.math.max(t_minx, t_miny, t_minz)
        t_max = ti.math.min(t_maxx, t_maxy, t_maxz)

        return ti.math.vec2(t_min, t_max)
