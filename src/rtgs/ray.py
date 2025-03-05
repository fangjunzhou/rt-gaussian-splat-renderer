import taichi as ti


@ti.dataclass
class Ray:
    """A ray representation.

    :param origin: the origin of the ray.
    :param direction: the ray direction.
    :param start: the minimum t for raycast.
    :param end: the maximun t for raycast, this should be set to ti.math.inf
        for camera rays.
    """

    origin: ti.math.vec3
    direction: ti.math.vec3
    start: ti.f32
    end: ti.f32

    @ti.func
    def init(self,
             origin=ti.math.vec3(0, 0, 0),
             direction=ti.math.vec3(0, 1, 0),
             start=0,
             end=ti.math.inf):
        """Taichi scope Ray initialization function.

        :param origin: the origin of the ray.
        :type origin: ti.math.vec3
        :param direction: the ray direction.
        :type direction: ti.math.vec3
        :param start: the minimum t for raycast.
        :type start: ti.f32
        :param end: the maximun t for raycast, this should be set to ti.math.inf
            for camera rays.
        :type end: ti.f32
        """
        self.origin = origin
        self.direction = direction
        self.start = start
        self.end = end

    @ti.func
    def get(self, t):
        """Evaluate ray position parameterized at time t.

        :param t: ray parameter t.
        :type t: ti.f32
        :return: ray position at time t.
        :rtype: ti.math.vec3
        """
        return self.origin + t * self.direction


def new_ray(origin: ti.math.vec3 = ti.math.vec3(0, 0, 0),
            direction: ti.math.vec3 = ti.math.vec3(0, 1, 0),
            start: ti.f32 = 0,
            end: ti.f32 = ti.math.inf) -> Ray:
    """Python scope ray constructor to create a new Ray.

    :param origin: the origin of the ray.
    :param direction: the ray direction.
    :param start: the minimum t for raycast.
    :param end: the maximun t for raycast, this should be set to ti.math.inf
        for camera rays.
    :return: a Taichi ray dataclass.
    """
    return Ray(origin, direction, start, end)
