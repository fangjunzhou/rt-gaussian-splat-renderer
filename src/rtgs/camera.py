import taichi as ti

from rtgs.ray import Ray, new_ray
from rtgs.utils.quaternion import rot_vec3


@ti.data_oriented
class Camera:
    position: ti.math.vec3
    rotation: ti.math.vec4
    buf_size: ti.types.vector(2, ti.i32)
    censor_size: ti.math.vec2
    focal_length: ti.math.vec2
    cam_ray_field: ti.StructField

    def __init__(
            self,
            position: ti.math.vec3,
            rotation: ti.math.vec4,
            buf_size: ti.types.vector(2, ti.i32),
            focal_length: ti.math.vec2):
        self.position = position
        self.rotation = rotation
        self.buf_size = buf_size
        self.censor_size = buf_size
        self.focal_length = focal_length

        self.cam_ray_field = Ray.field(shape=(buf_size.x, buf_size.y))

    @ti.func
    def generate_ray(self, uv: ti.math.vec2) -> Ray:
        """Generate ray based on screen space coordinate. (u, v) is from (0, 0)
        to (1, 1), with (0, 0) being the bottom left of the screen and (1, 1)
        being the top right of the screen.

        :param uv: screen space coordinate.
        :return: camera ray.
        """
        pxy = (self.censor_size * uv - 0.5 *
               self.censor_size) / self.focal_length

        dir_camera = ti.math.vec3(pxy.x, pxy.y, -1)
        dir_camera = ti.math.normalize(dir_camera)

        dir_world = rot_vec3(self.rotation, dir_camera)
        ray = Ray()
        ray.init(self.position, dir_world)
        return ray

    @ti.kernel
    def generate_ray_field(self):
        """Taichi kernel to generate camera ray field."""
        for i, j in self.cam_ray_field:
            uv = ti.math.vec2(
                (i + 0.5) / self.buf_size.x,
                (j + 0.5) / self.buf_size.y)
            self.cam_ray_field[i, j] = self.generate_ray(uv)
