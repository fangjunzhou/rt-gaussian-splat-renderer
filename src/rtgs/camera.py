import taichi as ti

from rtgs.ray import Ray, new_ray
from rtgs.utils.quaternion import rot_vec3
from rtgs.utils.types import vec2i


@ti.data_oriented
class Camera:
    position: ti.math.vec3
    rotation: ti.math.vec4
    buf_size: vec2i
    censor_size: ti.math.vec2
    focal_length: ti.math.vec2
    cam_ray_field: ti.StructField

    def __init__(
            self,
            position: ti.math.vec3,
            rotation: ti.math.vec4,
            buf_size: vec2i,
            focal_length: ti.math.vec2):
        self.position = position
        self.rotation = rotation
        self.buf_size = buf_size
        self.censor_size = buf_size
        self.focal_length = focal_length

        self.cam_ray_field = Ray.field(shape=(buf_size.x, buf_size.y))

    @ti.func
    def generate_ray(self, position, rotation, uv):
        """Generate ray based on screen space coordinate. (u, v) is from (0, 0)
        to (1, 1), with (0, 0) being the bottom left of the screen and (1, 1)
        being the top right of the screen.

        :param position: camera position vec3.
        :type position: ti.math.vec3
        :param rotation: camera rotation quaternion.
        :type rotation: ti.math.vec4
        :param uv: screen space coordinate from (0, 0) to (1, 1).
        :type uv: ti.math.vec2
        :return: camera ray.
        :rtype: Ray
        """
        pxy = (self.censor_size * uv - 0.5 *
               self.censor_size) / self.focal_length

        dir_camera = ti.math.vec3(pxy.x, pxy.y, -1)
        dir_camera = ti.math.normalize(dir_camera)

        dir_world = rot_vec3(rotation, dir_camera)
        ray = Ray()
        ray.init(position, dir_world)
        return ray

    @ti.kernel
    def generate_ray_field(
            self,
            position: ti.math.vec3,
            rotation: ti.math.vec4):
        """Taichi kernel to generate camera ray field.

        :param position: camera position vec3.
        :param rotation: camera rotation quaternion.
        """
        for i, j in self.cam_ray_field:
            uv = ti.math.vec2(
                (i + 0.5) / self.buf_size.x,
                (j + 0.5) / self.buf_size.y)
            self.cam_ray_field[i, j] = self.generate_ray(position, rotation, uv)
