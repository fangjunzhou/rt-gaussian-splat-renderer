"""Ray tracer Gaussian splatting renderer."""

import taichi as ti

from rtgs.camera import Camera
from rtgs.scene import Scene
from rtgs.utils.ti_math import random_vec3
from rtgs.utils.types import vec2i


@ti.data_oriented
class RayTracer:
    # Scene and Camera.
    scene: Scene
    camera: Camera

    # Render buffer.
    sample_buf: ti.Field
    attenuation_buf: ti.Field
    num_samples: ti.i32
    # Display buffer.
    disp_buf: ti.Field

    def __init__(
            self,
            buf_size: vec2i,
            scene: Scene,
            camera: Camera) -> None:
        self.scene = scene
        self.camera = camera

        self.sample_buf = ti.field(ti.math.vec3, shape=(buf_size.x, buf_size.y))
        self.attenuation_buf = ti.field(ti.f32, shape=(buf_size.x, buf_size.y))
        self.num_samples = 0
        self.disp_buf = ti.field(ti.math.vec3, shape=(buf_size.x, buf_size.y))

    def sample(self, depth: int):
        """Accumulate one sample to the sample buffer.

        :param depth: ray depth (the maximum number of gaussian hit per ray).
        """
        # Reset attenuation and camera ray.
        self.clear_attenuation()
        self.camera.generate_ray_field()
        # Ray trace depth steps.
        for _ in range(depth):
            self.sample_step()
        self.num_samples += 1

    @ti.kernel
    def clear_sample(self):
        """Clear sample buffer."""
        for i, j in self.sample_buf:
            self.attenuation_buf[i, j] = ti.math.vec3(0)

    @ti.kernel
    def clear_attenuation(self):
        """Clear light attenuation field."""
        for i, j in self.attenuation_buf:
            self.attenuation_buf[i, j] = 1

    @ti.kernel
    def generate_disp_buffer(self, num_samples: ti.i32):
        """Generate display buffer from sample buffer (average n samples)."""
        for i, j in self.disp_buf:
            self.disp_buf[i, j] = self.sample_buf[i, j] / num_samples

    @ti.kernel
    def sample_step(self):
        """One sample iteration that cast camera ray field in the scene and
        accumulate radiance into sample buffer. This kernel only accumulate
        one Gaussian.
        """
        for i, j in self.sample_buf:
            ray = self.camera.cam_ray_field[i, j]
            hit = self.scene.hit(ray)
            if hit.gaussian_idx != -1:
                gaussian = self.scene.gaussian_field[hit.gaussian_idx]
                # TODO: Evaluate gaussian color.
                self.sample_buf[i, j] += gaussian.color
                # TODO: Update camera ray field.
