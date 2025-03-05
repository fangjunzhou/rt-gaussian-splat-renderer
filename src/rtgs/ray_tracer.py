import taichi as ti

from rtgs.camera import Camera
from rtgs.scene import Scene


@ti.data_oriented
class RayTracer:
    sample_buf: ti.field(ti.math.vec3)
    attenuation_buf: ti.field(ti.f32)

    @ti.kernel
    def sample_iter(self, scene: Scene, cam: Camera):
        """One sample iteration that cast camera ray field in the scene and
        accumulate radiance into sample buffer.

        :param scene: the scene to path trace.
        :param cam: the main camera of the scene.
        """
        pass
