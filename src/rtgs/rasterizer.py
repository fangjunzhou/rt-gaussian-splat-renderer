import taichi as ti

from rtgs.camera import Camera
from rtgs.scene import Scene
from rtgs.utils.types import vec2i


@ti.data_oriented
class Rasterizer:
    # Scene and Camera.
    scene: Scene
    camera: Camera

    # Display buffer.
    disp_buf: ti.Field

    def __init__(self,
                 buf_size: vec2i,
                 scene: Scene,
                 camera: Camera) -> None:
        self.scene = scene
        self.camera = camera

        self.disp_buf = ti.field(ti.math.vec3, shape=(buf_size.x, buf_size.y))

    def rasterize(self):
        # TODO: Implement Gaussian splat differentiable rasterization.

        # TODO: Cull viewing frustum.

        # TODO: Create tiles and duplicate gaussians with tile+depth key.

        # TODO: GPU radix sort.

        # TODO: Rasterization kernel.

        pass
