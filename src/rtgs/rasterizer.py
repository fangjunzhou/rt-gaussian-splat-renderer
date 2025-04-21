from typing import List
import taichi as ti

from rtgs.camera import Camera
from rtgs.scene import Scene
from rtgs.utils.types import vec2i


@ti.dataclass
class ScreenSpaceGaussian:
    """Screen-space Gaussian.

    :param position:
    :param cov:
    """
    tile: int
    depth: float
    position: ti.math.vec2
    cov: ti.math.mat2


@ti.data_oriented
class Rasterizer:
    # Scene and Camera.
    scene: Scene
    camera: Camera

    # Display buffer.
    disp_buf: ti.Field

    offsets: List[int]
    ss_gaussian_field: ti.Field
    num_ss_gaussian: int

    def __init__(self,
                 buf_size: vec2i,
                 scene: Scene,
                 camera: Camera) -> None:
        self.scene = scene
        self.camera = camera

        self.disp_buf = ti.field(ti.math.vec3, shape=(buf_size.x, buf_size.y))
        self.ss_gaussian_field = ScreenSpaceGaussian.field(
            shape=scene.gaussian_field.shape)
        self.num_ss_gaussian = 0

    @ti.kernel
    def cull(self):
        # TODO: Implement viewing frustum culling.
        pass

    def sort(self):
        # TODO: Implement GPU radix sort.
        pass

    def rasterize_tile(self):
        # TODO: Implement tile rasterization.
        pass

    def rasterize(self):
        # TODO: Implement Gaussian splat differentiable rasterization.

        # TODO: Cull viewing frustum.

        # TODO: Create tiles and duplicate gaussians with tile+depth key.

        # TODO: GPU radix sort.

        # TODO: Rasterization kernel.

        pass
