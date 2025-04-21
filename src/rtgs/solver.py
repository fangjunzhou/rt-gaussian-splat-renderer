import taichi as ti
import numpy as np

from rtgs.camera import Camera
from rtgs.rasterizer import Rasterizer
from rtgs.scene import Scene
from rtgs.utils.types import vec2i


@ti.data_oriented
class Solver:
    # Scene and Camera.
    scene: Scene

    # State.
    camera: Camera
    rasterizer: Rasterizer
    curr_step: int

    # TODO: Solver hyper-parameters

    def train_step(self,
                   position: ti.math.vec3,
                   rotation: ti.math.vec4,
                   buf_size: vec2i,
                   focal_length: ti.math.vec2,
                   image: np.ndarray):
        # TODO: Update camera parameters

        # TODO: Rasterize scene.

        # TODO: Compute loss(L1 loss + D-SSIM loss).

        # TODO: Optimize.

        # TODO: Adaptive density control.

        pass

    def train(self):
        # TODO: Load training data and run training loop.
        pass
