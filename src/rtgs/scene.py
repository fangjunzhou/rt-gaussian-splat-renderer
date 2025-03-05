"""
Gaussian splatting scene that implements ray cast Taichi function.
"""

import logging
import taichi as ti
from taichi import StructField
import pandas as pd
import pathlib
from pyntcloud import PyntCloud
from rtgs.gaussian import Gaussian
from rtgs.ray import Ray
from rtgs.utils.math import sigmoid


logger = logging.getLogger(__name__)


@ti.dataclass
class SceneHit:
    # Index of the gaussian in the scene Gaussian field.
    gaussian_idx: int
    # The solution to Gaussian.hit(ray).
    intersections: ti.math.vec2


@ti.data_oriented
class Scene:
    # 1D field of gaussians.
    gaussian_field: StructField
    # TODO: Implement Taichi BVH.

    def __init__(self) -> None:
        self.gaussian_field = Gaussian.field(shape=())

    def load_file(self, path: pathlib.Path):
        """Load ply or splt file as a Gaussian splatting scene.

        :param path: ply or splt file path.
        """
        # Load point cloud.
        point_cloud = PyntCloud.from_file(str(path.resolve()))
        points: pd.DataFrame = point_cloud.points
        num_points = len(points)
        logger.info(f"Point cloud loaded from {path} with {num_points} points.")

        # Extract Gaussian features.
        positions = points[["x", "y", "z"]].to_numpy()
        # FIX: Potential quaternion scalar oder mismatch.
        rotations = points[["rot_0", "rot_1", "rot_2", "rot_3"]].to_numpy()
        scales = points[["scale_0", "scale_1", "scale_2"]].to_numpy()
        colors = points[["f_dc_0", "f_dc_1", "f_dc_2"]].to_numpy()
        opacities = points["opacity"].to_numpy()
        # Convert colors and opacity with sigmoid.
        colors = sigmoid(colors)
        opacities = sigmoid(opacities)

        # Transfer to Taichi.
        pos_field = ti.field(ti.math.vec3, shape=(num_points,))
        rot_field = ti.field(ti.math.vec4, shape=(num_points,))
        sca_field = ti.field(ti.math.vec3, shape=(num_points,))
        col_field = ti.field(ti.math.vec3, shape=(num_points,))
        opa_field = ti.field(ti.f32, shape=(num_points,))
        pos_field.from_numpy(positions)
        rot_field.from_numpy(rotations)
        sca_field.from_numpy(scales)
        col_field.from_numpy(colors)
        opa_field.from_numpy(opacities)

        # Build Gaussian field.
        self.gaussian_field = Gaussian.field(shape=(num_points,))

        @ti.kernel
        def ker_build_gaussian():
            for i in range(num_points):
                position = pos_field[i]
                rotation = rot_field[i]
                scale = sca_field[i]
                color = col_field[i]
                opacity = opa_field[i]
                self.gaussian_field[i].init(
                    position, rotation, scale, color, opacity)

        ker_build_gaussian()
        logger.info(f"Gaussian field loaded successfully.")

    @ti.func
    def hit(self, ray: Ray) -> SceneHit:
        """Ray cast hit the Gaussian scene.

        :param ray: a camera ray.
        :return: Scene hit info.
        """
        # TODO: Implement ray Gaussian intersection.
        return SceneHit(gaussian_idx=-1)
