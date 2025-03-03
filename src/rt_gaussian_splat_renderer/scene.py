import taichi as ti
from taichi import StructField
import pathlib
from rt_gaussian_splat_renderer.gaussian import Gaussian
from rt_gaussian_splat_renderer.ray import Ray


@ti.data_oriented
class Scene:
    # 1D field of gaussians.
    gaussian_field: StructField | None
    # TODO: Implement Taichi BVH.

    def __init__(self) -> None:
        pass

    def load_file(self, path: pathlib.Path):
        """Load ply or splt file as a Gaussian splatting scene.

        :param path: ply or splt file path.
        """
        pass

    @ti.func
    def hit(self, ray: Ray) -> Gaussian:
        """Ray cast hit the Gaussian scene.

        :param ray:
        :return:
        """
        return Gaussian()
