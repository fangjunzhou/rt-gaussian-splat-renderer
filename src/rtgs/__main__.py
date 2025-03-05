import logging
import os
import argparse
import pathlib
from typing import Tuple
import taichi as ti
import numpy as np

from rtgs.camera import Camera
from rtgs.ray_tracer import RayTracer
from rtgs.scene import Scene
from rtgs.utils.types import vec2i


# Environment variable log level.
env_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
log_level = log_levels.get(env_level, logging.INFO)
logging.basicConfig(level=log_level)

logger = logging.getLogger(__name__)


def main():
    ti.init(arch=ti.gpu)

    argparser = argparse.ArgumentParser(
        "rtgs",
        description="Taichi implementation of 3D Gaussian Ray Tracing."
    )
    argparser.add_argument(
        "-o",
        "--open",
        help="Path to the .splt or .ply Gaussian splatting scene file.",
        type=pathlib.Path
    )
    argparser.add_argument(
        "-r",
        "--res",
        help="Render resolution",
        type=lambda s: tuple(map(int, s.split(','))),
        default=(960, 540)
    )
    argparser.add_argument(
        "-f",
        "--fov",
        help="Vertical FOV in degree.",
        type=float,
        default=90
    )
    argparser.add_argument(
        "-s",
        "--sample",
        help="Render sample rate.",
        type=int,
        default=16
    )
    argparser.add_argument(
        "-d",
        "--depth",
        help="Render sample depth.",
        type=int,
        default=4
    )
    args = argparser.parse_args()

    # Camera parameters.
    res: Tuple[int, int] = args.res
    fov: float = args.fov
    # Calculate focal length.
    half_angle = (fov * np.pi) / 360
    focal_length = (res[1] / 2) / np.tan(half_angle)

    # Load scene file.
    scene_path: pathlib.Path = args.open
    scene = Scene()
    scene.load_file(scene_path)
    logger.info(f"Scene file loaded from {scene_path}.")

    # Setup camera.
    # TODO: Support camera pose.
    camera = Camera(
        ti.math.vec3(0),
        ti.math.vec4(0, 0, 0, 1),
        vec2i(res),
        ti.math.vec2(focal_length, focal_length)
    )

    # Setup ray tracer.
    ray_tracer = RayTracer(vec2i(res), scene, camera)

    # Render parameters.
    num_sample: int = args.sample
    num_depth: int = args.depth

    # Display rendered result.
    gui = ti.GUI("Ray Traced Gaussian Splatting", res=res)  # pyright: ignore
    rendering = True
    while gui.running:
        # Take samples.
        if ray_tracer.num_samples < num_sample:
            ray_tracer.sample(num_depth)
            ray_tracer.generate_disp_buffer(ray_tracer.num_samples)
        elif rendering:
            rendering = False
            logging.info(f"Finish sampling.")
        gui.set_image(ray_tracer.disp_buf)
        gui.show()


if __name__ == "__main__":
    main()
