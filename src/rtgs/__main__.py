import logging
import os
import argparse
import pathlib
from typing import Tuple
import taichi as ti
import numpy as np
import quaternion

from rtgs.camera import Camera
from rtgs.ray_tracer import RayTracer
from rtgs.scene import Scene
from rtgs.utils.types import vec2i


# Environment variable log level.
env_level = os.getenv("LOG_LEVEL", "WARNING").upper()
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
    argparser.add_argument(
        "-v",
        "--bvh",
        help="BVH size",
        type=int,
        default=512
    )
    argparser.add_argument(
        "--scale",
        help="Global Gaussian Scale",
        type=float,
        default=1
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
    bvh_size: int = args.bvh
    global_scale: float = args.scale
    scene = Scene(bvh_size, 4, 16)
    scene.load_file(scene_path, global_scale)
    logger.info(f"Scene file loaded from {scene_path}.")

    # Setup camera.
    global_quat = quaternion.from_float_array([1, 0, 0, 0])
    cursor = np.array([0, 0, 0])
    cam_right = np.array([1, 0, 0])
    cam_up = np.array([0, 0, 1])
    theta = 0
    phi = np.pi / 2
    r = 1
    camera = Camera(
        ti.math.vec3(0),
        ti.math.vec4(0, 0, 0, 1),
        vec2i(res),
        ti.math.vec2(focal_length, focal_length)
    )

    def update_cursor(cursor: np.array, u: float, v: float):
        return cursor - u * cam_right - v * cam_up

    def update_camera_pose(theta: float, phi: float, r: float):
        pos = np.array([
            r * np.cos(theta) * np.sin(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(phi)
        ])
        look = -pos / np.linalg.norm(pos)
        cam_right = np.array([-np.sin(theta), np.cos(theta), 0])
        cam_up = np.linalg.cross(cam_right, look)
        rot = np.array([
            cam_right.tolist(),
            cam_up.tolist(),
            (-look).tolist()
        ]).T
        quat = quaternion.from_rotation_matrix(rot)
        global_mat = quaternion.as_rotation_matrix(global_quat)
        quat = global_quat * quat
        pos = global_mat @ (pos + cursor)
        # cam_right = global_mat @ cam_right
        # cam_up = global_mat @ cam_up
        camera.position = ti.math.vec3(pos)
        camera.rotation = ti.math.vec4(quat.x, quat.y, quat.z, quat.w)
        return cam_right, cam_up

    update_camera_pose(theta, phi, r)

    # Setup ray tracer.
    ray_tracer = RayTracer(vec2i(res), scene, camera)

    # Render parameters.
    num_sample: int = args.sample
    num_depth: int = args.depth

    # Display rendered result.
    gui = ti.GUI("Ray Traced Gaussian Splatting", res=res)  # pyright: ignore
    panning = False
    pan_sensitivity = 2
    scroll_sensitivity = 0.1
    moving = False
    move_sensitivity = 2
    mouse_x, mouse_y = 0, 0

    rot_x = gui.slider(
        "Rot X",
        0,
        np.pi *
        2,
        step=np.pi /
        180)  # pyright: ignore
    rot_y = gui.slider(
        "Rot Y",
        0,
        np.pi *
        2,
        step=np.pi /
        180)  # pyright: ignore
    rot_z = gui.slider(
        "Rot Z",
        0,
        np.pi *
        2,
        step=np.pi /
        180)  # pyright: ignore
    old_x = rot_x.value
    old_y = rot_y.value
    old_z = rot_z.value
    enable_panning = True
    panning_button = gui.button("Enable/Disable Panning")

    while gui.running:
        events = gui.get_events(
            ti.GUI.LMB,
            ti.GUI.RMB,
            ti.GUI.WHEEL,
            panning_button)
        for event in events:
            if event.key == panning_button:
                enable_panning = not enable_panning
            if event.key == ti.GUI.LMB:
                if event.type == ti.GUI.PRESS and enable_panning:
                    panning = True
                    mouse_x, mouse_y = gui.get_cursor_pos()
                elif event.type == ti.GUI.RELEASE:
                    panning = False
            if event.key == ti.GUI.RMB:
                if event.type == ti.GUI.PRESS:
                    moving = True
                    mouse_x, mouse_y = gui.get_cursor_pos()
                elif event.type == ti.GUI.RELEASE:
                    moving = False
            if event.key == ti.GUI.WHEEL:
                r += scroll_sensitivity * r * \
                    float(event.delta.y)  # pyright: ignore
                cam_right, cam_up = update_camera_pose(theta, phi, r)
                ray_tracer.clear_sample()
                ray_tracer.num_steps = 0
                ray_tracer.num_samples = 0
        curr_x = rot_x.value
        curr_y = rot_y.value
        curr_z = rot_z.value
        rotating = old_x != curr_x or old_y != curr_y or old_z != curr_z
        old_x, old_y, old_z = curr_x, curr_y, curr_z
        if rotating:
            quat_x = quaternion.from_rotation_vector([rot_x.value, 0, 0])
            quat_y = quaternion.from_rotation_vector([0, rot_y.value, 0])
            quat_z = quaternion.from_rotation_vector([0, 0, rot_z.value])
            global_quat = quat_z * quat_y * quat_x
            cam_right, cam_up = update_camera_pose(theta, phi, r)
            ray_tracer.clear_sample()
            ray_tracer.num_steps = 0
            ray_tracer.num_samples = 0
        if panning or moving:
            nx, ny = gui.get_cursor_pos()
            dx, dy = nx - mouse_x, ny - mouse_y
            if mouse_x != nx or mouse_y != ny:
                if panning:
                    # Update camera parameter.
                    theta -= dx * pan_sensitivity
                    phi += dy * pan_sensitivity
                    phi = max(0, min(phi, np.pi))
                if moving:
                    # Update 3D cursor.
                    cursor = update_cursor(
                        cursor,
                        dx * r * move_sensitivity,
                        dy * r * move_sensitivity
                    )
                cam_right, cam_up = update_camera_pose(theta, phi, r)
                ray_tracer.clear_sample()
                ray_tracer.num_steps = 0
                ray_tracer.num_samples = 0
            mouse_x, mouse_y = nx, ny
        # Take samples.
        if ray_tracer.num_samples < num_sample:
            ray_tracer.sample(num_depth)
            ray_tracer.generate_disp_buffer(
                ray_tracer.num_samples, ray_tracer.num_steps, num_depth)
        gui.set_image(ray_tracer.disp_buf)
        gui.show()


if __name__ == "__main__":
    main()
