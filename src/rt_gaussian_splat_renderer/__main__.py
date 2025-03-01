from camera import Camera
from ray import Ray
import taichi as ti
import math


def main():
    print("Hello World!")

if __name__ == "__main__":
    position = ti.math.vec3(0.0, 2.0, 3.0)
    looking_at = ti.math.vec3(0.0, 0.0, 0.0)
    up = ti.math.vec3(0.0, 1.0, 0.0)
    vert_fov = 90.0
    aspect_ratio = 16.0 / 9.0
    camera = Camera(position, looking_at,up, vert_fov, aspect_ratio)
    main()
