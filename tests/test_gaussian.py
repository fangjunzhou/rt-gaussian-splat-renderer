import taichi as ti
import logging
import numpy as np
import rtgs.utils.quaternion as quat

from rtgs.gaussian import Gaussian, new_gaussian
from rtgs.ray import Ray, new_ray

logger = logging.getLogger(__name__)

# Init taichi runtime.
ti.init(arch=ti.gpu, random_seed=42)
logger.info(f"Current Taichi backend: {ti.cfg.arch}")  # pyright: ignore


def test_new_gaussian():
    """Test python scope Gaussian constructor."""
    # Default gaussian.
    gaussian = new_gaussian()
    assert gaussian.position == ti.math.vec3(0)
    assert gaussian.rotation == ti.math.vec4(0, 0, 0, 1)
    assert gaussian.scale == ti.math.vec3(1, 1, 1)
    assert gaussian.color == ti.math.vec3(1, 0, 1)
    assert gaussian.opacity == 1
    # Custom gaussian.
    prosition = ti.math.vec3(1, 2, 3)
    q = quat.from_euler_angles([np.pi / 4, 0, 0])
    # NOTE: Convert scalar-first to scalar-last representation.
    q.z, q.w = q.w, q.z
    logger.info(f"Rotation quaternion: {q}")
    rotation = ti.math.vec4(quat.as_float_array(q))
    scale = ti.math.vec3(2, 3, 4)
    color = ti.math.vec3(1, 0, 0)
    gaussian = new_gaussian(prosition, rotation, scale, color, 0.75)
    assert gaussian.position == ti.math.vec3(1, 2, 3)
    assert gaussian.rotation == ti.math.vec4(quat.as_float_array(q))
    assert gaussian.scale == ti.math.vec3(2, 3, 4)
    assert gaussian.color == ti.math.vec3(1, 0, 0)
    assert gaussian.opacity == 0.75


def test_gaussian_field():
    """Test the Gaussian field generation."""
    SIZE = (32,)
    # Taichi scope initialization.
    gaussian_field = Gaussian.field(shape=SIZE)
    assert gaussian_field.shape == SIZE
    assert gaussian_field[0].position == ti.math.vec3(0)
    assert gaussian_field[0].rotation == ti.math.vec4(0)
    assert gaussian_field[0].scale == ti.math.vec3(0)
    assert gaussian_field[0].color == ti.math.vec3(0)
    assert gaussian_field[0].opacity == 0

    @ti.kernel
    def init_gaussian_field():
        for I in ti.grouped(gaussian_field):
            gaussian_field[I].init()
    init_gaussian_field()
    assert gaussian_field[0].position == ti.math.vec3(0)
    assert gaussian_field[0].rotation == ti.math.vec4(0, 0, 0, 1)
    assert gaussian_field[0].scale == ti.math.vec3(1, 1, 1)
    assert gaussian_field[0].color == ti.math.vec3(1, 0, 1)
    assert gaussian_field[0].opacity == 1

def test_gaussian_hit():
    '''Test Gaussian hit method.'''
    SIZE = (4,)
    gaussian_field = Gaussian.field(shape=SIZE)
    gaussian = new_gaussian()
    gaussian_field[0] = gaussian
    ray = new_ray()
    hit = gaussian.hit(ray)
    assert hit == ti.math.vec2(0, ti.math.inf)
    ray = new_ray(ti.math.vec3(0, 0, 0), ti.math.vec3(0, 1, 0), 0, 1)
    hit = gaussian.hit(ray)
    assert hit == ti.math.vec2(0, ti.math.inf)
    ray = new_ray(ti.math.vec3(0, 0, 0), ti.math.vec3(0, 1, 0), 0, 1)
    gaussian.position = ti.math.vec3(0, 0, 1)
    hit = gaussian.hit(ray)
    assert hit == ti.math.vec2(1, ti.math.inf)
