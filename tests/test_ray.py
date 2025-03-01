import taichi as ti
import logging
import pytest

from rt_gaussian_splat_renderer.ray import Ray, new_ray


logger = logging.getLogger(__name__)

# Init taichi runtime.
ti.init(arch=ti.gpu, random_seed=42)
logger.info(f"Current Taichi backend: {ti.cfg.arch}")  # pyright: ignore


def test_new_ray():
    """Test python scope ray constructor."""
    # Default ray.
    ray = new_ray()
    assert ray.origin == ti.math.vec3(0)
    assert ray.direction == ti.math.vec3(0, 1, 0)
    assert ray.start == 0
    assert ray.end == ti.math.inf
    # Custom ray.
    ray = new_ray(ti.math.vec3(1, 2, 3), ti.math.vec3(4, 5, 6), 37, 42)
    assert ray.origin == ti.math.vec3(1, 2, 3)
    assert ray.direction == ti.math.vec3(4, 5, 6)
    assert ray.start == 37
    assert ray.end == 42


def test_ray_field():
    """Test the Ray field generation."""
    SIZE = (32, 32)
    # Taichi scope initialization.
    ray_field = Ray.field(shape=SIZE)
    assert ray_field.shape == SIZE
    assert ray_field[0, 0].origin == ti.math.vec3(0)
    assert ray_field[0, 0].direction == ti.math.vec3(0)
    assert ray_field[0, 0].start == 0
    assert ray_field[0, 0].end == 0

    @ti.kernel
    def init_ray_field():
        for I in ti.grouped(ray_field):
            ray_field[I].init()
    init_ray_field()
    assert ray_field[0, 0].origin == ti.math.vec3(0)
    assert ray_field[0, 0].direction == ti.math.vec3(0, 1, 0)
    assert ray_field[0, 0].start == 0
    assert ray_field[0, 0].end == ti.math.inf


def test_ray_eval():
    """Test Ray.get method."""
    SIZE = (4, 4)
    ray_field = Ray.field(shape=SIZE)

    @ti.func
    def rand_vec3():
        x = ti.random(dtype=ti.f32)
        y = ti.random(dtype=ti.f32)
        z = ti.random(dtype=ti.f32)
        v = ti.math.vec3(x, y, z)
        v = ti.math.normalize(v)
        return v

    @ti.kernel
    def init_ray_field():
        for I in ti.grouped(ray_field):
            ray_field[I].origin = rand_vec3()
            ray_field[I].direction = rand_vec3()

    init_ray_field()

    # Time scalar field.
    t_field = ti.field(ti.f32, shape=SIZE)

    @ti.kernel
    def init_t_field():
        for I in ti.grouped(t_field):
            t_field[I] = ti.random(ti.f32)

    init_t_field()

    # Reference ray pos.
    p_field_ref = ti.field(ti.math.vec3, shape=SIZE)
    for i in range(p_field_ref.shape[0]):
        for j in range(p_field_ref.shape[1]):
            p_field_ref[i, j] = ray_field[i, j].origin + \
                t_field[i, j] * ray_field[i, j].direction

    logger.info(
        f"Ray {ray_field[0, 0]} at time {t_field[0, 0]}: {p_field_ref[0, 0]}")

    # Time scalar field.
    p_field = ti.field(ti.math.vec3, shape=SIZE)

    @ti.kernel
    def eval_ray():
        for I in ti.grouped(ray_field):
            p_field[I] = ray_field[I].get(t_field[I])

    eval_ray()

    e_field = ti.field(ti.f32, shape=SIZE)

    @ti.kernel
    def eval_err():
        for I in ti.grouped(e_field):
            e_field[I] = ti.math.length(p_field_ref[I] - p_field[I])

    eval_err()

    for i in range(e_field.shape[0]):
        for j in range(e_field.shape[1]):
            assert e_field[i, j] == pytest.approx(0)
