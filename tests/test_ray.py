import taichi as ti

from rt_gaussian_splat_renderer.ray import Ray


ti.init()


def test_default_ray():
    """Test the default value of a Ray"""
    ray = Ray()
    assert ray.origin == ti.math.vec3(0)
    assert ray.dir == ti.math.vec3(0)
    assert ray.start == 0
    assert ray.end == 0


def test_ray_field():
    ray_field = Ray.field(shape=(960, 540))
    assert ray_field.shape == (960, 540)
    assert ray_field[0, 0].start == 0
    assert ray_field[0, 0].end == 0
    # Set max t of all rays to inf.
    @ti.kernel
    def init_ray_field():
        for i, j in ray_field:
            ray_field[i, j].end = ti.math.inf
    init_ray_field()
    assert ray_field[0, 0].end == ti.math.inf
