import taichi as ti

from rt_gaussian_splat_renderer.ray import Ray


def test_default_ray():
    """Test the default value of a Ray"""
    ray = Ray()
    assert ray.origin == ti.math.vec3(0)
    assert ray.dir == ti.math.vec3(0)
    assert ray.start == 0
    assert ray.end == 0
