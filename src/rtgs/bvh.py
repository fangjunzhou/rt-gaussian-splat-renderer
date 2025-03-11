import taichi as ti
import copy
from rtgs.bounding_box import Bounds
import random
import math

# dataclass


@ti.dataclass
class BVHNode:
    bounds: Bounds
    left: ti.i32
    right: ti.i32
    primitive_left: ti.i32
    primitive_right: ti.i32
    depth: ti.i32

    @ti.func
    def init(
            self,
            bounds=Bounds(),
            left=-1,
            right=-1,
            primitive_left=-1,
            primitive_right=-1,
            depth=0):
        self.bounds = bounds
        self.left = left
        self.right = right
        self.primitive_left = primitive_left
        self.primitive_right = primitive_right
        self.depth = depth

    @ti.func
    def hit(self, ray):
        """Ray cast hit the BVH node.

        :param ray: a camera ray.
        :type ray: Ray
        :return: hit information.
        :rtype: bool
        """
        return self.bounds.hit(ray)
