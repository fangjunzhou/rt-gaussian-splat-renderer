import taichi as ti
import copy
from rtgs.bounding_box import Bound
import random
import math

# dataclass


@ti.dataclass
class BVHNode:
    bounds: Bound
    left: ti.i32
    right: ti.i32
    prim_left: ti.i32
    prim_right: ti.i32
    depth: ti.i32

    @ti.func
    def init(
            self,
            bounds=Bound(),
            left=-1,
            right=-1,
            prim_left=-1,
            prim_right=-1,
            depth=0):
        self.bounds = bounds
        self.left = left
        self.right = right
        self.prim_left = prim_left
        self.prim_right = prim_right
        self.depth = depth

    @ti.func
    def hit(self, ray):
        """Ray cast hit the BVH node.

        :param ray: a camera ray.
        :type ray: Ray
        :return: hit information.
        :rtype: ti.math.vec2
        """
        return self.bounds.hit(ray)
