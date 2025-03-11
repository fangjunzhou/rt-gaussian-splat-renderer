"""
Gaussian splatting scene that implements ray cast Taichi function.
"""

import logging
from typing import Tuple
import taichi as ti
from taichi import StructField
import pandas as pd
import pathlib
from tqdm import tqdm
import numpy as np
from pyntcloud import PyntCloud
from rtgs.gaussian import Gaussian
from rtgs.ray import Ray
from rtgs.utils.math import sigmoid
from rtgs.bvh import BVHNode
from rtgs.bounding_box import Bound

ti.init(arch=ti.gpu)


logger = logging.getLogger(__name__)


@ti.dataclass
class SceneHit:
    """Scene hit information.

    :param gaussian_idx: index of the gaussian in the scene Gaussian field.
    :param intersections: the solution to Gaussian.hit(ray) as a vec2 interval.
    """
    gaussian_idx: int
    intersections: ti.math.vec2


vec_stack = ti.types.vector(32, ti.i32)


@ti.dataclass
class Stack:
    stack: vec_stack
    top: int

    def __init__(self) -> None:
        self.stack = vec_stack(0)
        self.top = 0

    @ti.func
    def size(self):
        return self.top

    @ti.func
    def push(self, idx):
        self.stack[self.top] = idx
        self.top += 1

    @ti.func
    def pop(self):
        val = self.stack[self.top - 1]
        self.top -= 1
        return val


@ti.data_oriented
class Scene:
    """A Gaussian splatting scene representation.

    :param gaussian_field: a list of all the Gaussians in the scene.
    """
    # 1D field of gaussians.
    gaussian_field: StructField
    # Implement Taichi BVH.
    bvh_field: StructField
    max_num_node: int

    def __init__(self, max_num_node: int = 128) -> None:
        self.gaussian_field = Gaussian.field(shape=())
        self.max_num_node = max_num_node
        self.bvh_field = BVHNode.field(shape=(max_num_node,))

    def load_file(self, path: pathlib.Path):
        """Load ply or splt file as a Gaussian splatting scene.

        :param path: ply or splt file path.
        """
        # Load point cloud.
        point_cloud = PyntCloud.from_file(str(path.resolve()))
        points: pd.DataFrame = point_cloud.points
        num_points = len(points)
        logger.info(f"Point cloud loaded from {path} with {num_points} points.")

        # Extract Gaussian features.
        positions = points[["x", "y", "z"]].to_numpy()
        # FIX: Potential quaternion scalar oder mismatch.
        rotations = points[["rot_0", "rot_1", "rot_2", "rot_3"]].to_numpy()
        scales = points[["scale_0", "scale_1", "scale_2"]].to_numpy()
        colors = points[["f_dc_0", "f_dc_1", "f_dc_2"]].to_numpy()
        opacities = points["opacity"].to_numpy()
        # Convert data with sigmoid.
        scales = sigmoid(scales)
        colors = sigmoid(colors)
        opacities = sigmoid(opacities)

        # Transfer to Taichi.
        pos_field = ti.field(ti.math.vec3, shape=(num_points,))
        rot_field = ti.field(ti.math.vec4, shape=(num_points,))
        sca_field = ti.field(ti.math.vec3, shape=(num_points,))
        col_field = ti.field(ti.math.vec3, shape=(num_points,))
        opa_field = ti.field(ti.f32, shape=(num_points,))
        pos_field.from_numpy(positions)
        rot_field.from_numpy(rotations)
        sca_field.from_numpy(scales)
        col_field.from_numpy(colors)
        opa_field.from_numpy(opacities)

        # Build Gaussian field.
        self.gaussian_field = Gaussian.field(shape=(num_points,))

        @ti.kernel
        def build_gaussian():
            for i in range(num_points):
                position = pos_field[i]
                rotation = rot_field[i]
                scale = sca_field[i]
                color = col_field[i]
                opacity = opa_field[i]
                self.gaussian_field[i].init(
                    position, rotation, scale, color, opacity, i)

        build_gaussian()
        logger.info(f"Gaussian field loaded successfully.")

        bbox_field = Bound.field(shape=(num_points,))
        bbox_buf = Bound.field(shape=(num_points,))
        left_gaussian_idx = ti.field(ti.i32, shape=(num_points,))
        right_gaussian_idx = ti.field(ti.i32, shape=(num_points,))
        gaussian_buf = Gaussian.field(shape=(num_points,))

        @ti.kernel
        def load_bbox_gaussian(start: int, end: int):
            for i in range(end - start):
                bbox_field[i] = self.gaussian_field[start + i].bounding_box()

        @ti.kernel
        def load_left():
            for i in left_gaussian_idx:
                bbox_field[i] = self.gaussian_field[left_gaussian_idx[i]
                                                    ].bounding_box()

        @ti.kernel
        def load_right():
            for i in right_gaussian_idx:
                bbox_field[i] = self.gaussian_field[right_gaussian_idx[i]
                                                    ].bounding_box()

        @ti.kernel
        def reduction(size: int):
            for i in range(int(size / 2)):
                bbox_buf[i] = bbox_field[i * 2].union(bbox_field[i * 2 + 1])
            if size % 2 == 1:
                bbox_buf[int(size / 2)] = bbox_field[size - 1]
            for i in range(int((size + 1) / 2)):
                bbox_field[i] = bbox_buf[i]

        @ti.kernel
        def split(start: int, end: int, axis: int,
                  threshold: float) -> Tuple[int, int]:
            num_left = 0
            num_right = 0

            for i in range(start, end):
                gaussian = self.gaussian_field[i]
                center = gaussian.position[axis]

                if center <= threshold:
                    idx = ti.atomic_add(num_left, 1)
                    left_gaussian_idx[idx] = i
                else:
                    idx = ti.atomic_add(num_right, 1)
                    right_gaussian_idx[idx] = i

            return num_left, num_right

        @ti.kernel
        def reorder(start: int, end: int, num_left: int, num_right: int):
            for i in range(num_left):
                gaussian_buf[i] = self.gaussian_field[left_gaussian_idx[i]]
            for i in range(num_right):
                gaussian_buf[num_left +
                             i] = self.gaussian_field[right_gaussian_idx[i]]
            for i in range(num_left + num_right):
                self.gaussian_field[start + i] = gaussian_buf[i]

        def split_node(node_idx: int, top: int) -> Tuple[bool, int]:
            logger.info(f"Splitting node {node_idx}.")
            node = self.bvh_field[node_id]
            if node.depth == -1:
                return False, top

            num_primitives = node.prim_right - node.prim_left

            # Construct bunding box for current node.
            load_bbox_gaussian(node.prim_left, node.prim_right)
            bbox_size = num_primitives
            # Union bounding box.
            while bbox_size > 1:
                reduction(bbox_size)
                bbox_size = int((bbox_size + 1) / 2)

            node.bound = bbox_field[0]
            logger.info(f"Current bounding box {bbox_field[0]}.")

            # Split node.
            if num_primitives < 2 or top >= self.max_num_node:
                return True, top

            best_axis = -1
            best_threshold = 0.0
            best_cost = float("inf")

            # Find optimal split.
            for axis in range(3):
                axis_min = node.bound.p_min[axis]
                axis_max = node.bound.p_max[axis]
                for threshold in np.linspace(axis_min, axis_max, 8)[1:-1]:
                    logger.info(f"Split at axis {axis}, threshold {threshold}.")
                    num_left, num_right = split(
                        node.prim_left, node.prim_right, axis, threshold)
                    logger.info(
                        f"Split {num_left} left children and {num_right} right children.")
                    load_left()
                    # Union bounding box.
                    bbox_size = num_left
                    while bbox_size > 1:
                        reduction(bbox_size)
                        bbox_size = int((bbox_size + 1) / 2)
                    left_bbox = bbox_field[0]
                    left_area = left_bbox.area_py()
                    logger.info(f"Left bbox: {left_bbox}")
                    load_right()
                    # Union bounding box.
                    bbox_size = num_right
                    while bbox_size > 1:
                        reduction(bbox_size)
                        bbox_size = int((bbox_size + 1) / 2)
                    right_bbox = bbox_field[0]
                    logger.info(f"Right bbox: {right_bbox}")
                    right_area = right_bbox.area_py()
                    parent_area = node.bound.area_py()
                    logger.info(
                        f"Left area: {left_area}. Right area: {right_area}.")

                    left_prob = left_area / parent_area if parent_area > 0 else 0.0
                    right_prob = right_area / parent_area if parent_area > 0 else 0.0

                    cost = left_prob * num_left + right_prob * num_right
                    logger.info(f"Cost {cost}.")

                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_threshold = threshold

            logger.info(
                f"Found optimal split with axis {best_axis}, and threshold {best_threshold}.")

            # Split node.
            num_left, num_right = split(
                node.prim_left, node.prim_right, best_axis, best_threshold)
            if num_left == 0 or num_right == 0:
                return True, top

            logger.info(
                f"Split to {num_left} left children and {num_right} right children.")

            # Reorder gaussian.
            reorder(node.prim_left, node.prim_right, num_left, num_right)
            mid = node.prim_left + num_left
            left_idx = top
            right_idx = top + 1

            self.bvh_field[left_idx].prim_left = node.prim_left
            self.bvh_field[left_idx].prim_right = mid
            self.bvh_field[left_idx].depth = node.depth + 1
            self.bvh_field[right_idx].prim_left = mid
            self.bvh_field[right_idx].prim_right = node.prim_right
            self.bvh_field[right_idx].depth = node.depth + 1

            self.bvh_field[node_idx].left = left_idx
            self.bvh_field[node_idx].right = right_idx

            return True, top + 2

        @ti.kernel
        def init_bvh():
            for i in self.bvh_field:
                self.bvh_field[i].init()
            self.bvh_field[0].init(
                bound=Bound(),
                prim_left=0,
                prim_right=self.gaussian_field.shape[0],
                depth=0
            )

        init_bvh()
        top = 1
        for node_id in tqdm(range(self.max_num_node)):
            has_split, top = split_node(node_id, top)
            if not has_split:
                break
        logging.info(f"Build {top} BVH nodes in total.")

    @ti.func
    def hit(self, ray):
        """Ray cast hit the Gaussian scene.

        :param ray: a camera ray.
        :type ray: Ray
        :return: Scene hit info.
        :rtype: SceneHit
        """
        hit = SceneHit(gaussian_idx=-1)
        hit_t = ti.math.inf

        stack = Stack()
        stack.push(0)

        while stack.size() != 0:
            node = self.bvh_field[stack.pop()]
            hit_inter = node.hit(ray)
            # Prune far child.
            if hit_inter.x > hit_t:
                continue
            if hit_inter.x < hit_inter.y:
                # Base case: leaf node.
                if node.left == -1 and node.right == -1:
                    for i in range(node.prim_left, node.prim_right):
                        gaussian = self.gaussian_field[i]
                        intersections = gaussian.hit(ray)
                        if intersections.x < ray.end and intersections.y > ray.start:
                            if intersections.x < hit_t:
                                hit = SceneHit(
                                    gaussian_idx=i, intersections=intersections)
                                hit_t = intersections.x
                # Hit intermediate node.
                else:
                    # TODO: Push close child first.
                    stack.push(node.left)
                    stack.push(node.right)

        return hit
