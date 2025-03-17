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


logger = logging.getLogger(__name__)


@ti.dataclass
class SceneHit:
    """Scene hit information.

    :param gaussian_idx: index of the gaussian in the scene Gaussian field.
    :param intersections: the solution to Gaussian.hit(ray) as a vec2 interval.
    """
    gaussian_idx: int
    intersections: ti.math.vec2
    depth: int


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
    balance_weight: int
    leaf_prim: int

    def __init__(
            self,
            max_num_node: int = 128,
            balance_weight: int = 1,
            leaf_prim=8) -> None:
        self.gaussian_field = Gaussian.field(shape=())
        self.max_num_node = max_num_node
        self.bvh_field = BVHNode.field(shape=(max_num_node,))
        self.balance_weight = balance_weight
        self.leaf_prim = leaf_prim

    def load_file(self, path: pathlib.Path, scale: float = 1):
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
        # Convert quaternion order from scalar first to scalar last.
        rotations = points[["rot_1", "rot_2", "rot_3", "rot_0"]].to_numpy()
        scales = points[["scale_0", "scale_1", "scale_2"]].to_numpy()
        colors = points[["f_dc_0", "f_dc_1", "f_dc_2"]].to_numpy()
        spherical_harmonics = points[[f"f_rest_{i}" for i in range(45)]] \
            .to_numpy().reshape((-1, 3, 15))
        opacities = points["opacity"].to_numpy()
        # Convert data with sigmoid.
        rotations = rotations / \
            np.linalg.norm(rotations, axis=-1)[:, np.newaxis]
        scales = np.exp(scales) * scale
        colors = sigmoid(colors)
        opacities = sigmoid(opacities)

        # Transfer to Taichi.
        pos_field = ti.field(ti.math.vec3, shape=(num_points,))
        rot_field = ti.field(ti.math.vec4, shape=(num_points,))
        sca_field = ti.field(ti.math.vec3, shape=(num_points,))
        col_field = ti.field(ti.math.vec3, shape=(num_points,))
        opa_field = ti.field(ti.f32, shape=(num_points,))
        sh_field = ti.field(ti.math.vec3, shape=(num_points, 15))
        pos_field.from_numpy(positions)
        rot_field.from_numpy(rotations)
        sca_field.from_numpy(scales)
        col_field.from_numpy(colors)
        sh_field.from_numpy(spherical_harmonics)
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
                    position, rotation, scale, color, opacity)
                self.gaussian_field[i].sh_10 = sh_field[i, 0]
                self.gaussian_field[i].sh_11 = sh_field[i, 1]
                self.gaussian_field[i].sh_12 = sh_field[i, 2]
                self.gaussian_field[i].sh_20 = sh_field[i, 3]
                self.gaussian_field[i].sh_21 = sh_field[i, 4]
                self.gaussian_field[i].sh_22 = sh_field[i, 5]
                self.gaussian_field[i].sh_23 = sh_field[i, 6]
                self.gaussian_field[i].sh_24 = sh_field[i, 7]
                self.gaussian_field[i].sh_30 = sh_field[i, 8]
                self.gaussian_field[i].sh_31 = sh_field[i, 9]
                self.gaussian_field[i].sh_32 = sh_field[i, 10]
                self.gaussian_field[i].sh_33 = sh_field[i, 11]
                self.gaussian_field[i].sh_34 = sh_field[i, 12]
                self.gaussian_field[i].sh_35 = sh_field[i, 13]
                self.gaussian_field[i].sh_36 = sh_field[i, 14]

        build_gaussian()
        logger.info(f"Gaussian field loaded successfully.")

        NUM_THRESHOLD = 32

        bbox_field = Bound.field(shape=(3, NUM_THRESHOLD, num_points))
        bbox_buf = Bound.field(shape=(3, NUM_THRESHOLD, num_points))
        left_gaussian_idx = ti.field(
            ti.i32, shape=(
                3, NUM_THRESHOLD, num_points))
        right_gaussian_idx = ti.field(
            ti.i32, shape=(
                3, NUM_THRESHOLD, num_points))
        gaussian_buf = Gaussian.field(shape=(num_points,))

        thresholds = ti.field(ti.f32, shape=(3, NUM_THRESHOLD))
        num_lefts = ti.field(ti.i32, shape=(3, NUM_THRESHOLD))
        num_rights = ti.field(ti.i32, shape=(3, NUM_THRESHOLD))
        area_lefts = ti.field(ti.f32, shape=(3, NUM_THRESHOLD))
        area_rights = ti.field(ti.f32, shape=(3, NUM_THRESHOLD))

        @ti.kernel
        def load_bbox_gaussian(start: int, end: int):
            """Load Gaussian bounding box from gaussian_field to bbox_field.

            :param start: start index of the gaussian.
            :param end: end index of the gaussian, not included.
            """
            size = end - start
            for i in range(size):
                bbox_field[0, 0, i] = self.gaussian_field[start + i] \
                    .bounding_box()
            for i in range(size, num_points):
                bbox_field[0, 0, i] = self.gaussian_field[start + size - 1] \
                    .bounding_box()

        @ti.kernel
        def load_left(max_size: int):
            for i, j, k in ti.ndrange(3, NUM_THRESHOLD, max_size):
                if k < num_lefts[i, j]:
                    bbox_field[i,
                               j,
                               k] = self.gaussian_field[left_gaussian_idx[i,
                                                                          j,
                                                                          k]].bounding_box()
                else:
                    bbox_field[i,
                               j,
                               k] = self.gaussian_field[left_gaussian_idx[i,
                                                                          j,
                                                                          num_lefts[i,
                                                                                    j] - 1]].bounding_box()

        @ti.kernel
        def load_right(max_size: int):
            for i, j, k in ti.ndrange(3, NUM_THRESHOLD, max_size):
                if k < num_rights[i, j]:
                    bbox_field[i,
                               j,
                               k] = self.gaussian_field[right_gaussian_idx[i,
                                                                           j,
                                                                           k]].bounding_box()
                else:
                    bbox_field[i,
                               j,
                               k] = self.gaussian_field[right_gaussian_idx[i,
                                                                           j,
                                                                           num_rights[i,
                                                                                      j] - 1]].bounding_box()

        @ti.kernel
        def reduction(max_size: int):
            for i, j, k in ti.ndrange(
                3, NUM_THRESHOLD, int(
                    (max_size + 1) / 2)):
                bbox_buf[i, j, k] = bbox_field[i, j, k *
                                               2].union(bbox_field[i, j, k * 2 + 1])
            for i, j in ti.ndrange(3, NUM_THRESHOLD):
                bbox_buf[i, j, int((max_size + 1) / 2)
                         ] = bbox_field[i, j, max_size - 1]
            for i, j, k in ti.ndrange(
                3, NUM_THRESHOLD, int(
                    (max_size + 1) / 2) + 1):
                bbox_field[i, j, k] = bbox_buf[i, j, k]

        @ti.kernel
        def area_left():
            for i, j in ti.ndrange(3, NUM_THRESHOLD):
                area_lefts[i, j] = bbox_field[i, j, 0].area()

        @ti.kernel
        def area_right():
            for i, j in ti.ndrange(3, NUM_THRESHOLD):
                area_rights[i, j] = bbox_field[i, j, 0].area()

        @ti.kernel
        def split(start: int, end: int):
            # Clear pointer.
            for i, j in ti.ndrange(3, NUM_THRESHOLD):
                num_lefts[i, j] = 0
                num_rights[i, j] = 0

            # Split gaussian.
            for i, j, k in ti.ndrange(3, NUM_THRESHOLD, (start, end)):
                gaussian = self.gaussian_field[k]
                center = gaussian.position[i]

                if center <= thresholds[i, j]:
                    idx = ti.atomic_add(num_lefts[i, j], 1)
                    left_gaussian_idx[i, j, idx] = k
                else:
                    idx = ti.atomic_add(num_rights[i, j], 1)
                    right_gaussian_idx[i, j, idx] = k

        @ti.kernel
        def reorder(start: int, end: int, axis: int, threshold: int):
            num_left = num_lefts[axis, threshold]
            num_right = num_rights[axis, threshold]
            for i in range(num_left):
                gaussian_buf[i] = self.gaussian_field[left_gaussian_idx[axis, threshold, i]]
            for i in range(num_right):
                gaussian_buf[num_left +
                             i] = self.gaussian_field[right_gaussian_idx[axis, threshold, i]]
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

            node.bound = bbox_field[0, 0, 0]
            logger.info(f"Current bounding box {bbox_field[0, 0, 0]}.")

            # Split node.
            if num_primitives < self.leaf_prim or top >= self.max_num_node:
                return True, top

            # Find optimal split.
            # Load thresholds.
            p_min = node.bound.p_min
            p_max = node.bound.p_max
            thresholds.from_numpy(
                np.linspace(
                    [p_min[0], p_min[1], p_min[2]],
                    [p_max[0], p_max[1], p_max[2]],
                    NUM_THRESHOLD + 2,
                    axis=-1
                ).astype(np.float32)[:, 1:-1]
            )
            split(node.prim_left, node.prim_right)
            # Calculate left bounding boxes.
            bbox_size = int(np.max(num_lefts.to_numpy()))
            load_left(bbox_size)
            while bbox_size > 1:
                reduction(bbox_size)
                bbox_size = int((bbox_size + 1) / 2)
            area_left()
            left_area = area_lefts.to_numpy()
            # Calculate left bounding boxes.
            bbox_size = int(np.max(num_rights.to_numpy()))
            load_right(bbox_size)
            while bbox_size > 1:
                reduction(bbox_size)
                bbox_size = int((bbox_size + 1) / 2)
            area_right()
            right_area = area_rights.to_numpy()

            parent_area = node.bound.area_py()

            left_prob = (left_area / parent_area).astype(np.float64)
            right_prob = (right_area / parent_area).astype(np.float64)

            costs = left_prob * num_lefts.to_numpy().astype(np.float64) ** self.balance_weight + \
                right_prob * num_rights.to_numpy().astype(np.float64) ** self.balance_weight

            axis, threshold = np.unravel_index(np.argmin(costs), costs.shape)
            axis, threshold = int(axis), int(threshold)

            logger.info(
                f"Found optimal split with axis {axis}, and threshold {threshold}.")

            # Empty child.
            if num_lefts[axis,
                         threshold] == 0 or num_rights[axis,
                                                       threshold] == 0:
                return True, top

            # Reorder gaussian.
            reorder(node.prim_left, node.prim_right, axis, threshold)
            mid = node.prim_left + num_lefts[axis, threshold]
            left_idx = top
            right_idx = top + 1

            logger.info(f"Split to {num_lefts[axis,
                                              threshold]} left children and {num_rights[axis,
                                                                                        threshold]} right children.")

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
        max_node_size = 0
        for node_id in tqdm(range(self.max_num_node)):
            has_split, top = split_node(node_id, top)
            if self.bvh_field[node_id].left == - \
                    1 and self.bvh_field[node_id].right == -1:
                node_size = self.bvh_field[node_id].prim_right - \
                    self.bvh_field[node_id].prim_left
                max_node_size = max(max_node_size, node_size)
            if not has_split:
                break
        print(
            f"Build {top} BVH nodes in total. Max leaf node size is {max_node_size}.")

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
                        if intersections.x < ray.end and intersections.x > ray.start:
                            if intersections.x < hit_t:
                                hit = SceneHit(
                                    gaussian_idx=i, intersections=intersections, depth=node.depth)
                                hit_t = intersections.x
                # Hit intermediate node.
                else:
                    # Push close child last.
                    left_hit = self.bvh_field[node.left].hit(ray)
                    right_hit = self.bvh_field[node.right].hit(ray)
                    if left_hit.x < right_hit.x:
                        stack.push(node.right)
                        stack.push(node.left)
                    else:
                        stack.push(node.left)
                        stack.push(node.right)

        return hit
