"""
Gaussian splatting scene that implements ray cast Taichi function.
"""

import logging
import taichi as ti
from taichi import StructField
import pandas as pd
import pathlib
from pyntcloud import PyntCloud
from rtgs.gaussian import Gaussian
from rtgs.ray import Ray
from rtgs.utils.math import sigmoid
from rtgs.bvh import BVHNode
from rtgs.bounding_box import Bounds

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


@ti.data_oriented
class Scene:
    """A Gaussian splatting scene representation.

    :param gaussian_field: a list of all the Gaussians in the scene.
    """
    # 1D field of gaussians.
    gaussian_field: StructField
    # TODO: Implement Taichi BVH.
    bvh_field: StructField

    def __init__(self) -> None:
        self.gaussian_field = Gaussian.field(shape=())
        self.bvh_field = BVHNode.field(shape=(1024,))
        self.stack = ti.field(ti.i32, shape=1024)
        self.max_node_num = 1024

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
        # Convert colors and opacity with sigmoid.
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
                    position, rotation, scale, color, opacity,i)

        build_gaussian()
        logger.info(f"Gaussian field loaded successfully.")
        # for i in range(num_points):
        # @ti.kernel
        # def get_cost(node_id: ti.i32, split_axis: ti.i32, threshold: ti.f32):
        #     """Get the cost of splitting a node."""
        #     node = self.bvh_field[node_id]
    
        #     # Counters for left and right children
        #     left_count = 0
        #     right_count = 0
            
        #     # Bounding boxes for left and right subsets
        #     left_min = ti.Vector([ti.inf, ti.inf, ti.inf])
        #     left_max = ti.Vector([-1e10, -1e10, -1e10])
        #     right_min = ti.Vector([ti.inf, ti.inf, ti.inf])
        #     right_max = ti.Vector([-1e10, -1e10, -1e10])
            
        #     # Iterate through the Gaussians in this node
        #     for i in range(node.primitive_left, node.primitive_right):
        #         gaussian = self.gaussian_field[i]
        #         gauss_min, gauss_max = gaussian.bounding_box()

        #         center = (gauss_min[split_axis] + gauss_max[split_axis]) * 0.5
        #         if center <= threshold:
        #             left_min = ti.min(left_min, gauss_min)
        #             left_max = ti.max(left_max, gauss_max)
        #             left_count += 1
        #         else:
        #             right_min = ti.min(right_min, gauss_min)
        #             right_max = ti.max(right_max, gauss_max)
        #             right_count += 1

        #     def surface_area(min_bound, max_bound):
        #         extent = max_bound - min_bound
        #         return 2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0])

        #     left_area = surface_area(left_min, left_max) if left_count > 0 else 0.0
        #     right_area = surface_area(right_min, right_max) if right_count > 0 else 0.0
        #     parent_area = surface_area(node.min_bound, node.max_bound)
            
        #     left_prob = left_area / parent_area if parent_area > 0 else 0.0
        #     right_prob = right_area / parent_area if parent_area > 0 else 0.0
            
        #     cost = left_prob * left_count + right_prob * right_count
        #     return cost
        
        @ti.func
        def surface_area(min_bound, max_bound):
            extent = max_bound - min_bound
            return 2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0])

        
        @ti.kernel
        def split_bvh(node_id: ti.i32):
            """Splits a BVH node once using SAH."""
            node = self.bvh_field[node_id]

            num_primitives = node.primitive_right - node.primitive_left
            should_split = num_primitives > 2  

            # Compute the bounding box of the node
            bounds = Bounds()
            for i in range(node.primitive_left, node.primitive_right):
                gaussian = self.gaussian_field[i]
                p_min, p_max = gaussian.bounding_box()
                bounds = bounds.bounds_union(Bounds(p_min, p_max))

            node.bounds = bounds
            best_axis = -1
            best_threshold = 0.0
            best_cost = float('inf')

            if should_split:
                for axis in range(3):
                    left_count = 0
                    right_count = 0
                    left_min = ti.Vector([1e10, 1e10, 1e10])
                    left_max = ti.Vector([-1e10, -1e10, -1e10])
                    right_min = ti.Vector([1e10, 1e10, 1e10])
                    right_max = ti.Vector([-1e10, -1e10, -1e10])

                    for i in range(node.primitive_left, node.primitive_right):
                        gaussian = self.gaussian_field[i]
                        gauss_min, gauss_max = gaussian.bounding_box()
                        center = (gauss_min[axis] + gauss_max[axis]) * 0.5

                        if center <= node.bounds.get_centroid()[axis]:
                            left_min = ti.min(left_min, gauss_min)
                            left_max = ti.max(left_max, gauss_max)
                            left_count += 1
                        else:
                            right_min = ti.min(right_min, gauss_min)
                            right_max = ti.max(right_max, gauss_max)
                            right_count += 1



                    left_area = surface_area(left_min, left_max) if left_count > 0 else 0.0
                    right_area = surface_area(right_min, right_max) if right_count > 0 else 0.0
                    parent_area = surface_area(node.bounds.p_min, node.bounds.p_max)

                    left_prob = left_area / parent_area if parent_area > 0 else 0.0
                    right_prob = right_area / parent_area if parent_area > 0 else 0.0

                    cost = left_prob * left_count + right_prob * right_count

                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_threshold = node.bounds.get_centroid()[axis]

                if best_axis == -1:
                    should_split = False
                    
                left_count = 0
                if should_split:
                    for i in range(node.primitive_left, node.primitive_right):
                        gaussian = self.gaussian_field[i]
                        center = (gaussian.bounding_box()[0][best_axis] + gaussian.bounding_box()[1][best_axis]) * 0.5
                        if center <= best_threshold:
                            left_count += 1

                    if left_count == 0 or left_count == num_primitives:
                        should_split = False

                if should_split:
                    mid = node.primitive_left + left_count - 1
                    left_child_id = node_id * 2 + 1
                    right_child_id = node_id * 2 + 2

                    self.bvh_field[left_child_id].init(bounds=Bounds(), primitive_left=node.primitive_left, primitive_right=mid, depth=node.depth + 1)
                    self.bvh_field[right_child_id].init(bounds=Bounds(), primitive_left=mid + 1, primitive_right=node.primitive_right, depth=node.depth + 1)

                    self.bvh_field[node_id].left = left_child_id
                    self.bvh_field[node_id].right = right_child_id

        @ti.kernel
        def build_bvh():
            self.bvh_field[0].init(bounds=Bounds(), primitive_left=0, primitive_right=self.gaussian_field.shape[0], depth=0)

        build_bvh()
        for node_id in range(self.max_node_num):  
            if node_id >= self.bvh_field.shape[0]:  
                continue
            split_bvh(node_id) 
    def bounding_box(self, i):
        return self.bvh_min(i), self.bvh_max(i)

    

    @ti.func
    def hit(self, ray):
        """Ray cast hit the Gaussian scene.

        :param ray: a camera ray.
        :type ray: Ray
        :return: Scene hit info.
        :rtype: SceneHit
        """
        hit = SceneHit(gaussian_idx=-1) 
        hit_pos = ti.math.inf

        curr = 0  
        stack_idx = 0  

        while curr != -1 or stack_idx > 0:
            if curr != -1:
                node = self.bvh_field[curr]

                if not node.hit(ray):
                    if stack_idx > 0:
                        stack_idx -= 1
                        curr = self.stack[stack_idx] 
                    else:
                        curr = -1 
                    continue  

                if node.left == -1: 
                    for i in range(node.primitive_left, node.primitive_right):
                        gaussian = self.gaussian_field[i]
                        intersections = gaussian.hit(ray)
                        if intersections.x < ray.end and intersections.y > ray.start:
                            if intersections.x < hit_pos:
                                hit = SceneHit(gaussian_idx=i, intersections=intersections)
                                hit_pos = intersections.x
                    curr = -1 

                else:
                    if node.right != -1:
                        if stack_idx < self.max_node_num:
                            self.stack[stack_idx] = node.right  
                            stack_idx += 1

                    curr = node.left

            else:
                if stack_idx > 0:
                    stack_idx -= 1  
                    curr = self.stack[stack_idx]
                else:
                    curr = -1  

        return hit
       
        # for i in range(self.gaussian_field.shape[0]):
        #     gaussian = self.gaussian_field[i]
        #     intersections = gaussian.hit(ray)
        #     if intersections.x < ray.end and intersections.y > ray.start:
        #         if intersections.x < hit_pos:
        #             hit = SceneHit(gaussian_idx=i, intersections=intersections)
        #             hit_pos = intersections.x
        # return hit
