import taichi as ti
import math


@ti.data_oriented
class Camera:
    def __init__(self, position, looking_at,up, vert_fov, aspect_ratio):
        # Did not implement rotation yet!
        h = math.tan(math.radians(vert_fov) / 2.0)
        v_height = 2.0 * h
        v_width = v_height * aspect_ratio

        # direction = (looking_at - position).normalized()
        # dir=rot.rotate(direction)
        # looking_at = position + dir

        w = (position - looking_at).normalized()
        self.u = up.cross(w).normalized()
        self.v = w.cross(self.u)

        self.origin = position
        self.horizontal =  v_width * self.u
        self.vertical = v_height * self.v
        self.lower_left_corner = self.origin - (self.horizontal / 2.0) - (self.vertical / 2.0) - w

    @ti.func
    def generate_ray(self, u, v):
        return self.origin, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin
    



