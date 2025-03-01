import taichi as ti
ti.init(arch=ti.vulkan)

@ti.dataclass
class Ray:
    """A ray representation.

    :param origin: the origin of the ray.
    :param dir: the ray direction.
    :param start: the minimum t for raycast.
    :param end: the maximun t for raycast, this should be set to ti.math.inf 
        for camera rays.
    """
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.start = 0.0
        self.end = float('inf')

    # @staticmethod
    # def field(shape):
    #     return ti.Struct.field({
    #         "origin": ti.types.vector(3, ti.f32),
    #         "direction": ti.types.vector(3, ti.f32),
    #     }, shape=shape)

    def get(self, t):
        return self.origin + t * self.direction
    
