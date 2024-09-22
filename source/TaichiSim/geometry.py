from TaichiLib import *

ti.data_oriented
class Geometry:
    def __init__(self,num_point:int,num_triangle:int) -> None:
        self.positions=ti.field(vec,num_point)
        self.triangles=ti.field(veci,num_triangle)