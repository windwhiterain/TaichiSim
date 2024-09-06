from TaichiLib import *
from . import spatial_query
from . import simulator

@ti.data_oriented
class CollisionHandler:
    def __init__(self,size:vec,unit:float,simulator:'simulator.Simulator') -> None:
        self.simulator=simulator
        self.unit=unit
        self.points_query=spatial_query.Grid(size,self.simulator.positions.shape[0])
    @ti.kernel
    def step(self):
        for i in self.simulator.positions:
            point=self.simulator.positions[i]
            self.points_query.register(Bound(point,0.1).round(self.unit),i)
        self.points_query.update_overlap()