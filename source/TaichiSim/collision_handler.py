from TaichiLib import *
from . import spatial_query
from . import simulator

@ti.data_oriented
class CollisionHandler:
    def __init__(self,unit:float,simulator:'simulator.Simulator') -> None:
        self.simulator=simulator
        self.unit=unit
        self.points_query=spatial_query.Grid(self.get_sizei(),self.simulator.NV)
    @ti.kernel
    def get_sizei(self)->veci:
        return tm.round(self.simulator.bound.to_size(self.unit),int)
    @ti.func
    def into_space(self,point:vec)->vec:
        return (point-self.simulator.bound.min)/self.unit
    @ti.kernel
    def register_points(self):
        for i in self.simulator.positions:
            point=self.simulator.positions[i]
            self.points_query.register(Bound(self.into_space(point),0.1).round(self.unit),i)
        
    def step(self):
        #self.register_points()
        self.points_query.update()