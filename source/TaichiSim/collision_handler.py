from TaichiLib import *
from . import spatial_query
from . import simulator

@ti.data_oriented
class CollisionHandler:
    def __init__(self,unit:float,simulator:'simulator.Simulator') -> None:
        self.simulator=simulator
        self.unit=unit
        self.points_query=spatial_query.Grid(self.simulator.NV)
    @ti.func
    def vec_into_space(self,point:vec)->vec:
        return (point-self.simulator.bound.min)/self.unit
    @ti.func
    def bound_into_space(self,bound:Bound)->Bound:
        return Bound(self.vec_into_space(bound.min),self.vec_into_space(bound.max))
    @ti.kernel
    def append_points(self):
        for i in self.simulator.positions:
            point=self.simulator.positions[i]
            self.points_query.append(self.bound_into_space(Bound(point,0.1)).round(),i,i)
        
    def step(self):
        self.points_query.clear()
        self.append_points()
        self.points_query.update()