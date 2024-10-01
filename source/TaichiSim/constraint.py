from TaichiLib import *
from . import simulator

class Constraint:
    def __init__(self,scale:float) -> None:
        super().__init__()
        self.scale=scale
        self.simulator:'simulator.Simulator'=None
    def update(self):pass
    def step(self,update_loss:bool):pass
    def get_loss(self)->float:return 0

@ti.data_oriented
class MaxLength(Constraint):
    def __init__(self, scale: float, max_length:float) -> None:
        super().__init__(scale)
        self.loss=ti.field(int,shape=())
        self.max_length=max_length
    def get_loss(self):
        return self.loss[None]
    def step(self,update_loss:bool):
        self._step(update_loss)
    @ti.kernel
    def _step(self,update_loss:ti.template()):
        UPDATE_LOSS=ti.static(update_loss)
        simulator=self.simulator
        geometry=simulator.geometry
        if UPDATE_LOSS:
            self.loss[None]=0
        for e in geometry.edges:
            edge=geometry.edges[e]
            position_x=geometry.positions[edge.x]
            position_y=geometry.positions[edge.y]
            direction=(position_y-position_x).normalized()
            mass_x=geometry.masses[edge.x]
            mass_y=geometry.masses[edge.y]
            length=(position_x-position_y).norm()
            delta_length=tm.max(0,length-self.max_length)
            if delta_length>0:
                delta_length=length-self.max_length/self.scale
                ti.atomic_add(simulator.delta_positions[edge.x],delta_length*mass_y/(mass_x+mass_y)*direction)
                ti.atomic_sub(simulator.delta_positions[edge.y],delta_length*mass_x/(mass_x+mass_y)*direction)
                if UPDATE_LOSS:
                    self.loss[None]=tm.max(self.loss[None],1)


