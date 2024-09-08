from TaichiLib import *
from abc import ABC,abstractmethod

class Constraint(ABC):
    @abstractmethod
    def step(self,update_loss:bool):pass
    @abstractmethod
    def get_loss(self):pass



@ti.data_oriented
class MaxLength(Constraint):
    def __init__(self,max_length:float,edges:ti.Field,positions:ti.Field,masses:ti.Field) -> None:
        self.max_length=max_length
        self.edges=edges
        self.positions=positions
        self.masses=masses
        self.loss=ti.field(int,shape=())
        #temp
        self._delta_position=ti.field(vec,self.positions.shape)
        self._delta_position.fill(vec(0))
        self.update_loss=False
    def get_loss(self):
        return self.loss[None]
    def step(self,update_loss):
        self.update_loss=update_loss
        self._step()
    @ti.kernel
    def _step(self):
        update_loss=ti.static(self.update_loss)
        if update_loss:
            self.loss[None]=0
        for E in self.edges:
            edge=self.edges[E]
            position_x=self.positions[edge.x]
            position_y=self.positions[edge.y]
            direction=(position_y-position_x).normalized()
            mass_x=self.masses[edge.x]
            mass_y=self.masses[edge.y]
            length=(position_x-position_y).norm()
            delta_length=tm.max(0,length-self.max_length)
            if delta_length>0:
                delta_length=length-self.max_length*0.9
                self._delta_position[edge.x]+=delta_length*mass_y/(mass_x+mass_y)*direction
                self._delta_position[edge.y]-=delta_length*mass_x/(mass_x+mass_y)*direction
                if update_loss:
                    self.loss[None]=tm.max(self.loss[None],1)
        for V in self.positions:
            self.positions[V]+=self._delta_position[V]
            self._delta_position[V]=vec(0)


