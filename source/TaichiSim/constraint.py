from TaichiLib import *
from abc import ABC,abstractmethod, abstractproperty

class Constraint(ABC):
    def set_tightness(self,tightness:float):
        self.tightness=tightness
    @abstractmethod
    def step(self,update_loss:bool):pass
    @abstractmethod
    def get_loss(self):pass

def update_constraints(constraints:list[Constraint],group_num:int,max_loss):
    def check_round()->bool:
        ret=True
        for constraint in constraints:
            constraint.step(True)
            if constraint.get_loss()>max_loss:
                ret=False
        return ret
    def uncheck_round(round_num:int):
        for iteration in range(round_num):
            for constraint in constraints:
                constraint.step(False)
    if check_round():
        return
    while True:
        uncheck_round(group_num)
        if check_round():
            return


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
    def step(self,update_loss:bool):
        self._delta_position.fill(vec(0))
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
                delta_length=length-self.max_length*self.tightness
                ti.atomic_add(self._delta_position[edge.x],delta_length*mass_y/(mass_x+mass_y)*direction)
                ti.atomic_sub(self._delta_position[edge.y],delta_length*mass_x/(mass_x+mass_y)*direction)
                if update_loss:
                    self.loss[None]=tm.max(self.loss[None],1)
        if not update_loss:
            for V in self.positions:
                self.positions[V]+=self._delta_position[V]


