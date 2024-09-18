from locale import normalize
from TaichiLib import *
from TaichiSim import simulator

@ti.data_oriented
class Target:
    def __init__(self,simulator:'simulator.Simulator',num:int,scale:float) -> None:
        self.simulator=simulator
        self.indices=ti.field(int,num)
        self.positions=ti.field(vec,num)
        self.scale=scale
    @ti.kernel
    def update_gradiants(self):
        simulator=self.simulator
        for V in self.indices:
            delta_vec=simulator.positions[V]-self.positions[V]
            simulator.gradiants[V]+=2*self.scale*delta_vec
    @ti.kernel
    def update_hession(self):
        simulator=self.simulator
        for V in self.indices:
            for d in ti.static(range(dim)):
                simulator.hession[V*dim+d,V*dim+d]+=2*self.scale