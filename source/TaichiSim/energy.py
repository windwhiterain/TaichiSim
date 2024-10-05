from locale import normalize
from TaichiLib import *
from TaichiSim import simulator

class Energy:
    def __init__(self,num:int,scale:float) -> None:
        self.simulator:'simulator.Simulator'=None
        self.num=num
        self.scale=scale
        self.scales=ti.field(float,self.num)
        self.scales.fill(1)
    def update_value(self):pass
    def update_gradiants(self):pass
    def update_hession(self):pass
    @ti.pyfunc
    def get_scale(self,idx:int):
        return self.scale*self.scales[idx]


@ti.data_oriented
class Target(Energy):
    def __init__(self,num:int,scale:float) -> None:
        super().__init__(num,scale)
        self.indices=ti.field(int,self.num)
        self.positions=ti.field(vec,self.num)
    @ti.kernel
    def update_gradiants(self):
        simulator=self.simulator
        for i in self.indices:
            p=self.indices[i]
            delta_vec=simulator.constrainted_positions[p]-self.positions[i]
            simulator.gradiant[p]+=2*self.get_scale(i)*delta_vec
    @ti.kernel
    def update_hession(self):
        simulator=self.simulator
        for i in self.indices:
            p=self.indices[i]
            simulator.hession[p,p]+=mat.identity()*2*self.get_scale(i)

@ti.data_oriented
class String(Energy):
    def __init__(self, num: int, scale: float) -> None:
        super().__init__(num, scale)
        self.edges=ti.field(pairi,self.num)
        self.rest_lengths=ti.field(float,self.num)
        #simulate
        self.edge_hessions=ti.field(mat,self.num)
        #render
        self.indices=ti.field(int,self.num*2)
    @ti.kernel
    def update_gradiants(self):
        for i in self.edges:
            scale=self.scales[i]
            edge=self.edges[i]
            edge_vec=self.simulator.constrainted_positions[edge.y]-self.simulator.constrainted_positions[edge.x]
            norm=(edge_vec.norm()-self.rest_lengths[i])*self.get_scale(i)
            direction=edge_vec.normalized()
            gradiant=direction*norm
            self.simulator.gradiant[edge[0]]-=gradiant
            self.simulator.gradiant[edge[1]]+=gradiant
    def update_hession(self):
        self.update_string_hessions()
        self._update_hession()
    @ti.kernel
    def update_string_hessions(self):
        for i in self.edges:
            edge=self.edges[i]
            edge_vec=self.simulator.constrainted_positions[edge.y]-self.simulator.constrainted_positions[edge.x]
            length=edge_vec.norm()
            rest_len=self.rest_lengths[i]
            scale=self.scales[i]
            outer_product=edge_vec.outer_product(edge_vec)
            self.edge_hessions[i]=self.get_scale(i)*(outer_product/(length)**2+rest_len/length*(ti.Matrix.identity(float,dim)-outer_product/length**2))
    @ti.kernel
    def _update_hession(self):
        for i in self.edges:
            edge=self.edges[i]
            string_hession=self.edge_hessions[i]
            self.simulator.hession[edge[0],edge[0]]+=string_hession
            self.simulator.hession[edge[1],edge[1]]+=string_hession
            self.simulator.hession[edge[0],edge[1]]-=string_hession
            self.simulator.hession[edge[1],edge[0]]-=string_hession
    @ti.kernel
    def update_edge(self):
        for i in self.edges:
            for d in ti.static(range(2)):
                self.indices[i*2+d]=self.edges[i][d]