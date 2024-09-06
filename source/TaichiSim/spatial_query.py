from TaichiLib import *

from abc import abstractmethod,ABC

class SpatialQuery(ABC):
    @abstractmethod
    def register(self,bound:BoundI,id:int):pass 
    @abstractmethod
    def update_overlap(self):pass


@ti.data_oriented
class Grid(SpatialQuery):
    def __init__(self,size:veci,elem_num:int) -> None:
        self.ids=ti.field(int)
        ti.root.pointer(ti.ijk,size).dynamic(ti.i,16)
        self.overlaps=ti.field(tt.u1)
        ti.root.pointer(ti.ij,(elem_num,elem_num))
    @ti.func
    def register(self,bound:BoundI,id:int):
        for i in ti.grouped(ti.ndrange((bound.min.x,bound.max.x+1),(bound.min.y,bound.max.y+1),(bound.min.z,bound.max.z+1))):
            self.ids[i].append(id)
    @ti.kernel
    def update_overlap(self):
        for i in ti.grouped(self.ids):
            for j in self.ids[i]:
                for k in range(j,self.ids[i].length):
                    self.overlaps[j,k]=1