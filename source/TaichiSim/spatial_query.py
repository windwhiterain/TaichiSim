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
        self.ids_sparse_node=ti.root.pointer(ti.ijk,size)
        self.ids_sparse_node.dynamic(ti.l,16,2).place(self.ids)
        self.overlaps=ti.field(tt.u8)
        self.overlaps_sparse_node=ti.root.pointer(ti.ij,(elem_num,elem_num))
        self.overlaps_sparse_node.place(self.overlaps)
    @ti.func
    def register(self,bound:BoundI,id:int):
        for i,j,k in ti.ndrange((bound.min.x,bound.max.x+1),(bound.min.y,bound.max.y+1),(bound.min.z,bound.max.z+1)):
            self.ids[i,j,k].append(id)
    @ti.kernel
    def update_overlap(self):
        for i,j,k in self.ids_sparse_node:
            cell=self.ids_sparse_node[i,j,k]
            for n in range(0,cell.length()):
                for m in range(n+1,cell.length()):
                    self.overlaps[cell[n],cell[m]]=ti.u8(1)
    @ti.kernel
    def clear_ids(self):
        for i in ti.grouped(self.ids):
            print(i)
    def update(self):
        self.clear_ids()
        #self.update_overlap()
        self.overlaps_sparse_node.deactivate_all()