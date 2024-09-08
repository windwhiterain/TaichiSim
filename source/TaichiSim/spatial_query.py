from cgi import print_form
from TaichiLib import *

from abc import abstractmethod,ABC

class SpatialQuery(ABC):
    @abstractmethod
    def append(self,bound:BoundI,elem_idx:int,id:int):pass 
    @abstractmethod
    def update(self):pass
    @abstractmethod
    def clear(self):pass


@ti.data_oriented
class Grid(SpatialQuery):
    def __init__(self,elem_num:int) -> None:
        self.keys=ti.field(ti.u64,elem_num*2**dim)
        self._clear_keys()
        self.ids=ti.field(int,elem_num*2**dim)
        self.overlaps=ti.field(ti.u8)
        self.overlaps_sparse_node=root.pointer(ti.ij,(elem_num,elem_num))
        self.overlaps_sparse_node.place(self.overlaps)
    @ti.func
    def append(self,bound:BoundI,elem_idx:int,id:int):
        for i in ti.static(ti.grouped(ti.ndrange((0,2),(0,2),(0,2)))):
            if i.x<=bound.max.x and i.y<=bound.max.y and i.z<=bound.max.z:
                idx=elem_idx+i.x*4+i.y*2+i.z
                self.keys[idx]=get_morton(ti.Vector([i.x,i.y,i.z]))
                self.ids[idx]=id
    @ti.kernel
    def update_overlaps(self):
        length=self.keys.shape[0]
        for i in self.keys:
            key_i=self.keys[i]
            if key_i==ti.u64(morton_invalid):
                continue
            id_i=self.ids[i]
            j=i+1
            while j<length:
                key_j=self.keys[j]
                if key_j!=key_i:
                    break
                id_j=self.ids[j]
                self.overlaps[id_i,id_j]=ti.u8(1)
                self.overlaps[id_j,id_i]=ti.u8(1)
                j+=1
    @ti.kernel
    def _clear_keys(self):
        self.keys.fill(ti.u64(morton_invalid))
    def clear(self):
        self._clear_keys()
        self.overlaps_sparse_node.deactivate_all()
    def update(self):
        ti.algorithms.parallel_sort(self.keys,self.ids)
        self.update_overlaps()
        