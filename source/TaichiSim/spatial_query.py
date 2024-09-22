from cgi import print_form
from TaichiLib import *
from . import collision_handler

from abc import abstractmethod,ABC

class SpatialQuery(ABC):
    @abstractmethod
    def append(self,bound:BoundI,center:veci,idx:int):pass 
    @abstractmethod
    def update(self):pass
    @abstractmethod
    def clear(self):pass


@ti.data_oriented
class Grid(SpatialQuery):
    def __init__(self,elem_num:int) -> None:
        self.item_num=elem_num*2**dim
        self.dtype=tt.struct(idx=int,is_center=bool)
        self.keys=ti.field(ti.u64,self.item_num)
        self._clear_keys()
        self.values=self.dtype.field(shape=self.item_num)
        self.center_idxs=ti.field(int,elem_num)
    @ti.func
    def append(self,bound:Bound,center:veci,idx:int):
        boundi=bound.get_rounded()
        centeri=tm.round(center,int)
        for i,j,k in ti.static(ti.ndrange((0,2),(0,2),(0,2))):
            I=veci(i,j,k)
            I+=boundi.min
            if I.x<=boundi.max.x and I.y<=boundi.max.y and I.z<=boundi.max.z:
                item_idx=idx*8+i*4+i*2+k
                self.keys[item_idx]=get_morton(I)
                is_center=(centeri==I).all()
                self.values[item_idx]=self.dtype(idx=idx,is_center=is_center)
    @ti.kernel
    def update_overlaps(self):
        for i in self.values:
            value=self.values[i]
            if value.is_center:
                self.center_idxs[value.idx]=i
        
        for i in self.center_idxs:
            item_idx=self.center_idxs[i]
            value_i=self.values[item_idx]
            key_i=self.keys[item_idx]
            j=i+1
            step=1
            while j>=0 and j<self.item_num:
                key_j=self.keys[j]
                if key_j!=key_i:
                    if step==1:
                        j=i-1
                        step=-1
                        continue
                    else:
                        break
                value_j=self.values[j]
                self.collistion_action.on_query(value_i.idx,value_j.idx)
                j+=1
    @ti.kernel
    def _clear_keys(self):
        self.keys.fill(ti.u64(morton_invalid))
    def clear(self):
        self._clear_keys()
    def update(self,collistion_action:'collision_handler.CollisionAction'):
        self.collistion_action=collistion_action
        ti.algorithms.parallel_sort(self.keys,self.values)
        self.update_overlaps()
        