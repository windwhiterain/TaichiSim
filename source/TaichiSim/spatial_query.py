from TaichiLib import *
from TaichiLib.linq import Linq
from . import collision_handler

class SpatialQuery:
    def append(self,bound:BoundI,center:veci,idx:int):pass 
    def update(self):pass
    def clear(self):pass


@ti.data_oriented
class Grid(SpatialQuery):
    def __init__(self,elem_num:int) -> None:
        self.max_item_num=elem_num*2**dim
        self.dtype=tt.struct(idx=int,is_center=bool)
        self.keys=ti.field(ti.u64,self.max_item_num)
        self._clear_keys()
        self.values=self.dtype.field(shape=self.max_item_num)
        self.center_idxs=ti.field(int,elem_num)
        self.item_num=ti.field(int,())
    @ti.func
    def append(self,bound:Bound,center:vec,idx:int):
        boundi=bound.get_rounded()
        centeri=tm.round(center,int)
        for i,j,k in ti.static(ti.ndrange((0,2),(0,2),(0,2))):
            I=veci(i,j,k)
            I+=boundi.min
            if I.x<=boundi.max.x and I.y<=boundi.max.y and I.z<=boundi.max.z:
                item_idx=idx*8+i*4+j*2+k
                self.keys[item_idx]=get_morton(I)
                is_center=(centeri==I).all()
                self.values[item_idx]=self.dtype(idx=idx,is_center=is_center)
    @ti.kernel
    def update_overlaps(self,on_query:ti.template()):
        self.item_num[None]=self.max_item_num
        for i in range(self.max_item_num-1):
            prev_key=self.keys[i]
            next_key=self.keys[i+1]
            if prev_key!=ti.u64(morton_invalid) and next_key==ti.u64(morton_invalid):
                self.item_num[None]=i+1
        for i in range(self.item_num[None]):
            value=self.values[i]
            if value.is_center:
                self.center_idxs[value.idx]=i
        for i in self.center_idxs:
            item_idx=self.center_idxs[i]
            key_i=self.keys[item_idx]
            j=item_idx+1
            step=1
            while True:
                if j<0 or j>=self.max_item_num or self.keys[j]!=key_i:
                    if step==1:
                        j=item_idx-1
                        step=-1
                        continue
                    else:
                        break
                value_j=self.values[j]
                on_query(i,value_j.idx)
                j+=step
    @ti.kernel
    def _clear_keys(self):
        self.keys.fill(ti.u64(morton_invalid))
    def clear(self):
        self._clear_keys()
    def update(self):
        ti.algorithms.parallel_sort(self.keys,self.values)  
    def tranverse(self,on_query:Callable[[int,int],None]):
        self.update_overlaps(on_query)
        