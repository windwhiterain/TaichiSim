from .common import *

@ti.func
def get_morton(idx:tt.vector(3,ti.u32))->ti.i64:
    split_x=_split(idx.x)
    split_y=_split(idx.y)<<1
    split_z=_split(idx.z)<<2
    return split_x|split_y|split_z

@ti.func
def _split(_:ti.u32)->ti.u64:
    ret=ti.u64(_)
    ret=ti.u64(ret&0x0000_0000_000F_FFFF)
    ret=ret|(ret<<32)
    ret&=ti.u64(0xFFFF_0000_0000_FFFF)
    ret=ret|(ret<<16)
    ret&=ti.u64(0x00FF_0000_FF00_00FF)
    ret=ret|(ret<<8)
    ret&=ti.u64(0xF00F_00F0_0F00_F00F)
    ret=ret|(ret<<4)
    ret&=ti.u64(0x30C3_0C30_C30C_30C3)
    ret=ret|(ret<<2)
    ret&=ti.u64(0x9249_2492_4924_9249)
    return ret

morton_invalid=0xFFFF_FFFF_FFFF_FFFF

@ti.data_oriented
class SparseDynamicField:
    def __init__(self,dtype,max_length:int,chunk_size=32) -> None:
        self.dtype=dtype
        self.keys=ti.field(ti.u64,max_length)
        self.values=ti.field(self.dtype,max_length)
        self.keys_temp=ti.field(ti.u64)
        self.keys_dynamic=root.dynamic(ti.i,max_length,chunk_size)
        self.keys_dynamic.place(self.keys_temp)
        self.values_temp=ti.field(self.dtype)
        self.values_dynamic=root.dynamic(ti.i,max_length,chunk_size)
        self.values_dynamic.place(self.values_temp)
        self.length=0
    @ti.func
    def append(self,key:ti.u64,value):
        self.keys_temp.append(key)
        self.values_temp.append(value)
    def _dense_temp(self):
        self.length=self.__dense_temp()
    @ti.kernel
    def __dense_temp(self) -> int:
        self.keys.fill(ti.u64(0xFFFF_FFFF_FFFF_FFFF))
        for i in self.keys_dynamic:
            self.keys[i]=self.keys_temp[i]
        for i in self.values_dynamic:
            self.values[i]=self.values_temp[i]  
        return self.keys_temp.length()
    def update(self):
        self._dense_temp()
        print(self.keys.shape)
        ti.algorithms.parallel_sort(self.keys,self.values)
    def clear(self):
        self.keys_dynamic.deactivate_all()
        self.values_dynamic.deactivate_all()
