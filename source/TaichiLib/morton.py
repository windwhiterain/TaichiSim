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
