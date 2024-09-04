from .common import *

@ti.func
def get_morton(position:tt.vector(3,float),unit:float)->ti.i64:
    split_x=_split(tm.round(position.x/unit,ti.i32))
    split_y=_split(tm.round(position.y/unit,ti.i32))<<1
    split_z=_split(tm.round(position.z/unit,ti.i32))<<2
    return split_x|split_y|split_z

@ti.func
def _split(_:ti.i32)->ti.i64:
    ret=ti.i64(_)
    ret=ret&0x0000_0000_000F_FFFF
    ret=ret|(ret<<32)
    ret&=0xFFFF_0000_0000_FFFF
    ret=ret|(ret<<16)
    ret&=0x00FF_0000_FF00_00FF
    ret=ret|(ret<<8)
    ret&=0xF00F_00F0_0F00_F00F
    ret=ret|(ret<<4)
    ret&=0x30C3_0C30_C30C_30C3
    ret=ret|(ret<<2)
    ret&=0x9249_2492_4924_9249
    return ret
