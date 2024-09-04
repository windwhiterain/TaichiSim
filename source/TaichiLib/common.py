import taichi as ti
import taichi.math as tm
import taichi.types as tt

dim=3
vec=tt.vector(dim,float)
pairi=tt.vector(2,int)
mat=tt.matrix(dim,dim,float)
up=vec(0,0,1)

@ti.dataclass
class Triangle:
    x:vec
    y:vec
    z:vec

@ti.dataclass
class Bound:
    max:vec
    min:vec

@ti.func
def get_bound(triangle:Triangle)->Bound:
    ret=Bound()
    for d in ti.static(range(dim)):
        ret.max[d]=tm.max(triangle.x[d],triangle.y[d],triangle.z[d])
        ret.min[d]=tm.min(triangle.x[d],triangle.y[d],triangle.z[d])




