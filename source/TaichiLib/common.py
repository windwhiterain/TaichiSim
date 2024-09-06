import taichi as ti
import taichi.math as tm
import taichi.types as tt

dim=3
vec=tt.vector(dim,float)
veci=tt.vector(dim,int)
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
    min:vec
    max:vec
    @ti.func
    def round(self,unit:float=1.0)->'BoundI':
        return BoundI(tm.round(self.min/unit,int),tm.round(self.max/unit,int))

@ti.func
def get_sphere_bound(center:vec,radius:float)->Bound:
    return Bound(center-vec(radius),center+vec(radius))

@ti.dataclass
class BoundI:
    min:veci
    max:veci

@ti.func
def get_traingle_bound(triangle:Triangle)->Bound:
    ret=Bound()
    for d in ti.static(range(dim)):
        ret.min[d]=tm.min(triangle.x[d],triangle.y[d],triangle.z[d])
        ret.max[d]=tm.max(triangle.x[d],triangle.y[d],triangle.z[d])





