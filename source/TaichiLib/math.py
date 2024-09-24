import taichi as ti
import taichi.math as tm
import taichi.types as tt

dim=3
vec=tt.vector(dim,float)
veci=tt.vector(dim,int)
pair=tt.vector(2,float)
pairi=tt.vector(2,int)
mat=tt.matrix(dim,dim,float)
@ti.pyfunc
def mat_identity()->mat:
    return ti.Matrix.identity(float,dim)
mat.identity=mat_identity
up=vec(0,0,1)
root:ti.SNode=ti.root
epsilon=1e-8

@ti.pyfunc
def remove_component(self:vec,component:vec)->vec:
    return self-tm.dot(self,component)*component.normalized()