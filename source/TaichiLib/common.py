import taichi as ti
import taichi.math as tm
import taichi.types as tt

dim=3
vec=tt.vector(dim,float)
veci=tt.vector(dim,int)
pair=tt.vector(2,float)
pairi=tt.vector(2,int)
mat=tt.matrix(dim,dim,float)
up=vec(0,0,1)
root:ti.SNode=ti.root

@ti.func
def is_adjacent(x:pairi,y:pairi):
    return x.x==y.x or x.x==y.y or x.y==y.x or x.y==y.y

@ti.dataclass
class Bound:
    min:vec
    max:vec
    @ti.func
    def get_rounded(self,unit:float=1.0)->'BoundI':
        return BoundI(tm.round(self.min/unit,int),tm.round(self.max/unit,int))
    @ti.pyfunc
    def get_size(self,unit:float=1.0)->vec:
        return (self.max-self.min)/unit
    @ti.pyfunc
    def get_center(self)->vec:
        return (self.min+self.max)/2
    @ti.pyfunc
    def get_extended(self,distance:float)->'Bound':
        return Bound(self.min-vec(distance),self.max+vec(distance))

@ti.dataclass
class BoundI:
    min:veci
    max:veci

@ti.dataclass
class Triangle:
    x:vec
    y:vec
    z:vec
    @ti.func
    def get_bound(self)->'Bound':
        ret=Bound()
        for d in ti.static(range(dim)):
            ret.min[d]=tm.min(self.x[d],self.y[d],self.z[d])
            ret.max[d]=tm.max(self.x[d],self.y[d],self.z[d])
        return ret
    
@ti.dataclass
class Segment:
    x:vec
    y:vec
    @ti.func
    def get_length(self):
        return (self.x-self.y).norm()
    @ti.func
    def get_bound(self):
        ret=Bound()
        for d in ti.static(range(dim)):
            ret.min[d]=tm.min(self.x[d],self.y[d])
            ret.max[d]=tm.max(self.x[d],self.y[d])
        return ret

@ti.func
def get_sphere_bound(center:vec,radius:float)->Bound:
    return Bound(center-vec(radius),center+vec(radius))

@ti.func
def get_distance_point_segment(point:vec,segment:Segment)->tt.struct(parameter=float,distance=float):
    length=(segment.y-segment.x).norm()
    direction=(segment.y-segment.x)/length
    displace=tm.dot(point-segment.x,direction)
    displace=tm.clamp(displace,0,length)
    return ti.Struct(parameter=displace/length,distance=((segment.x+direction*displace)-point).norm())

@ti.func
def get_distance_line(line_x:Segment,line_y:Segment)->tt.struct(parameter=pair,distance=float):
    vector_x=line_x.y-line_x.x
    vector_y=line_y.y-line_y.x
    n=tm.cross(vector_x,vector_y)
    n_x=tm.cross(vector_x,n)
    n_y=tm.cross(vector_y,n)
    parameter_x=tm.dot(line_y.x-line_x.x,n_y)/tm.dot(vector_x,n_y)
    parameter_y=tm.dot(line_x.x-line_y.x,n_x)/tm.dot(vector_y,n_x)
    return ti.Struct(parameter=pair(parameter_x,parameter_y),distance=((line_x.x+parameter_x*vector_x)-(line_y.x+parameter_y*vector_y)).norm())

@ti.func
def get_distance_segment(segment_x:Segment,segment_y:Segment)->float:
    distance_line=get_distance_line(segment_x,segment_y)
    ret=0.
    if distance_line.parameter.x>=0 and distance_line.parameter.x<=1 and distance_line.parameter.y>=0 and distance_line.parameter.y<=1:
        ret=distance_line.distance
    else:
        ret=tm.min(
            get_distance_point_segment(segment_x.x,segment_y).distance,
            get_distance_point_segment(segment_x.y,segment_y).distance,
            get_distance_point_segment(segment_y.x,segment_x).distance,
            get_distance_point_segment(segment_y.x,segment_x).distance
        )
    return ret








