from math import inf
from os import remove
from turtle import Shape
from .math import *

@ti.func
def is_adjacent(x:pairi,y:pairi):
    return x.x==y.x or x.x==y.y or x.y==y.x or x.y==y.y

@ti.pyfunc
def flatten_idx2(idx:pairi,shape:pairi)->int:
    return idx.x*shape.y+idx.y

@ti.pyfunc
def flatten_idx3(idx:veci,shape:veci)->int:
    return idx.x*shape.y*shape.z+idx.y*shape.z+idx.z

@ti.pyfunc
def x0y(_:pair)->vec:
    return vec(_.x,0,_.y)

@ti.pyfunc
def lerp(x:ti.template(),y:ti.template(),f:float)->ti.template():
    return x*(1-f)+y*f

@ti.dataclass
class Bound:
    min:vec
    max:vec
    @ti.pyfunc
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
class Box2:
    min:pair
    max:pair

@ti.dataclass
class BoundI:
    min:veci
    max:veci


@ti.dataclass
class BoxI2:
    min:pairi
    max:pairi
    @ti.pyfunc
    def size(self)->veci:
        return self.max-self.min+pairi(1)

@ti.dataclass
class Sphere:
    center:vec
    radius:float
    @ti.pyfunc
    def get_bound_box(self):
        return Bound(self.center-vec(self.radius),self.center+vec(self.radius))
    @ti.pyfunc
    def get_scaled(self,scale:float)->'Sphere':
        return Sphere(self.center,self.radius*scale)
    @ti.pyfunc
    def get_extended(self,distance:float)->'Sphere':
        return Sphere(self.center,self.radius+distance)


@ti.dataclass
class Triangle:
    x:vec
    y:vec
    z:vec
    @ti.pyfunc
    def get_bound(self)->'Bound':
        ret=Bound()
        for d in ti.static(range(dim)):
            ret.min[d]=tm.min(self.x[d],self.y[d],self.z[d])
            ret.max[d]=tm.max(self.x[d],self.y[d],self.z[d])
        return ret
    @ti.pyfunc
    def get_bound_sphere(self)->Sphere:
        ret=Sphere()
        dotABAB = tm.dot(self.y - self.x, self.y - self.x)
        dotABAC = tm.dot(self.y - self.x, self.z - self.x)
        dotACAC = tm.dot(self.z - self.x, self.z - self.x)
        d = 2*(dotABAB*dotACAC - dotABAC*dotABAC)
        referencePt = self.x
        if ti.abs(d) <= epsilon:
            # a, b, and c lie on a line. Circle center is center of AABB of the
            # points, and radius is distance from circle center to AABB corner
            bbox = self.get_bound()
            ret.center = 0.5 * (bbox.min + bbox.max);
            referencePt = bbox.min;
        else:
            s = (dotABAB*dotACAC - dotACAC*dotABAC) / d
            t = (dotACAC*dotABAB - dotABAB*dotABAC) / d
            # s controls height over AC, t over AB, (1-s-t) over BC
            if s <= 0:
                ret.center  = 0.5 * (self.x + self.z)
            elif t <= 0:
                ret.center  = 0.5 * (self.x + self.y)
            elif s + t >= 1: 
                ret.center  = 0.5 * (self.y + self.z)
                referencePt = self.y
            else:
                ret.center  = self.x + s*(self.y - self.x) + t*(self.z - self.x)
        ret.radius = (ret.center - referencePt).norm()
        return ret
    
@ti.dataclass
class Segment:
    x:vec
    y:vec
    @ti.pyfunc
    def get_length(self):
        return (self.x-self.y).norm()
    @ti.pyfunc
    def get_bound(self):
        ret=Bound()
        for d in ti.static(range(dim)):
            ret.min[d]=tm.min(self.x[d],self.y[d])
            ret.max[d]=tm.max(self.x[d],self.y[d])
        return ret
    @ti.pyfunc
    def get_bound_sphere(self)->Sphere:
        return Sphere((self.x+self.y)/2,tm.length(self.x-self.y)/2)
    @ti.pyfunc
    def get_vector(self)->vec:
        return self.y-self.x

@ti.func
def get_sphere_bound(center:vec,radius:float)->Bound:
    return Bound(center-vec(radius),center+vec(radius))

@ti.func
def get_distance_point_segment(point:vec,segment:Segment)->tt.struct(parameter=float,vector=vec):
    length=(segment.y-segment.x).norm()
    direction=(segment.y-segment.x)/length
    displace=tm.dot(point-segment.x,direction)
    displace=tm.clamp(displace,0,length)
    return ti.Struct(parameter=displace/length,vector=(segment.x+direction*displace)-point)

@ti.func
def get_distance_line(line_x:Segment,line_y:Segment)->tt.struct(parameter=pair,vector=vec):
    vector_x=line_x.y-line_x.x
    vector_y=line_y.y-line_y.x
    n=tm.cross(vector_x,vector_y)
    n_x=tm.cross(vector_x,n)
    n_y=tm.cross(vector_y,n)
    parameter_x=tm.dot(line_y.x-line_x.x,n_y)/tm.dot(vector_x,n_y)
    parameter_y=tm.dot(line_x.x-line_y.x,n_x)/tm.dot(vector_y,n_x)
    return ti.Struct(parameter=pair(parameter_x,parameter_y),vector=(line_y.x+parameter_y*vector_y)-(line_x.x+parameter_x*vector_x))

@ti.pyfunc
def get_distance_segment(segment_x:Segment,segment_y:Segment)->tt.struct(paramter=pair,vector=vec):
    vector=vec(0)
    parameter=pair(0)
    vector_x=segment_x.get_vector()
    vector_y=segment_y.get_vector()
    if (ti.abs(tm.cross(vector_x,vector_y))<=vec(epsilon)).all():
        param_yx=tm.dot(segment_y.x-segment_x.x,vector_x)
        param_yy=tm.dot(segment_y.y-segment_x.x,vector_x)
        if param_yx>1 and param_yy>1:
            if param_yx<param_yy:
                vector=segment_y.x-segment_x.y
                parameter=pair(1,0)
            else:
                vector=segment_y.y-segment_x.y
                parameter=pair(1,1)
        elif param_yx<0 and param_yy<0:
            if param_yx>param_yy:
                vector=segment_y.x-segment_x.x
                parameter=pair(0,1)
            else:
                vector=segment_y.y-segment_x.x
                parameter=pair(0,1)
        else:
            param_center=(tm.max(0,tm.min(param_yx,param_yy))+tm.min(1,tm.max(param_yx,param_yy)))/2
            parameter=(param_center,remap01(param_center,param_yx,param_yy))
            vector=remove_component(segment_y.x-segment_x.x,vector_x)
    else:
        distance_line=get_distance_line(segment_x,segment_y)
        if distance_line.parameter.x>=0 and distance_line.parameter.x<=1 and distance_line.parameter.y>=0 and distance_line.parameter.y<=1:
            vector=distance_line.vector
            parameter=distance_line.parameter
        else:
            parameter_vectors=(
                get_distance_point_segment(segment_x.x,segment_y),
                get_distance_point_segment(segment_x.y,segment_y),
                get_distance_point_segment(segment_y.x,segment_x),
                get_distance_point_segment(segment_y.y,segment_x)
            )
            parameters=(
                pair(0,parameter_vectors[0].parameter),
                pair(1,parameter_vectors[1].parameter),
                pair(parameter_vectors[2].parameter,0),
                pair(parameter_vectors[3].parameter,1),
            )
            vectors=(
                parameter_vectors[0].vector,
                parameter_vectors[1].vector,
                -parameter_vectors[2].vector,
                -parameter_vectors[3].vector   
            )
            min_distance=inf
            for i in ti.static(range(4)):
                distance_sqr=vectors[i].norm_sqr()
                if distance_sqr<min_distance:
                    min_distance=distance_sqr
                    vector=vectors[i]
                    parameter=parameters[i]
    return ti.Struct(parameter=parameter,vector=vector)








