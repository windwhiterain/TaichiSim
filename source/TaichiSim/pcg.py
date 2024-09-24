from TaichiLib import *
from .geometry import Geometry
from .energy import String

@ti.data_oriented
class Grid:
    def __init__(self,box:Box2,division:pairi) -> None:
        self.box=box
        self.division=division
        self.size_point=self.division+pairi(1)
        self.num_point=self.size_point.x*self.size_point.y
        self.num_quad=self.division.x*self.division.y
        #geometry
        self.num_triangles=self.num_quad*2
        self.geometry=Geometry(self.num_point,self.num_triangles)
        self.update_geometry()
        #simulate
        self.num_string_grid=2*self.num_quad+division.sum()
        self.num_string_cross=2*self.num_quad
        self.num_string=self.num_string_grid+self.num_string_cross
        self.string_energy=String(self.num_string,1)
        self.update_string()
        self.string_energy.update_edge()
    @ti.pyfunc
    def point_idx(self,x:int,y:int)->int:
        return flatten_idx2(pairi(x,y),pairi(self.size_point.x,self.size_point.y))
    @ti.pyfunc
    def triangle_idx(self,x:int,y:int,z:int)->int:
        return flatten_idx3(veci(x,y,z),veci(self.division,2))
    def update_geometry(self):
        self._update_geometry()
        self.geometry.update_triangle()
    @ti.kernel
    def _update_geometry(self):
        for i,j in ti.ndrange(*self.size_point):
            self.geometry.positions[self.point_idx(i,j)]=x0y(lerp(self.box.min,self.box.max,pair(i,j)/(self.size_point-pairi(1))))
        for i,j in ti.ndrange(*self.division):
            points=(
                self.point_idx(i,j),
                self.point_idx(i,j+1),
                self.point_idx(i+1,j),
                self.point_idx(i+1,j+1),
            )
            self.geometry.triangles[self.triangle_idx(i,j,0)]=veci(points[0],points[3],points[1])
            self.geometry.triangles[self.triangle_idx(i,j,1)]=veci(points[0],points[2],points[3])
    @ti.kernel
    def update_string(self):
        #axis edge
        start=0
        vertical_shape=pairi(self.size_point.x, self.division.y)
        for i, j in ti.ndrange(*vertical_shape):
            idx, idx1, idx2 = flatten_idx2(pairi(i,j),vertical_shape),self.point_idx(i,j),self.point_idx(i,j+1)
            self.string_energy.edges[idx] = pairi(idx1,idx2)
            self.string_energy.rest_lengths[idx] = (self.geometry.positions[idx1] - self.geometry.positions[idx2]).norm()
        start+=vertical_shape.x * vertical_shape.y

        horizontal_shape=pairi(self.division.x,self.size_point.y)
        for i, j in ti.ndrange(*horizontal_shape):
            idx, idx1, idx2 = (
                start + flatten_idx2(pairi(i,j),horizontal_shape),
                self.point_idx(i,j),
                self.point_idx(i+1,j),
            )
            self.string_energy.edges[idx] = pairi(idx1,idx2)
            self.string_energy.rest_lengths[idx] = (self.geometry.positions[idx1] - self.geometry.positions[idx2]).norm()
        start += horizontal_shape.x*horizontal_shape.y

        #diagnal edge
        for i, j in ti.ndrange(*self.division):
            idx, idx1, idx2 = (
                start + flatten_idx2(pairi(i,j),pairi(self.division)),
                self.point_idx(i,j),
                self.point_idx(i+1,j+1),
            )
            self.string_energy.edges[idx] = pairi(idx1,idx2)
            self.string_energy.rest_lengths[idx] = (self.geometry.positions[idx1] - self.geometry.positions[idx2]).norm()
        start += self.division.x*self.division.y
        for i, j in ti.ndrange(*self.division):
            idx, idx1, idx2 = (
                start + flatten_idx2(pairi(i,j),pairi(self.division)),
                self.point_idx(i+1,j),
                self.point_idx(i,j+1),
            )
            self.string_energy.edges[idx] = pairi(idx1,idx2)
            self.string_energy.rest_lengths[idx] = (self.geometry.positions[idx1] - self.geometry.positions[idx2]).norm()
            

