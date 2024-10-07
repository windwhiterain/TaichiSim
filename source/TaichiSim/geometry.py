from turtle import position
from TaichiLib import *

@ti.data_oriented
class Geometry:
    def __init__(self,num_point:int,num_triangle:int) -> None:
        self.num_point=num_point
        self.num_triangle=num_triangle
        self.positions=ti.field(vec,self.num_point)
        self.faces=ti.field(veci,self.num_triangle)
        self.masses=ti.field(float,self.num_point)
        self.masses.fill(1)
        #render
        self.indices=ti.field(int,self.num_triangle*3)
    def update_triangle(self):
        temp_edge=set[pairi]()
        for i in FieldIterator(self.faces):
            temp_edge.add((i.x,i.y))
            temp_edge.add((i.x,i.z))
            temp_edge.add((i.y,i.z))
        self.num_edge=len(temp_edge)
        self.edges=ti.field(pairi,self.num_edge)
        for k,v in enumerate(temp_edge):
            self.edges[k]=pairi(v[0],v[1])
        self._update_triangle()
    @ti.kernel
    def _update_triangle(self):
        for i in self.faces:
            for d in ti.static(range(3)):
                self.indices[i*3+d]=self.faces[i][d]
    @ti.func
    def get_triangle(self,positions:ti.template(),idx:int):
        face=self.faces[idx]
        return Triangle(positions[face.x],positions[face.y],positions[face.z])
        
   

