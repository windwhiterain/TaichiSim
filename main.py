

import numpy as np

import taichi as ti
import taichi.math as tm
import taichi.types as tt

from copy import copy
from taichi.linalg import SparseMatrixBuilder
from taichi.linalg import SparseSolver
from abc import ABC, abstractmethod


dim=3
vec=tt.vector(dim,float)
pairi=tt.vector(2,int)
mat=tt.matrix(dim,dim,float)
up=vec(0,0,1)

class Solver(ABC):
    def fit(self,simulator:'Simulator'):
        self.simulator=simulator
        self._fit()
    def _fit(self):pass
    @abstractmethod
    def get_requires(self)->list[str]:pass
    def begin_step(self,dt:float):
        self.dt=dt
        self._begin_step()
    def _begin_step(self):pass
    @abstractmethod
    def temp_step(self):pass
    def end_step(self):pass

@ti.data_oriented
class NewtonRaphsonSolver(Solver):
    def __init__(self) -> None:
        super().__init__()    
    #impl Solver
    def _fit(self):
        self.b=ti.ndarray(float,dim*self.simulator.NV)
    def get_requires(self):
        return ['M','gradiants','H']
    def _begin_step(self):
        self.solver = SparseSolver(solver_type="LDLT")
        self.A_pattern_dirty=True
    def temp_step(self):
        A=self.simulator.M*(1/self.dt**2)+self.simulator.H
        if self.A_pattern_dirty:
            self.solver.analyze_pattern(A)
            self.A_pattern_dirty=False
        self.solver.factorize(A)
        self.update_b(self.dt)
        d_positions=self.solver.solve(self.b)
        self.update_temp_position(d_positions)
    #
    def update_b(self,dt:float):
        self._update_b(dt,self.b)
    @ti.kernel
    def _update_b(self,dt:float,b:tt.ndarray()):
        for V in range(self.simulator.NV):
            mass=self.simulator.masses[V]
            for d in ti.static(range(dim)):
                b[dim*V+d]=-mass*(self.simulator.temp_positions[V][d]-self.simulator.positions[V][d])/dt-self.simulator.gradiants[V][d]

    @ti.kernel
    def update_temp_position(self,d_position:tt.ndarray()):
        for V in range(self.simulator.NV):
            if self.simulator.mask[V]:
                for d in ti.static(range(dim)):
                    self.simulator.temp_positions[V][d]+=d_position[dim*V+d]

@ti.data_oriented
class ProjectiveDynamicSolver(Solver):
    def __init__(self) -> None:
        super().__init__()
    #impl Solver
    def fit(self,simulator:'Simulator'):
        pass


@ti.data_oriented
class ConjugateHessionSolver(Solver):
    pass

@ti.data_oriented
class DiagnalHessionSolver(Solver):
    def __init__(self) -> None:
        super().__init__()
    #impl Solver
    def _fit(self):
        self.diag_hession=ti.field(vec,self.simulator.NV)
    def get_requires(self) -> list[str]:
        return ['hessions','gradiants']
    def temp_step(self):
        self.update_diag_hession()
        self.update_temp_position()
    #
    @ti.kernel
    def update_diag_hession(self):
        self.diag_hession.fill(vec(0))
        for E in range(self.simulator.NE):
            edge=self.simulator.edges[E]
            hession=self.simulator.hessions[E]
            for d in ti.static(range(dim)):
                self.diag_hession[edge[0]][d]+=hession[d,d]
                self.diag_hession[edge[1]][d]+=hession[d,d]

    @ti.kernel
    def update_temp_position(self):
        for V in range(self.simulator.NV):
            if self.simulator.mask[V]:
                mass=self.simulator.masses[V]
                temp_position=self.simulator.temp_positions[V]
                self.simulator.temp_positions[V]+=(-mass*(temp_position-self.simulator.positions[V])/self.dt**2-self.simulator.gradiants[V])/(mass/self.dt**2+self.diag_hession[V])


@ti.data_oriented
class Simulator:
    require_dependencies={
        'M':[],
        'gradiants':[],
        'hessions':[],
        'H':['hessions'],
    }
    def __init__(self,N:int,k:float,solver:Solver) -> None:
        #geometry
        self.create_geometry(N)

        #attribute
        self.elasticities=ti.field(float,self.NE)
        self.elasticities.fill(k)
        
        self.masses=ti.field(float,self.NV)
        self.masses.fill(1)
        self.M_builder=SparseMatrixBuilder(dim * self.NV, dim * self.NV, max_num_triplets=self.NV*dim)
        self.update_M()

        self.mask=ti.field(bool,self.NV)
        self.mask.fill(True)

        #state
        self.positions=ti.field(vec,self.NV)
        self.positions.copy_from(self.init_positions)
        self.velocities=ti.field(vec,self.NV)
        self.velocities.fill(vec(0))
        
        #simulate
        self.temp_positions=ti.field(vec,self.NV)
        self.gradiants=ti.field(vec,self.NV)
        self.hessions=ti.field(mat,self.NE)
        self.H_builder=SparseMatrixBuilder(dim * self.NV, dim * self.NV, max_num_triplets=self.NE*dim**2*4) 
        self.b=ti.ndarray(float,dim*self.NV)
        self.step=4

        #solver
        self.solver=solver
        self.solver.fit(self)
        self.solve_requrie_dependency(solver.get_requires())
        
    def solve_requrie_dependency(self,requires:list[str]):
        self.requires=requires
        for require in requires:
            dependencies=self.require_dependencies[require]
            for dependency in dependencies:
                if not dependency in self.requires:
                    self.requires.append(dependency)

    def create_geometry(self,N:int):
        self.N=N
        self.NV = (N + 1) ** 2  # number of vertices
        self.NE = 2 * N * (N + 1) + 2 * N * N  # numbser of edges
        self.init_positions=ti.field(vec,self.NV)
        self.edges=ti.field(pairi,(self.NE))
        self.indices=ti.field(int,self.NE*2)
        self.rest_lens=ti.field(float,self.NE)
        self._create_geometry()

        
    @ti.kernel
    def _create_geometry(self):
        #position
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            self.init_positions[k] = vec(i,0, j) / self.N - vec(0.5,0, 0.5)
        #axis edge
        for i, j in ti.ndrange(self.N + 1, self.N):
            idx, idx1 = i * self.N + j, i * (self.N + 1) + j
            self.edges[idx] = pairi(idx1,idx1 + 1)
            self.rest_lens[idx] = (self.init_positions[idx1] - self.init_positions[idx1 + 1]).norm()
        start = self.N * (self.N + 1)
        for i, j in ti.ndrange(self.N, self.N + 1):
            idx, idx1, idx2 = (
                start + i + j * self.N,
                i * (self.N + 1) + j,
                i * (self.N + 1) + j + self.N + 1,
            )
            self.edges[idx] = pairi(idx1,idx2)
            self.rest_lens[idx] = (self.init_positions[idx1] - self.init_positions[idx2]).norm()
        #diagnal edge
        start = 2 * self.N * (self.N + 1)
        for i, j in ti.ndrange(self.N, self.N):
            idx, idx1, idx2 = (
                start + i * self.N + j,
                i * (self.N + 1) + j,
                (i + 1) * (self.N + 1) + j + 1,
            )
            self.edges[idx] = pairi(idx1,idx2)
            self.rest_lens[idx] = (self.init_positions[idx1] - self.init_positions[idx2]).norm()
        start = 2 * self.N * (self.N + 1) + self.N * self.N
        for i, j in ti.ndrange(self.N, self.N):
            idx, idx1, idx2 = (
                start + i * self.N + j,
                i * (self.N + 1) + j + 1,
                (i + 1) * (self.N + 1) + j,
            )
            self.edges[idx] = pairi(idx1,idx2)
            self.rest_lens[idx] = (self.init_positions[idx1] - self.init_positions[idx2]).norm()
        #index
        for i in self.edges:
            self.indices[2 * i + 0] = self.edges[i][0]
            self.indices[2 * i + 1] = self.edges[i][1]


    
    def update_M(self):
        self._create_M(self.M_builder)
        self.M=self.M_builder.build()
    
    @ti.kernel
    def _create_M(self,builder:ti.types.sparse_matrix_builder()):
        for V in self.masses:
            for d in ti.static(range(dim)):
                builder[V*dim+d,V*dim+d]+=self.masses[V]
                
    @ti.kernel
    def update_gradiant(self):
        self.gradiants.fill(ti.Vector.zero(float,dim))
        for E in range(self.NE):
            k=self.elasticities[E]
            edge=self.edges[E]
            edge_vec=self.get_edge_vec(self.temp_positions,E)
            norm=(edge_vec.norm()-self.rest_lens[E])*k
            direction=edge_vec.normalized()
            gradiant=direction*norm
            self.gradiants[edge[0]]-=gradiant
            self.gradiants[edge[1]]+=gradiant

    @ti.kernel
    def update_hessions(self):
        for E in self.edges:
            edge_vec=self.get_edge_vec(self.temp_positions,E)
            len=edge_vec.norm()
            rest_len=self.rest_lens[E]
            k=self.elasticities[E]
            outer_product=edge_vec.outer_product(edge_vec)
            self.hessions[E]=k*(outer_product/(len)**2+rest_len/len*(ti.Matrix.identity(float,dim)-outer_product/len**2))
    

    def update_H(self):
        self._update_H(self.H_builder)
        self.H=self.H_builder.build()
    @ti.kernel
    def _update_H(self,builder:ti.types.sparse_matrix_builder()):
        for E in self.edges:
            edge=self.edges[E]
            hession=self.hessions[E]
            for n,m in ti.static(ti.ndrange(dim,dim)):
                builder[edge[0]*dim+n,edge[0]*dim+m]+=hession[n,m]
                builder[edge[1]*dim+n,edge[1]*dim+m]+=hession[n,m]
                builder[edge[0]*dim+n,edge[1]*dim+m]-=hession[n,m]
                builder[edge[1]*dim+n,edge[0]*dim+m]-=hession[n,m]
    

    @ti.kernel
    def apply_velocity(self,dt:float):
        for V in range(self.NV):
            if self.mask[V]:
                self.positions[V]+=self.velocities[V]*dt

    @ti.kernel
    def apply_temp_position(self):
        for V in range(self.NV):
            if self.mask[V]:
                self.positions[V]=self.temp_positions[V]

    @ti.kernel
    def update_velocity(self,dt:float):
        for V in range(self.NV):
            if self.mask[V]:
                self.velocities[V]+=(self.temp_positions[V]-self.positions[V])/dt

    def update(self, dt:float):
        self.apply_velocity(dt)
        self.temp_positions.copy_from(self.positions)

        self.solver.begin_step(dt)
        for _ in range(self.step):
            if 'gradiants' in self.requires:
                self.update_gradiant()
            if 'hessions' in self.requires:
                self.update_hessions()
            if 'H' in self.requires:
                self.update_H()
            self.solver.temp_step()
        self.solver.end_step()

        self.update_velocity(dt)
        self.apply_temp_position()



    def displayGGUI(self, scene:ti.ui.Scene, radius=0.02, color=(1.0, 1.0, 1.0)):
        scene.lines(self.positions, width=1, indices=self.indices, color=(1, 1, 1))
        scene.particles(self.positions, radius, color)
    @ti.func
    def get_edge_vec(self,positions:ti.template(),E:int):
        return positions[self.edges[E][1]]-positions[self.edges[E][0]]


def main():
    ti.init(arch=ti.gpu,default_fp=ti.f32)

    time=0
    dt = 0.2
    pause = False
    cloth = Simulator(N=19,k=8,solver=DiagnalHessionSolver())
    cloth.mask[0]=False
    rest_position=cloth.init_positions[0]
    cloth.masses[0]=1024
    cloth.update_M()

    #ui
    window = ti.ui.Window("Implicit Mass Spring System", res=(500, 500))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.up(up.x, up.y, up.z)
    camera.lookat(0, 0, 0)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)
    canvas = window.get_canvas()
    
    camera_position=vec(0,-5,0)
    move_a=False
    move_d=False
    move_w=False
    move_s=False

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            if window.event.key == 'a':
                move_a=True
            if window.event.key == 'd':
                move_d=True
            if window.event.key == 'w':
                move_w=True
            if window.event.key == 's':
                move_s=True
        if window.get_event(ti.ui.RELEASE):
            if window.event.key == 'a':
                move_a=False
            if window.event.key == 'd':
                move_d=False
            if window.event.key == 'w':
                move_w=False
            if window.event.key == 's':
                move_s=False

        if move_a:
            camera_position-=camera_position.cross(up).normalized()*dt
        if move_d:
            camera_position+=camera_position.cross(up).normalized()*dt
        if move_w:
            camera_position-=camera_position.normalized()*dt
        if move_s:
            camera_position+=camera_position.normalized()*dt

        if window.is_pressed(ti.ui.SPACE):
            pause = not pause

        if not pause:
            cloth.positions[0]=rest_position+vec(0,tm.sin(time/10),tm.cos(time/10))*0.2*min(time/6,1)
            cloth.update(dt)
            time+=dt

        camera.position(camera_position.x,camera_position.y,camera_position.z)
        scene.set_camera(camera)
        scene.point_light(pos=camera_position, color=(1, 1, 1))
        cloth.displayGGUI(scene)
        canvas.scene(scene)
        

        window.show()


if __name__ == "__main__":
    main()