from numpy import indices
from taichi.lang.struct import Struct
from taichi.linalg import SparseMatrixBuilder
from TaichiLib import *
from . import energy
from . import solver 
from . import collision_handler
from . import constraint
from . import geometry

@ti.data_oriented
class Simulator:
    require_dependencies={
        'M':[],
        'gradiants':[],
        'hession':[],
        'H':['hession'],
    }
    def __init__(self,bound:Bound,solver:'solver.Solver',geometry:'geometry.Geometry') -> None:
        self.bound=bound
        self.max_displace=1/(2/19)/4
        self.max_radius=2/(2/19)
        self.geometry=geometry

        #attribute
        self.M_builder=SparseMatrixBuilder(dim * self.geometry.num_point, dim * self.geometry.num_point, max_num_triplets=self.geometry.num_point*dim)
        self.update_M()

        self.mask=ti.field(bool,self.geometry.num_point)
        self.mask.fill(True)

        #state
        self.positions=ti.field(vec,geometry.num_point)
        self.positions.copy_from(self.geometry.positions)
        self.velocities=ti.field(vec,geometry.num_point)
        self.velocities.fill(vec(0))
        
        #simulate
        self.energies=list['energy.Energy']()
        self.prev_positions=ti.field(vec,self.geometry.num_point)
        self.constrainted_positions=ti.field(vec,self.geometry.num_point)
        self.gradiants=ti.field(vec,self.geometry.num_point)
        self.hession=ti.field(mat)
        self._hession_sparse=root.pointer(ti.ij,(self.geometry.num_point,self.geometry.num_point)).place(self.hession)
        self.H_builder=SparseMatrixBuilder(dim * self.geometry.num_point, dim * self.geometry.num_point, max_num_triplets=(self.geometry.num_point*dim)**2) 
        self.b=ti.ndarray(float,dim*self.geometry.num_point)
        self.step=4

        #solver
        self.solver=solver
        self.solver.fit(self)
        self.solve_requrie_dependency(solver.get_requires())

        #collision
        self.collision_handler=collision_handler.CollisionHandler(0,self)
        
    def solve_requrie_dependency(self,requires:list[str]):
        self.requires=requires
        for require in requires:
            dependencies=self.require_dependencies[require]
            for dependency in dependencies:
                if not dependency in self.requires:
                    self.requires.append(dependency)

    def create_geometry(self,N:int):
        self.positions=ti.field(vec,self.geometry.num_point)
        self.positions.copy_from(self.geometry.positions)
        
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
        for V in self.geometry.masses:
            for d in ti.static(range(dim)):
                builder[V*dim+d,V*dim+d]+=self.geometry.masses[V]
                
    @ti.kernel
    def update_gradiant(self):
        self.gradiants.fill(ti.Vector.zero(float,dim))
        for E in range(self.NE):
            k=self.elasticities[E]
            edge=self.edges[E]
            edge_vec=self.get_edge_vec(self.constrainted_positions,E)
            norm=(edge_vec.norm()-self.rest_lens[E])*k
            direction=edge_vec.normalized()
            gradiant=direction*norm
            self.gradiants[edge[0]]-=gradiant
            self.gradiants[edge[1]]+=gradiant

    @ti.kernel
    def update_string_hessions(self):
        for E in self.edges:
            edge_vec=self.get_edge_vec(self.constrainted_positions,E)
            len=edge_vec.norm()
            rest_len=self.rest_lens[E]
            k=self.elasticities[E]
            outer_product=edge_vec.outer_product(edge_vec)
            self.string_hessions[E]=k*(outer_product/(len)**2+rest_len/len*(ti.Matrix.identity(float,dim)-outer_product/len**2))

    def update_hession(self):
        self._hession_sparse.deactivate_all()
        for energy in self.energies:
            energy.update_hession()
        
    @ti.kernel
    def _update_hession(self):
        for E in self.edges:
            edge=self.edges[E]
            string_hession=self.string_hessions[E]
            self.hession[edge[0],edge[0]]+=string_hession
            self.hession[edge[1],edge[1]]+=string_hession
            self.hession[edge[0],edge[1]]-=string_hession
            self.hession[edge[1],edge[0]]-=string_hession
    

    def update_H(self):
        self._update_H(self.H_builder)
        self.H=self.H_builder.build()
    @ti.kernel
    def _update_H(self,builder:ti.types.sparse_matrix_builder()):
        for i,j in self._hession_sparse:
            for n,m in ti.static(ti.ndrange(dim,dim)):
                builder[i*dim+n,j*dim+j]+=self.hession[i,j][n,m]

    @ti.kernel
    def apply_velocity(self,dt:float):
        for V in range(self.geometry.num_point):
            if self.mask[V]:
                self.positions[V]+=self.velocities[V]*dt

    @ti.kernel
    def apply_temp_position(self):
        for V in range(self.geometry.num_point):
            if self.mask[V]:
                self.positions[V]=self.constrainted_positions[V]

    @ti.kernel
    def update_velocity(self,dt:float):
        for V in range(self.geometry.num_point):
            if self.mask[V]:
                self.velocities[V]+=(self.constrainted_positions[V]-self.positions[V])/dt

    def update(self, dt:float):
        self.prev_positions.copy_from(self.positions)
        self.apply_velocity(dt)
        self.constrainted_positions.copy_from(self.positions)

        self.solver.begin_step(dt)
        for _ in range(self.step):
            if 'gradiants' in self.requires:
                self.gradiants.fill(vec(0))
                for energy in self.energies:
                    energy.simulator=self
                    energy.update_gradiants()
            if 'hession' in self.requires:
                self.update_hession()
            if 'H' in self.requires:
                self.update_H()
            self.solver.temp_step()
        self.solver.end_step()



        #self.collision_handler.update()

        
        #self.collision_handler.step()

        self.update_velocity(dt)
        self.apply_temp_position()
     
    @ti.func
    def get_edge_vec(self,positions:ti.template(),E:int):
        return self.positions[self.edges[E][1]]-positions[self.edges[E][0]]
    @ti.func
    def get_edge_segment(self,positions:ti.template(),E:int) -> Segment:
        return Segment(self.positions[self.edges[E][0]],self.positions[self.edges[E][1]])
    
    
