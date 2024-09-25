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
        self.max_radius=4*(2/19)
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
        self.collision_handler=collision_handler.CollisionHandler((2/19)*0.3,(2/19)*0.1,self)
        
    def solve_requrie_dependency(self,requires:list[str]):
        self.requires=requires
        for require in requires:
            dependencies=self.require_dependencies[require]
            for dependency in dependencies:
                if not dependency in self.requires:
                    self.requires.append(dependency)
        
    def update_M(self):
        self._create_M(self.M_builder)
        self.M=self.M_builder.build()
    
    @ti.kernel
    def _create_M(self,builder:ti.types.sparse_matrix_builder()):
        for V in self.geometry.masses:
            for d in ti.static(range(dim)):
                builder[V*dim+d,V*dim+d]+=self.geometry.masses[V]
                
    def update_gradiant(self):
        self.gradiants.fill(vec(0))
        for energy in self.energies:
            energy.simulator=self
            energy.update_gradiants()

    def update_hession(self):
        self._hession_sparse.deactivate_all()
        for energy in self.energies:
            energy.update_hession()

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
                self.update_gradiant()
            if 'hession' in self.requires:
                self.update_hession()
            if 'H' in self.requires:
                self.update_H()
            self.solver.temp_step()
        self.solver.end_step()



        self.collision_handler.update()

        
        self.collision_handler.step()

        self.update_velocity(dt)
        self.apply_temp_position()
     
    @ti.func
    def get_edge_vec(self,positions:ti.template(),E:int)->vec:
        return self.positions[self.geometry.edges[E][1]]-positions[self.geometry.edges[E][0]]
    @ti.func
    def get_edge_segment(self,positions:ti.template(),E:int) -> Segment:
        return Segment(self.positions[self.geometry.edges[E][0]],self.positions[self.geometry.edges[E][1]])
    
    
