from TaichiLib import *
from abc import ABC, abstractmethod
from taichi.linalg import SparseSolver
from . import simulator

class Solver(ABC):
    def fit(self,simulator:'simulator.Simulator'):
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
                b[dim*V+d]=-mass*(self.simulator.constrainted_positions[V][d]-self.simulator.positions[V][d])/dt-self.simulator.gradiants[V][d]

    @ti.kernel
    def update_temp_position(self,d_position:tt.ndarray()):
        for V in range(self.simulator.NV):
            if self.simulator.mask[V]:
                for d in ti.static(range(dim)):
                    self.simulator.constrainted_positions[V][d]+=d_position[dim*V+d]

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
        return ['hession','gradiants']
    def temp_step(self):
        self.update_diag_hession()
        self.update_temp_position()
    #
    @ti.kernel
    def update_diag_hession(self):
        for V in range(self.simulator.NV):
            for d in ti.static(range(dim)):
                self.diag_hession[V][d]=self.simulator.hession[V,V][d,d]

    @ti.kernel
    def update_temp_position(self):
        for V in range(self.simulator.NV):
            if self.simulator.mask[V]:
                mass=self.simulator.masses[V]
                temp_position=self.simulator.constrainted_positions[V]
                self.simulator.constrainted_positions[V]+=(-mass*(temp_position-self.simulator.positions[V])/self.dt**2-self.simulator.gradiants[V])/(mass/self.dt**2+self.diag_hession[V])