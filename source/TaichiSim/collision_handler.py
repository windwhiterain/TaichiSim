from TaichiLib import *
from . import spatial_query
from . import simulator
from .constraint import Constraint, MaxLength, update_constraints
 
from abc import ABC,abstractmethod

class CollisionAction:
    def __init__(self,handler:'CollisionHandler') -> None:
        self.handler=handler
    @abstractmethod
    def begin_update(self):pass
    @abstractmethod
    def on_query(self,idx_center:int,idx_other:int):pass

@ti.data_oriented
class MaxDisplace(CollisionAction,Constraint):
    def __init__(self, handler: 'CollisionHandler') -> None:
        CollisionAction.__init__(self,handler)
        self.simulator=handler.simulator
        self.max_displaces=ti.field(float,self.simulator.NV)
        self.loss=ti.field(int,())
    def begin_update(self):
        self.max_displaces.fill(self.simulator.max_displace)
    @ti.func
    def on_query(self,idx_center:int,idx_other:int):
        simulator=self.handler.simulator
        edge_center=simulator.edges[idx_center]
        edge_other=simulator.edges[idx_other]
        if not is_adjacent(edge_center,edge_other):
            segment_x=simulator.get_edge_segment(simulator.prev_positions,idx_center)
            segment_y=simulator.get_edge_segment(simulator.prev_positions,idx_other)
            distance=get_distance_segment(segment_x,segment_y)
            displace=tm.max(0,distance-self.handler.min_distance)/2
            # if displace<self.simulator.max_displace:
            #     print(edge_center,edge_other)
            #     print(segment_x.x,segment_x.y,segment_y.x,segment_y.y)
            self.max_displaces[edge_center.x]=tm.min(self.max_displaces[edge_center.x],displace)
            self.max_displaces[edge_center.y]=tm.min(self.max_displaces[edge_center.y],displace)
    def step(self,update_loss:bool):
        self._step(update_loss)
    @ti.kernel
    def _step(self,update_loss:ti.template()):
        st_update_loss=ti.static(update_loss)
        if st_update_loss:
            self.loss[None]=0
        for V in range(self.simulator.NV):
            target_position=self.simulator.constrainted_positions[V]
            prev_position=self.simulator.prev_positions[V]
            target_length=(target_position-prev_position).norm()
            max_length=tm.min(self.max_displaces[V],target_length)
            if target_length>max_length:
                if st_update_loss:
                    self.loss[None]=tm.max(self.loss[None],1)
                else:
                    direction=(target_position-prev_position).normalized()
                    self.simulator.constrainted_positions[V]=prev_position+max_length*self.tightness*direction
    def get_loss(self):
        return self.loss[None]

@ti.data_oriented
class MinSegmentDistance(CollisionAction):
    def __init__(self, handler: 'CollisionHandler') -> None:
        super().__init__(handler)
    @ti.func
    def on_query(self, idx_center: int, idx_other: int):
        handler=self.handler
        simulator=self.handler.simulator
        edge_center=simulator.edges[idx_center]
        edge_other=simulator.edges[idx_other]

@ti.data_oriented
class CollisionHandler:
    def __init__(self,min_distance:float,simulator:'simulator.Simulator') -> None:
        self.min_distance=min_distance
        self.simulator=simulator
        self.unit=2*(simulator.max_radius+simulator.max_displace+self.min_distance/2)*tm.sqrt(2)
        self.points_query=spatial_query.Grid(self.simulator.NE)
        self.max_displace_constraint=MaxDisplace(self)
        self.max_displace_constraint.set_tightness(0.9)
        self.max_edge_length_constraint=MaxLength(self.simulator.max_radius,self.simulator.edges,self.simulator.constrainted_positions,self.simulator.masses)
        self.max_edge_length_constraint.set_tightness(0.9)
    @ti.func
    def vec_into_space(self,point:vec)->vec:
        return (point-self.simulator.bound.min)/self.unit
    @ti.func
    def bound_into_space(self,bound:Bound)->Bound:
        return Bound(self.vec_into_space(bound.min),self.vec_into_space(bound.max))
    @ti.kernel
    def append_segments(self):
        for i in self.simulator.edges:
            segment=self.simulator.get_edge_segment(self.simulator.prev_positions,i)
            bound=self.bound_into_space(segment.get_bound_sphere().get_extended(self.simulator.max_displace+self.min_distance/2).get_scaled(tm.sqrt(2)).get_bound_box())
            self.points_query.append(bound,bound.get_center(),i)
    @ti.func
    def on_query(self,idx_center:int,idx_other:int):
        self.max_displace_constraint.on_query(idx_center,idx_other)
    
    def update(self):
        self.max_displace_constraint.begin_update()
        self.points_query.clear()
        self.append_segments()
        self.points_query.update(self.max_displace_constraint)
        #print(self.max_displace_constraint.max_displaces)
    def step(self):
        update_constraints([self.max_displace_constraint],2,0)