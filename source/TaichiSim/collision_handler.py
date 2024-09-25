from cgitb import handler
from TaichiLib import *
from TaichiLib.linq import Linq
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
    def on_query_update(self,idx_center:int,idx_other:int,distance:float):pass

@ti.data_oriented
class MaxDisplace(CollisionAction,Constraint):
    def __init__(self, handler: 'CollisionHandler') -> None:
        CollisionAction.__init__(self,handler)
        self.simulator=handler.simulator
        self.max_displaces=ti.field(float,self.simulator.geometry.num_edge)
        self.loss=ti.field(int,())

        self.t=ti.field(int,self.simulator.geometry.num_edge)
        self.t.fill(0)
    def begin_update(self):
        self.max_displaces.fill(self.simulator.max_displace)
    @ti.func
    def on_query_update(self,idx_center:int,idx_other:int,distance:float):
        simulator=self.handler.simulator
        edge_center=simulator.geometry.edges[idx_center]
        displace=tm.max(0,distance-self.handler.min_distance)/2
        ti.atomic_min(self.max_displaces[edge_center.x],displace)
        ti.atomic_min(self.max_displaces[edge_center.y],displace)
    def step(self,update_loss:bool):
        self._step(update_loss)
    @ti.kernel
    def _step(self,update_loss:ti.template()):
        st_update_loss=ti.static(update_loss)
        if st_update_loss:
            self.loss[None]=0
        for V in range(self.simulator.geometry.num_point):
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
        self.determinants=ti.field(float,self.handler.simulator.geometry.num_edge)
    @ti.func
    def on_query_update(self,idx_center:int,idx_other:int,distance:float):
        simulator=self.handler.simulator
        segment_x=simulator.get_edge_segment(simulator.prev_positions,idx_center)
        segment_y=simulator.get_edge_segment(simulator.prev_positions,idx_other)
        pivot=segment_x.x
        vectors=(segment_x.y-pivot,segment_y.x-pivot,segment_y.y-pivot)
        tensor=ti.Matrix.cols([vectors[0],vectors[1],vectors[2]])
        self.determinants[idx_center]=tensor.determinant()
    @ti.func
    def on_query_step(self,idx_center:int,idx_other:int,distance:float):
        simulator=self.handler.simulator
        delta_distance=self.handler.repel_distance-distance
        if delta_distance>0:
            edge_center=simulator.geometry.edges[idx_center]
            vector_x=simulator.get_edge_vec(simulator.constrainted_positions,idx_center)
            vector_y=simulator.get_edge_vec(simulator.constrainted_positions,idx_other)
            determinant=self.determinants[idx_center]
            direction=tm.cross(vector_x,vector_y)
            simulator.constrainted_positions[edge_center.x]+=direction*delta_distance*tm.sign(determinant)
            simulator.constrainted_positions[edge_center.y]-=direction*delta_distance*tm.sign(determinant)
            

@ti.data_oriented
class CollisionHandler:
    def __init__(self,repel_distance:float,min_distance:float,simulator:'simulator.Simulator') -> None:
        self.repel_distance=repel_distance
        self.min_distance=min_distance
        self.simulator=simulator
        self.unit=2*(simulator.max_radius+simulator.max_displace+self.repel_distance/2)*tm.sqrt(2)
        self.points_query=spatial_query.Grid(self.simulator.geometry.num_edge)
        self.max_displace_constraint=MaxDisplace(self)
        self.max_displace_constraint.set_tightness(0.9)
        self.max_edge_length_constraint=MaxLength(self.simulator.max_radius,self.simulator.geometry.edges,self.simulator.constrainted_positions,self.simulator.geometry.masses)
        self.max_edge_length_constraint.set_tightness(0.9)
        self.min_segment_length_constraint=MinSegmentDistance(self)
    @ti.func
    def vec_into_space(self,point:vec)->vec:
        return (point-self.simulator.bound.min)/self.unit
    @ti.func
    def bound_into_space(self,bound:Bound)->Bound:
        return Bound(self.vec_into_space(bound.min),self.vec_into_space(bound.max))
    @ti.kernel
    def append_segments(self):
        for i in self.simulator.geometry.edges:
            segment=self.simulator.get_edge_segment(self.simulator.prev_positions,i)
            bound=self.bound_into_space(segment.get_bound_sphere().get_extended(self.simulator.max_displace+self.min_distance/2).get_scaled(tm.sqrt(2)).get_bound_box())
            self.points_query.append(bound,bound.get_center(),i)
    @ti.func
    def on_query_update(self,idx_center:int,idx_other:int):
        simulator=self.simulator
        edge_center=simulator.geometry.edges[idx_center]
        edge_other=simulator.geometry.edges[idx_other]
        if not is_adjacent(edge_center,edge_other):
            segment_x=simulator.get_edge_segment(simulator.prev_positions,idx_center)
            segment_y=simulator.get_edge_segment(simulator.prev_positions,idx_other)
            distance=get_distance_segment(segment_x,segment_y)
            self.max_displace_constraint.on_query_update(idx_center,idx_other,distance)
            self.min_segment_length_constraint.on_query_update(idx_center,idx_other,distance)
    @ti.func
    def on_query_step(self,idx_center:int,idx_other:int):
        simulator=self.simulator
        edge_center=simulator.geometry.edges[idx_center]
        edge_other=simulator.geometry.edges[idx_other]
        if not is_adjacent(edge_center,edge_other):
            segment_x=simulator.get_edge_segment(simulator.prev_positions,idx_center)
            segment_y=simulator.get_edge_segment(simulator.prev_positions,idx_other)
            distance=get_distance_segment(segment_x,segment_y)
            self.min_segment_length_constraint.on_query_step(idx_center,idx_other,distance)
    
    def update(self):
        self.max_displace_constraint.begin_update()
        self.points_query.clear()
        self.append_segments()
        self.points_query.update(self.on_query_update)
    def step(self):
        self.points_query.update(self.on_query_step)
        update_constraints([self.max_displace_constraint],2,0)