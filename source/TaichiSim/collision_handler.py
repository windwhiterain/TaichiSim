from TaichiLib import *
from . import spatial_query
from . import simulator
 
from abc import ABC,abstractmethod

class CollisionAction:
    def __init__(self,handler:'CollisionHandler') -> None:
        self.handler=handler
    @abstractmethod
    def on_query(self,idx_center:int,idx_other:int):pass

@ti.data_oriented
class MaxDisplace(CollisionAction):
    def __init__(self, handler: 'CollisionHandler') -> None:
        super().__init__(handler)
    @ti.func
    def on_query(self,idx_center:int,idx_other:int):
        handler=self.handler
        simulator=self.handler.simulator
        edge_center=simulator.edges[idx_center]
        edge_other=simulator.edges[idx_other]
        if not is_adjacent(edge_center,edge_other):
            distance=get_distance_segment(simulator.get_edge_segment(simulator.prev_positions,idx_center),simulator.get_edge_segment(simulator.prev_positions,idx_other))
            displace=tm.max(0,distance-self.handler.min_distance)/2
            handler.max_displace_lengths[edge_center.x]=tm.min(handler.max_displace_lengths[edge_center.x],displace)
            handler.max_displace_lengths[edge_center.y]=tm.min(handler.max_displace_lengths[edge_center.y],displace)

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
        self.unit=2*(simulator.max_radius+simulator.max_displace_length+self.min_distance/2)*tm.sqrt(2)
        self.points_query=spatial_query.Grid(self.simulator.NV)
        self.max_displace_lengths=ti.field(float,simulator.NV)
        self.max_displace_lengths.fill(self.simulator.max_displace_length)
        self.collision_action=MaxDisplace(self)
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
            bound=self.bound_into_space(segment.get_bound_sphere().get_extended(self.simulator.max_displace_length+self.min_distance/2).get_scaled(tm.sqrt(2)).get_bound_box())
            self.points_query.append(bound,bound.get_center(),i)
    @ti.func
    def on_query(self,idx_center:int,idx_other:int):
        self.collision_action.on_query(idx_center,idx_other)
    @ti.kernel
    def apply_max_displace_lengths(self):
        for V in range(self.simulator.NV):
            target_position=self.simulator.constrainted_positions[V]
            prev_position=self.simulator.prev_positions[V]
            target_length=(target_position-prev_position).norm()
            max_length=tm.min(self.max_displace_lengths[V],target_length)
            if target_length>max_length:
                direction=(target_position-prev_position).normalized()
                self.simulator.constrainted_positions[V]=prev_position+max_length*direction
    def update(self):
        self.points_query.clear()
        self.append_segments()
        self.max_displace_lengths.fill(self.simulator.max_displace_length)
        self.points_query.update(self.collision_action)
    def step(self):
        self.apply_max_displace_lengths()