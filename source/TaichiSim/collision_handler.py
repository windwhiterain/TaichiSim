from TaichiLib import *
from . import spatial_query
from . import simulator

@ti.data_oriented
class CollisionHandler:
    def __init__(self,unit:float,simulator:'simulator.Simulator') -> None:
        self.simulator=simulator
        self.unit=unit
        self.points_query=spatial_query.Grid(self.simulator.NV,self)
        self.max_displace_lengths=ti.field(float,simulator.NV)
        self.max_displace_lengths.fill(self.simulator.max_displace_length)
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
            bound=self.bound_into_space(segment.get_bound()).get_extended(self.simulator.max_displace_length+self.simulator.safe_distance/2)
            self.points_query.append(bound,bound.get_center(),i)
    @ti.func
    def on_query(self,idx_center:int,idx_other:int):
        edge_center=self.simulator.edges[idx_center]
        edge_other=self.simulator.edges[idx_other]
        if not is_adjacent(edge_center,edge_other):
            distance=get_distance_segment(self.simulator.get_edge_segment(self.simulator.prev_positions,idx_center),self.simulator.get_edge_segment(self.simulator.prev_positions,idx_other))
            displace=tm.max(0,distance-self.simulator.safe_distance)/2
            
            self.max_displace_lengths[edge_center.x]=tm.min(self.max_displace_lengths[edge_center.x],displace)
            self.max_displace_lengths[edge_center.y]=tm.min(self.max_displace_lengths[edge_center.y],displace)
    @ti.kernel
    def apply_max_displaces(self):
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
        self.max_displace_lengths.fill(self.simulator.max_displace_length)
        self.append_segments()
        self.points_query.update()
    def step(self):
        self.apply_max_displaces()