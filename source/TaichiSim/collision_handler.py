from cgitb import handler
from TaichiLib import *
from TaichiLib.linq import Linq
from . import spatial_query
from . import simulator
from .constraint import Constraint, MaxLength
 
from abc import ABC,abstractmethod

class CollisionAction:
    def __init__(self,handler:'CollisionHandler') -> None:
        self.handler=handler
    @abstractmethod
    def begin_update(self):pass
    @abstractmethod
    def on_query_update(self,idx_center:int,idx_other:int,distance:float):pass
    @abstractmethod
    def on_query_step(self,update_loss:bool,idx_center:int,idx_other:int,distance:float):pass

@ti.data_oriented
class Ground(Constraint):
    def __init__(self, scale: float,height:float) -> None:
        super().__init__(scale)
        self.height=height
        self.loss=ti.field(int,())
    def step(self,update_loss:bool):
        self._step(update_loss)
    @ti.kernel
    def _step(self,update_loss:ti.template()):
        UPDATE_LOSS=ti.static(update_loss)
        simulator=self.simulator
        geommetry=simulator.geometry
        if UPDATE_LOSS:
            self.loss[None]=0
        for p in simulator.constrainted_positions:
            positon=simulator.constrainted_positions[p]
            if positon.z<self.height:
                simulator.delta_positions[p]+=vec(0,0,self.height-positon.z)
                if UPDATE_LOSS:
                    self.loss[None]=tm.max(self.loss[None],1)
    def get_loss(self) -> float:
        return self.loss[None]

    

@ti.data_oriented
class MaxDisplace(CollisionAction,Constraint):
    def __init__(self, handler: 'CollisionHandler',scale:float,max_displace:float) -> None:
        CollisionAction.__init__(self,handler)
        Constraint.__init__(self,scale)
        self.max_displaces=ti.field(float,self.handler.simulator.geometry.num_edge)
        self.loss=ti.field(int,())
        self.max_displace=max_displace
    def begin_update(self):
        self.max_displaces.fill(self.max_displace)
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
        simulator=self.handler.simulator
        geometry=simulator.geometry
        UPDATE_LOSS=ti.static(update_loss)
        if UPDATE_LOSS:
            self.loss[None]=0
        for p in range(geometry.num_point):
            target_position=simulator.constrainted_positions[p]
            prev_position=simulator.prev_positions[p]
            target_length=(target_position-prev_position).norm()
            max_length=tm.max(self.max_displaces[p],0)
            if target_length>0 and target_length>max_length:
                direction=(prev_position-target_position).normalized()
                simulator.delta_positions[p]+=direction*ti.max(target_length-max_length/self.scale,0)
                if UPDATE_LOSS:
                    self.loss[None]=tm.max(self.loss[None],1)
                    
    def get_loss(self):
        return self.loss[None]

@ti.data_oriented
class MinSegmentDistance(CollisionAction):
    def __init__(self, handler: 'CollisionHandler',scale:float) -> None:
        super().__init__(handler)
        self.scale=scale
        self.loss=ti.field(float,())
    @ti.func
    def get_segment_normal(self,x:Segment,y:Segment)->vec:
        normal=tm.cross(x.get_vector(),y.get_vector())
        if normal.norm()<epsilon:
            tangent=tm.cross(x.x-y.x,y.get_vector())
            normal=tm.cross(tangent,y.get_vector())
        if normal.norm()<epsilon:
            normal=vec(0)
        else:
            normal=normal.normalized()
        project_x,project_y=tm.dot(x.x,normal),tm.dot(y.x,normal)
        return normal if project_y>project_x else -normal
    @ti.func
    def on_query_step(self,update_loss:ti.template(),idx_center:int,idx_other:int,distance:float):
        UPDATE_LOSS=ti.static(update_loss)
        simulator=self.handler.simulator
        geometry=simulator.geometry
        delta_distance=self.handler.repel_distance-distance
        if delta_distance>0:
            if UPDATE_LOSS:
                self.loss[None]=tm.max(self.loss[None],delta_distance)
            edge_center=geometry.edges[idx_center]
            segment_center=simulator.get_edge_segment(simulator.prev_positions,idx_center)
            segment_other=simulator.get_edge_segment(simulator.prev_positions,idx_other)
            prev_normal=self.get_segment_normal(segment_center,segment_other)
            segment_center=simulator.get_edge_segment(simulator.constrainted_positions,idx_center)
            segment_other=simulator.get_edge_segment(simulator.constrainted_positions,idx_other)
            normal=self.get_segment_normal(segment_center,segment_other)
            direction=tm.sign(tm.dot(prev_normal,normal))
            delta_distance=self.handler.repel_distance*self.scale-distance
            displace=normal*delta_distance*direction
            ti.atomic_add(simulator.delta_positions[edge_center.x],-displace)
            ti.atomic_add(simulator.delta_positions[edge_center.y],-displace)
    def get_loss(self)->float:
        return self.loss[None]
            

@ti.data_oriented
class CollisionHandler(Constraint):
    def __init__(self,repel_distance:float,min_distance:float,simulator:'simulator.Simulator') -> None:
        self.repel_distance=repel_distance
        self.min_distance=min_distance
        self.simulator=simulator
        self.unit=2*(simulator.max_radius+simulator.max_displace+self.repel_distance/2)*tm.sqrt(2)
        self.points_query=spatial_query.Grid(self.simulator.geometry.num_edge)
        self.max_displace_constraint=MaxDisplace(self,1.1,simulator.max_displace)
        self.max_edge_length_constraint=MaxLength(1.1,self.simulator.max_radius*2)
        self.min_segment_length_constraint=MinSegmentDistance(self,1.1)
        self.ground_constraint=Ground(1,-1.1)
        self.on_query_steps={True:self.get_on_query_step(True),False:self.get_on_query_step(False)}
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
            #self.min_segment_length_constraint.on_query_update(idx_center,idx_other,distance)
    @ti.func
    def on_query_step(self,update_loss:ti.template(),idx_center:int,idx_other:int):
        simulator=self.simulator
        edge_center=simulator.geometry.edges[idx_center]
        edge_other=simulator.geometry.edges[idx_other]
        if not is_adjacent(edge_center,edge_other):
            segment_x=simulator.get_edge_segment(simulator.prev_positions,idx_center)
            segment_y=simulator.get_edge_segment(simulator.prev_positions,idx_other)
            distance=get_distance_segment(segment_x,segment_y)
            self.min_segment_length_constraint.on_query_step(update_loss,idx_center,idx_other,distance)

    def get_on_query_step(self,update_loss:bool)->Callable[[int,int],None]:
        @ti.func
        def _(idx_center:int,idx_other:int) -> None:
            self.on_query_step(update_loss,idx_center,idx_other)
        return _
        
    def update(self):
        self.max_displace_constraint.begin_update()
        self.points_query.clear()
        self.append_segments()
        self.points_query.update()
        self.points_query.tranverse(self.on_query_update)
    def step(self,update_loss:bool):
        self.points_query.tranverse(self.on_query_steps[update_loss])
        self.max_displace_constraint.step(update_loss)
    def get_loss(self):
        return tm.max(self.min_segment_length_constraint.get_loss(),self.max_displace_constraint.get_loss())
        