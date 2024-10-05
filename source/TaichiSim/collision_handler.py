from cgitb import handler
from math import sqrt
from TaichiLib import *
from TaichiLib.linq import Linq
from . import energy
from . import spatial_query
from . import simulator
from .constraint import Constraint, MaxLength


class CollisionAction:
    def __init__(self,handler:'CollisionHandler') -> None:
        self.handler=handler
    def begin_update(self):pass
    def on_query_update(self,idx_center:int,idx_other:int,vector:vec,parameter:pair):pass
    def on_query_step(self,update_loss:bool,idx_center:int,idx_other:int,vector:vec,parameter:pair):pass

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
                simulator.constraint_weights[p]+=1
                if UPDATE_LOSS:
                    self.loss[None]=tm.max(self.loss[None],1)
    @ti.kernel
    def update_gradiant_hession(self):
        simulator=self.simulator
        geommetry=simulator.geometry
        for p in range(geommetry.num_point):
            positon=simulator.constrainted_positions[p]
            if positon.z<self.height:
                simulator.gradiant[p]+=-vec(0,0,self.height-positon.z)
                simulator.hession[p,p]+=vec(0,0,1).outer_product(vec(0,0,1))
    def get_loss(self) -> float:
        return self.loss[None]

    

@ti.data_oriented
class MaxDisplace(CollisionAction,Constraint):
    def __init__(self, handler: 'CollisionHandler',scale:float,max_displace:float) -> None:
        CollisionAction.__init__(self,handler)
        Constraint.__init__(self,scale)
        self.prev_positions=ti.field(vec,self.handler.simulator.geometry.num_point)
        self.max_displaces=ti.field(float,self.handler.simulator.geometry.num_point)
        self.loss=ti.field(int,())
        self.max_displace=max_displace
    def begin_update(self):
        self.prev_positions.copy_from(self.simulator.constrainted_positions)
        self.max_displaces.fill(inf)
    @ti.func
    def on_query_update(self,idx_center:int,idx_other:int,vector:vec,parameter:pair):
        simulator=self.handler.simulator
        distance=vector.norm()
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
            prev_position=self.prev_positions[p]
            target_length=(target_position-prev_position).norm()
            max_length=tm.max(self.max_displaces[p],0)
            root_position=simulator.prev_positions[p]
            position=target_position
            if target_length>max_length:
                direction=(target_position-prev_position).normalized()
                position=prev_position+max_length/self.scale*direction
            root_target_length=(position-root_position).norm()
            if root_target_length>self.max_displace:
                direction=(position-root_position).normalized()
                position=root_position+direction*self.max_displace
            if (position!=vec(0)).all():
                simulator.delta_positions[p]+=position-target_position
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
    def on_query_step(self,update_loss:ti.template(),idx_center:int,idx_other:int,vector:vec,parameter:pair):
        UPDATE_LOSS=ti.static(update_loss)
        simulator=self.handler.simulator
        geometry=simulator.geometry
        distance=vector.norm()
        delta_distance=self.handler.repel_distance-distance
        if delta_distance>0:
            if UPDATE_LOSS:
                self.loss[None]=tm.max(self.loss[None],delta_distance)
            edge_center=geometry.edges[idx_center]
            normal=vector.normalized()
            displace=normal*delta_distance
            ti.atomic_add(simulator.delta_positions[edge_center.x],-displace*parameter.x)
            ti.atomic_add(simulator.delta_positions[edge_center.y],-displace*(1-parameter.x))
            ti.atomic_add(simulator.constraint_weights[edge_center.x],1)
            ti.atomic_add(simulator.constraint_weights[edge_center.y],1)
    @ti.func
    def on_query_update_gradiant_hession(self,idx_center:int,idx_other:int,vector:vec,parameter:pair):
        simulator=self.handler.simulator
        geometry=simulator.geometry
        distance=vector.norm()
        delta_distance=self.handler.repel_distance-distance
        if delta_distance>0:
            edge_center=geometry.edges[idx_center]
            normal=vector.normalized()
            gradiant=normal*delta_distance
            hession=normal.outer_product(normal)
            ti.atomic_add(simulator.gradiant[edge_center.x],gradiant*parameter.x)
            ti.atomic_add(simulator.gradiant[edge_center.y],gradiant*(1-parameter.x))
            ti.atomic_add(simulator.hession[edge_center.x,edge_center.x],hession*parameter.x)
            ti.atomic_add(simulator.hession[edge_center.y,edge_center.y],hession*(1-parameter.x))
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
        self.min_segment_distance_constraint=MinSegmentDistance(self,1.1)
        self.ground_constraint=Ground(1,-1.1)
        self.target_positions=ti.field(vec,self.simulator.geometry.num_point)
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
            segment_x=simulator.get_edge_segment(simulator.constrainted_positions,idx_center)
            segment_y=simulator.get_edge_segment(simulator.constrainted_positions,idx_other)
            parameter_vector=get_distance_segment(segment_x,segment_y)
            self.max_displace_constraint.on_query_update(idx_center,idx_other,parameter_vector.vector,parameter_vector.parameter)
            #self.min_segment_length_constraint.on_query_update(idx_center,idx_other,distance)
    @ti.func
    def on_query_step(self,update_loss:ti.template(),idx_center:int,idx_other:int):
        simulator=self.simulator
        edge_center=simulator.geometry.edges[idx_center]
        edge_other=simulator.geometry.edges[idx_other]
        if not is_adjacent(edge_center,edge_other):
            segment_x=simulator.get_edge_segment(simulator.constrainted_positions,idx_center)
            segment_y=simulator.get_edge_segment(simulator.constrainted_positions,idx_other)
            parameter_vector=get_distance_segment(segment_x,segment_y)
            self.min_segment_distance_constraint.on_query_step(update_loss,idx_center,idx_other,parameter_vector.vector,parameter_vector.parameter)

    def get_on_query_step(self,update_loss:bool)->Callable[[int,int],None]:
        @ti.func
        def _(idx_center:int,idx_other:int) -> None:
            self.on_query_step(update_loss,idx_center,idx_other)
        return _
    
    @ti.func
    def on_query_update_gradiant_hession(self,idx_center:int,idx_other:int):
        simulator=self.simulator
        edge_center=simulator.geometry.edges[idx_center]
        edge_other=simulator.geometry.edges[idx_other]
        if not is_adjacent(edge_center,edge_other):
            segment_x=simulator.get_edge_segment(simulator.constrainted_positions,idx_center)
            segment_y=simulator.get_edge_segment(simulator.constrainted_positions,idx_other)
            parameter_vector=get_distance_segment(segment_x,segment_y)
            self.min_segment_distance_constraint.on_query_update_gradiant_hession(idx_center,idx_other,parameter_vector.vector,parameter_vector.parameter)
        
    def update(self):
        self.target_positions.copy_from(self.simulator.constrainted_positions)
        self.simulator.constrainted_positions.copy_from(self.simulator.prev_positions)

        self.ground_constraint.simulator=self.simulator
        self.max_displace_constraint.simulator=self.simulator
        self.max_displace_constraint.begin_update()
        self.points_query.clear()
        self.append_segments()
        self.points_query.update()

    def step(self,update_loss:bool):
        for i in range(2):
            self.min_segment_distance_constraint.begin_update()
            self.points_query.tranverse(self.on_query_update)

            self.simulator.gradiant.fill(vec(0))
            self.simulator._hession_sparse.deactivate_all()
            self.points_query.tranverse(self.on_query_update_gradiant_hession)
            self.ground_constraint.update_gradiant_hession()
            self._step()

            self.simulator.delta_positions.fill(vec(0))
            self.simulator.constraint_weights.fill(0)
            self.max_displace_constraint.step(update_loss)
            self.simulator.apply_delta_positions()
    @ti.kernel
    def _step(self):
        simulator=self.simulator
        geometry=simulator.geometry
        for p in range(geometry.num_point):
            g=simulator.gradiant[p]
            x_t=self.target_positions[p]
            x=vec(0)
            if not (ti.abs(g)<vec(epsilon)).all():
                h=simulator.hession[p,p]
                n=g.normalized()
                l=g.norm()
                x_p=self.simulator.constrainted_positions[p]
                la=(l+h@(x_t-x_p)@n)/(l*h@n@n)
                x=x_t-la*g
            else:
                x=x_t
            simulator.constrainted_positions[p]=x
    def get_loss(self):
        return tm.max(self.min_segment_distance_constraint.get_loss(),self.max_displace_constraint.get_loss())
        