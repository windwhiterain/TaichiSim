from .simulator import *
from .solver import *

def test():
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