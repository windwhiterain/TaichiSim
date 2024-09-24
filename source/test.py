from TaichiLib import *
import TaichiSim
ti.init(arch=ti.gpu,default_fp=ti.f32,kernel_profiler=True)
TaichiSim.test()

# @ti.kernel
# def test_segment_distance():
#     sample0=(Segment(vec(-0.13,0,-0.28),vec(-0.07,0,-0.33)),Segment(vec(-0.13,0,-0.33),vec(-0.07,0,-0.28)))
#     sample1=(Segment(vec(-1,0,0),vec(0,0,0)),Segment(vec(1,0,1),vec(2,0,1)))
#     line_dis0=get_distance_line(sample0[0],sample0[1])
#     seg_dis0=get_distance_segment(sample0[0],sample0[1])
#     line_dis1=get_distance_line(sample1[0],sample1[1])
#     seg_dis1=get_distance_segment(sample1[0],sample1[1])
#     print(line_dis0.parameter,line_dis0.distance)
#     print(seg_dis0)
#     print(line_dis1.parameter,line_dis1.distance)
#     print(seg_dis1)
# test_segment_distance()

# @ti.func
# def diag()->int:
#     return 2
# mat.diag=diag
# @ti.kernel
# def a()->int:
#     return mat.diag()

# print(a())

