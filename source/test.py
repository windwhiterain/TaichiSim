from TaichiLib import *
import TaichiSim
import TaichiSim.spatial_query
ti.init(arch=ti.gpu,default_fp=ti.f32,kernel_profiler=True)
TaichiSim.test()

# grid = TaichiSim.spatial_query.Grid(2)
# @ti.kernel
# def append():
#     grid.append(Bound(vec(0,0,0),vec(1,1,1)),vec(0,0,0),0)
#     grid.append(Bound(vec(1.2,1.2,1.2),vec(2,2,2)),vec(1,1,1),1)
# @ti.func
# def update(i:int,j:int):
#     print(i,j)

# append()
# grid.update(update)
# print(grid.keys)
# print(grid.values)
# print(grid.center_idxs)
# print(grid.item_num[None])


