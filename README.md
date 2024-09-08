# Taichi Sim
A modular, extensible, physic library using taichi.
## Focus
- cloth simulation and procedual generation and design
- physic assisted charactor animation generation and design
## Current implementation
- cloth simulation
  - kinematic solver
    - NewtonRaphsonSolver
    - DiagnalHessionSolver
    - [reference](https://zhuanlan.zhihu.com/p/473396818)
  - spatial query
    - spatial partition
    - [reference](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)
  - collision handler 0.4
    - DCD but safe via limitating max displacement by distance between simplices
    - [reference](https://wanghmin.github.io/publication/wang-2023-fgb/)

