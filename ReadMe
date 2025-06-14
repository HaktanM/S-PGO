## Build the Code
```bash
cd S-DPVO
mkdir build && cd build 
cmake ..
cmake --build . -- -j$(nproc)
```


### Introduction
This repository implements stereo pose graph optimization on the GPU. In the animation below, the green cameras represent the simulated trajectory, while the red ones show the estimated camera poses. Landmarks are also visualized. For visualization smoothness, the step size in the optimization has been kept small.

<div align="center">
  <img src="https://github.com/HaktanM/S-DPVO/blob/main/Figures/optimization_steps.gif" alt="description" width="80%">
</div>


## TO DO
* Compute global poses inside the kernel. Torch is too slow :(
* What is the effect of the window size in a noisy case? Highlight the fact that we need larger pose graphs for more accurate pose estimations. 
* Time tests for different window size
* Check cuSolver
