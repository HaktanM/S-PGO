## Introduction
This repository implements stereo pose graph optimization on the GPU. In the animation below, the green cameras represent the simulated trajectory, while the red ones show the estimated camera poses. Landmarks are also visualized. For visualization smoothness, the step size in the optimization has been kept small.

<div align="center">
  <img src="https://github.com/HaktanM/S-DPVO/blob/main/Figures/optimization_steps.gif" alt="description" width="80%">
</div>

Detailed documentation and implementation manual will be available after the publication of the related paper. 

## Build the Code
```bash
cd S-DPVO
mkdir build && cd build 
cmake ..
cmake --build . -- -j$(nproc)
```

## Run
All you need to do is to run **CUDAoptimization.py**. You can change the number of cameras and landmarks.
```python
class Manager():
    def __init__(self):

        # Number of keyframes
        self.n = 12

        # NUmber of landmarks per frame
        self.m = 96
```


## TO DO
* Compute global poses inside the kernel. Torch is too slow :(
* What is the effect of the window size in a noisy case? Highlight the fact that we need larger pose graphs for more accurate pose estimations. 
* Time tests for different window size
* Check cuSolver
