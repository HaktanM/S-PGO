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


## What is so special about this repo
I have implemented **Levenberg-Marquardt** for stereo pose estimation on CUDA. The pose graph contains **over 10000 edges**. This implementation efficiently computes the Hessian matrix, applies Schur’s complement, and solves the system in **under 50 milliseconds on an NVIDIA GeForce RTX 2060** for a single optimization step.

The goal of this repository is to demonstrate how to efficiently implement and solve a pose graph optimization on the GPU. **Detailed documentation will be available soon!**
