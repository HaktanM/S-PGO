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


## Time Benchmarking
To evaluate the effectiveness of our CUDA implementation, we report the elapsed time for a single optimization step, comparing both the Python and CUDA implementations in the table.

| Test | Number of Key Frames | Number of Landmarks per Frame | Total Measurements | CPU Time (ms) | CUDA Time (ms) |
| ------ | ---------- | ------------------- | ------------------ | ------------- | -------------- |
|   1  | 3          | 48                  | 1,728              | 135           | 2.20           |
|   2  | 3          | 96                  | 3,456              | 261           | 2.29           |
|   3  | 6          | 48                  | 5,184              | 679           | 11.43          |
|   4  | 6          | 96                  | 10,368             | 1,422         | 10.86          |
|   5  | 9          | 48                  | 10,368             | 2,137         | 27.94          |
|   6  | 9          | 96                  | 20,736             | 4,433         | 27.68          |
|   7  | 12         | 48                  | 17,280             | 5,442         | 59.20          |
|   8  | 12         | 96                  | 34,560             | 10,961        | 62.22          |
|   9  | 12         | 128                 | 46,080             | 14,903        | 60.04          |

The Table presents the elapsed times for a single optimization iteration. It is immediately apparent that the Python implementation scales approximately linearly with the total number of measurements. In contrast, the CUDA implementation's runtime is largely influenced by the number of keyframes. This behavior arises because certain operations are inherently serial and strongly dependent on the number of keyframes, ultimately dominating the overall execution time. **An in depth analysis will be presented in the upcoming paper!!!**
