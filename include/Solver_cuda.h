#ifndef SOLVER_CUDA_H
#define SOLVER_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>     // cuBLAS API
#include <torch/extension.h>
#include <chrono>  // For high_resolution_clock

#include "lie.h"

#define NUM_THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + NUM_THREADS - 1) / NUM_THREADS)

#define GPU_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i<n; i += blockDim.x * gridDim.x)


void updateState(
    torch::Tensor observations,
    torch::Tensor incremental_poses,
    torch::Tensor inverse_depths,
    const torch::Tensor intrinsics,
    const torch::Tensor T_r_to_l,
    const torch::Tensor anchor_frame_id,
    const torch::Tensor target_frame_id,
    const torch::Tensor feat_glob_id,
    const int num_of_poses,
    const int num_of_landmarks,
    const int measurement_count,
    const int iterations,
    const float step_size
);

#endif