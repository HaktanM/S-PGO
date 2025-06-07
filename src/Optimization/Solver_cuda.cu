#include "Solver_cuda.h"

__global__ void JacobianAndResidualKernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> observations,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> incremental_poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> inverse_depths,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> T_r_to_l_global_memory
    // torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> J_T,
    // torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> J_alpha,
    // torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> r
){
    __shared__ float Kl[9];
    __shared__ float Kl_inv[9];
    __shared__ float Kr[9];
    __shared__ float Kr_inv[9];
    __shared__ float T_r_to_l[16];
    // First load the data which is commonly used by all threads to the shared memory
    if (threadIdx.x == 0) {      // Load left camera parameters
        Kl[0] = intrinsics[0][0]; // fx
        Kl[1] = 0.0f; 
        Kl[2] = intrinsics[0][2]; // cx

        Kl[3] = 0.0f; 
        Kl[4] = intrinsics[0][1]; // fy
        Kl[5] = intrinsics[0][3]; // cy

        Kl[6] = 0.0f; 
        Kl[7] = 0.0f; 
        Kl[8] = 1.0f;   
    } else if (threadIdx.x == 1){
        Kl_inv[0] = 1.0f / intrinsics[0][0]; //  1 / fx
        Kl_inv[1] = 0.0f; 
        Kl_inv[2] = - intrinsics[0][2] / intrinsics[0][0]; //  -cx / fx

        Kl_inv[3] = 0.0f; 
        Kl_inv[4] = 1.0f / intrinsics[0][1]; //  1 / fy
        Kl_inv[5] = - intrinsics[0][3] / intrinsics[0][1]; //  -cy / fy

        Kl_inv[6] = 0.0f; 
        Kl_inv[7] = 0.0f; 
        Kl_inv[8] = 1.0f; 
    }
    else if (threadIdx.x == 2) {
        Kr[0] = intrinsics[1][0]; // fx
        Kr[1] = 0.0f; 
        Kr[2] = intrinsics[1][2]; // cx

        Kr[3] = 0.0f; 
        Kr[4] = intrinsics[1][1]; // fy
        Kr[5] = intrinsics[1][3]; // cy

        Kr[6] = 0.0f; 
        Kr[7] = 0.0f; 
        Kr[8] = 1.0f; 
    } else if (threadIdx.x == 3) {
        Kr_inv[0] = 1.0f / intrinsics[1][0]; //  1 / fx
        Kr_inv[1] = 0.0f; 
        Kr_inv[2] = - intrinsics[1][2] / intrinsics[1][0]; //  -cx / fx

        Kr_inv[3] = 0.0f; 
        Kr_inv[4] = 1.0f / intrinsics[1][1]; //  1 / fy
        Kr_inv[5] = - intrinsics[1][3] / intrinsics[1][1]; //  -cy / fy

        Kr_inv[6] = 0.0f; 
        Kr_inv[7] = 0.0f; 
        Kr_inv[8] = 1.0f; 
    } else if (threadIdx.x == 4) { // Load extrinsic calibration
        T_r_to_l[0] = T_r_to_l_global_memory[0][0];
        T_r_to_l[1] = T_r_to_l_global_memory[0][1];
        T_r_to_l[2] = T_r_to_l_global_memory[0][2];
        T_r_to_l[3] = T_r_to_l_global_memory[0][3];

        T_r_to_l[4] = T_r_to_l_global_memory[1][0];
        T_r_to_l[5] = T_r_to_l_global_memory[1][1];
        T_r_to_l[6] = T_r_to_l_global_memory[1][2];
        T_r_to_l[7] = T_r_to_l_global_memory[1][3];

        T_r_to_l[8]  = T_r_to_l_global_memory[2][0];
        T_r_to_l[9]  = T_r_to_l_global_memory[2][1];
        T_r_to_l[10] = T_r_to_l_global_memory[2][2];
        T_r_to_l[11] = T_r_to_l_global_memory[2][3];

        T_r_to_l[12] = 0.0f;
        T_r_to_l[13] = 0.0f;
        T_r_to_l[14] = 0.0f;
        T_r_to_l[15] = 1.0f;
    }

    // Make sure that the parameters are loaded to the shared memory
    __syncthreads();
    if (threadIdx.x == 5) {
        for(int row_idx = 0; row_idx<3; row_idx++){
            for(int col_idx = 0; col_idx<3; col_idx++){
                printf("%.4f, ", Kl[row_idx*3 + col_idx]);
            }
            printf("\n");
        }
        printf("\n");
        for(int row_idx = 0; row_idx<3; row_idx++){
            for(int col_idx = 0; col_idx<3; col_idx++){
                printf("%.4f, ", Kr_inv[row_idx*3 + col_idx]);
            }
            printf("\n");
        }
        printf("\n");
        for(int row_idx = 0; row_idx<4; row_idx++){
            for(int col_idx = 0; col_idx<4; col_idx++){
                printf("%.4f, ", T_r_to_l[row_idx*4 + col_idx]);
            }
            printf("\n");
        }
        printf("\n");

    }
}

void updateState(
    torch::Tensor observations,
    torch::Tensor incremental_poses,
    torch::Tensor inverse_depths,
    const torch::Tensor intrinsics,
    const torch::Tensor T_r_to_l,
    const torch::Tensor anchor_frame_id,
    const torch::Tensor target_frame_id,
    const torch::Tensor feat_glob_id
){
    // Compute the Jacobians and residual
    JacobianAndResidualKernel<<<NUM_BLOCKS(observations.size(0)), NUM_THREADS>>>(
        observations.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        incremental_poses.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        inverse_depths.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        T_r_to_l.packed_accessor32<float,2,torch::RestrictPtrTraits>()
        // J_T.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        // J_alpha.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        // r.packed_accessor32<float,2,torch::RestrictPtrTraits>()
    );


    // Check if there exists any error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Wait until the Jacobians are ready
    cudaDeviceSynchronize();  // Wait for kernel to finish to see printf output or errors
}