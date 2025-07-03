#include "Solver_cuda.h"
#include "LMutils.h"



inline __device__ void loadPose(
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>& poses,
    int pose_idx,
    int left_right,
    float* T_out
) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            T_out[4 * i + j] = poses[pose_idx][left_right][i][j];
        }
    }
}


__global__ void elementwiseInverseKernel(
    const float *in,
    float *out,
    const float eps,
    const int size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<size){
        out[idx] = 1.0f / (in[idx] + eps);
    }
}


__global__ void elementwiseSubtractionKernel(
    const float *in1,
    const float *in2,
    float *out,
    const int size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<size){
        out[idx] = in1[idx] - in2[idx];
    }
}



__global__ void LevenbergMarquardt(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> observations,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> inverse_depths,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> anchor_frame_ids,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> target_frame_ids,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> feat_glob_ids,
    LMvariables *lm_var
){
    __shared__ float Kl[9];
    __shared__ float Kl_inv[9];
    __shared__ float Kr[9];
    __shared__ int thread_max_limit[1];

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
        
        thread_max_limit[0] = lm_var->_measurement_count * 2;
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
    }

    // Get the global thread index
    int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure that the parameters are loaded to the shared memory
    __syncthreads();

    // Get the measurement index
    // Make sure measurement_idx is within bounds 
    if (thread_global_idx >= thread_max_limit[0]) return;

    int measurement_idx   = thread_global_idx / 2;
    int cam_idx           = thread_global_idx % 2;  // 0: Left Cam, 1: Right Cam

    int anchor_idx = anchor_frame_ids[measurement_idx];
    int target_idx = target_frame_ids[measurement_idx];
    int feat_idx   = feat_glob_ids[measurement_idx];
}



__global__ void printArr(float *arr, int dim){
    int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_global_idx < dim){
        printf("%d : %.5f\n", thread_global_idx, arr[thread_global_idx]);
    }
}


__global__ void compute_B_C_inv(LMvariables *lm_var){
    __shared__ int s_row_size;
    __shared__ int s_col_size;
    __shared__ int s_max_size;

    if(threadIdx.x == 0){
        s_row_size = lm_var->_number_of_pose_params;
        s_col_size = lm_var->_number_of_landmarks;
        s_max_size = s_row_size * s_col_size;
    }

    // Make sure that the parameters are loaded to the shared memory
    __syncthreads();

    // Get the measurement index
    int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure measurement_idx is within bounds 
    if (thread_global_idx >= s_max_size) return;

    int col_idx   = thread_global_idx % s_col_size; 

    lm_var->d_B_C_inv[thread_global_idx] = lm_var->d_B[thread_global_idx] / (lm_var->d_H_a[col_idx] + lm_var->_eps);
}


__global__ void add_damping_to_schur(LMvariables *lm_var){
    // Get the size of the alpha vector
    int row_size = lm_var->_number_of_pose_params;

    // Get the global thread index
    int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure measurement_idx is within bounds 
    if (thread_global_idx >= row_size) return;

    lm_var->d_H_schur[thread_global_idx * row_size + thread_global_idx] = lm_var->d_H_schur[thread_global_idx * row_size + thread_global_idx] + lm_var->_lambda_schur;
}

__global__ void compute_delta_a(LMvariables *lm_var){
    
    // Get the size of the alpha vector
    int row_size = lm_var->_number_of_landmarks;

    // Get the global thread index
    int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure measurement_idx is within bounds 
    if (thread_global_idx >= row_size) return;

    // Now you are ready to compute the delta alpha
    lm_var->d_delta_a[thread_global_idx] = (lm_var->d_g_a[thread_global_idx] - lm_var->d_B_T_delta_T[thread_global_idx]) / (lm_var->d_H_a[thread_global_idx] + lm_var->_eps);
}


void updateState(
    torch::Tensor observations,
    torch::Tensor poses,
    torch::Tensor inverse_depths,
    const torch::Tensor intrinsics,
    const torch::Tensor T_r_to_l,
    const torch::Tensor anchor_frame_id,
    const torch::Tensor target_frame_id,
    const torch::Tensor feat_glob_id,
    const int num_of_poses,
    const int num_of_landmark,
    const int measurement_count,
    const int iterations,
    const float step_size
){


    // First, initialize our Levenberg-Marquardt variables structure on the host (CPU)
    LMvariables h_lm_var;
    h_lm_var.allocateMemory(num_of_poses, num_of_landmark, measurement_count);

    // Declare a pointer to LMvariables that will reside on the device (GPU)
    LMvariables* d_lm_var;
    cudaMalloc((void**)&d_lm_var, sizeof(LMvariables));
    cudaMemcpy(d_lm_var, &h_lm_var, sizeof(LMvariables), cudaMemcpyHostToDevice);

    // Wait untill kernel is completed
    cudaDeviceSynchronize();

    for(int it_idx=0; it_idx<iterations; it_idx++){
        // Compute the A, B, C and g_a, g_T
        LevenbergMarquardt<<<NUM_BLOCKS(h_lm_var._measurement_count * 2), NUM_THREADS>>>(
            observations.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            poses.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            inverse_depths.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            anchor_frame_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            target_frame_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            feat_glob_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            d_lm_var
        );

        // Wait untill kernel is completed
        cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();
    cudaFree(d_lm_var);
    h_lm_var.freeAll();
}