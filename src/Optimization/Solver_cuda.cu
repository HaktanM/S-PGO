#include "Solver_cuda.h"


inline __device__ void loadPose(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>& poses,
    int pose_idx,
    float* T_out
) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            T_out[4 * i + j] = poses[pose_idx][i][j];
        }
    }
}


inline __device__ void getRotFromT(const float *T, float *R){
    R[0] = T[0];
    R[1] = T[1];
    R[2] = T[2];

    R[3] = T[4];
    R[4] = T[5];
    R[5] = T[6];

    R[6] = T[8];
    R[7] = T[9];
    R[8] = T[10];
}

inline __device__ void compute_del_tn_del_xi(const float *T_state_to_target, const float *t_feat_in_state, float *del_tn_del_xi){
    float R_state_to_target[9];
    getRotFromT(T_state_to_target, R_state_to_target);

    float skew_t_feat_in_state[9];
    skew(t_feat_in_state, skew_t_feat_in_state);

    float R_Skew_t[9];
    MatrixMultiplication(R_state_to_target, skew_t_feat_in_state, R_Skew_t, 3, 3, 3);

    del_tn_del_xi[0] = - R_Skew_t[0];
    del_tn_del_xi[1] = - R_Skew_t[1];
    del_tn_del_xi[2] = - R_Skew_t[2];
    del_tn_del_xi[3] = R_state_to_target[0];
    del_tn_del_xi[4] = R_state_to_target[1];
    del_tn_del_xi[5] = R_state_to_target[2];

    del_tn_del_xi[6]  = - R_Skew_t[3];
    del_tn_del_xi[7]  = - R_Skew_t[4];
    del_tn_del_xi[8]  = - R_Skew_t[5];
    del_tn_del_xi[9]  = R_state_to_target[3];
    del_tn_del_xi[10] = R_state_to_target[4];
    del_tn_del_xi[11] = R_state_to_target[5];

    del_tn_del_xi[12] = - R_Skew_t[6];
    del_tn_del_xi[13] = - R_Skew_t[7];
    del_tn_del_xi[14] = - R_Skew_t[8];
    del_tn_del_xi[15] = R_state_to_target[6];
    del_tn_del_xi[16] = R_state_to_target[7];
    del_tn_del_xi[17] = R_state_to_target[8];
}

inline __device__ void computePoseInverse(const float* T, float* T_inv) {
    // Extract rotation R and translation t
    // T is row-major, so:
    // R: T[0..2], T[4..6], T[8..10]
    // t: T[3], T[7], T[11]

    // Compute R^T (transpose of R)
    T_inv[0] = T[0];
    T_inv[1] = T[4];
    T_inv[2] = T[8];

    T_inv[4] = T[1];
    T_inv[5] = T[5];
    T_inv[6] = T[9];

    T_inv[8] = T[2];
    T_inv[9] = T[6];
    T_inv[10] = T[10];

    // Compute -R^T * t
    T_inv[3]  = -(T_inv[0] * T[3] + T_inv[1] * T[7] + T_inv[2] * T[11]);
    T_inv[7]  = -(T_inv[4] * T[3] + T_inv[5] * T[7] + T_inv[6] * T[11]);
    T_inv[11] = -(T_inv[8] * T[3] + T_inv[9] * T[7] + T_inv[10] * T[11]);

    // Last row is always [0, 0, 0, 1]
    T_inv[12] = 0.0f;
    T_inv[13] = 0.0f;
    T_inv[14] = 0.0f;
    T_inv[15] = 1.0f;
}


inline __device__ void compute_del_bn_del_tn(float *t_feat_cn, float *del_bn_del_tn){
    del_bn_del_tn[0] = 1.0f / t_feat_cn[2];
    del_bn_del_tn[1] = 0.0f;
    del_bn_del_tn[2] = -t_feat_cn[0] / t_feat_cn[2];

    del_bn_del_tn[3] = 0.0f;
    del_bn_del_tn[4] = 1.0f / t_feat_cn[2];
    del_bn_del_tn[5] = -t_feat_cn[1] / t_feat_cn[2];

    del_bn_del_tn[6] = 0.0f;
    del_bn_del_tn[7] = 0.0f;
    del_bn_del_tn[8] = 0.0f;
}

__global__ void JacobianAndResidualKernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> observations,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> inverse_depths,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> T_r_to_l_global_memory,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> anchor_frame_ids,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> target_frame_ids,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> feat_glob_ids
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

    // if (threadIdx.x == 5) {
    //     for(int row_idx = 0; row_idx<3; row_idx++){
    //         for(int col_idx = 0; col_idx<3; col_idx++){
    //             printf("%.4f, ", Kl[row_idx*3 + col_idx]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    //     for(int row_idx = 0; row_idx<3; row_idx++){
    //         for(int col_idx = 0; col_idx<3; col_idx++){
    //             printf("%.4f, ", Kr_inv[row_idx*3 + col_idx]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    //     for(int row_idx = 0; row_idx<4; row_idx++){
    //         for(int col_idx = 0; col_idx<4; col_idx++){
    //             printf("%.4f, ", T_r_to_l[row_idx*4 + col_idx]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");

    // }

    int measurement_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure measurement_idx is within bounds 
    if (measurement_idx >= anchor_frame_ids.size(0)) return;

    int anchor_idx = anchor_frame_ids[measurement_idx];
    int target_idx = target_frame_ids[measurement_idx];
    int feat_idx   = feat_glob_ids[measurement_idx];


    // Get global poses
    float T_anchor_to_glob[16];
    float T_target_to_glob[16];
    float T_state_to_glob[16];
    
    loadPose(poses, anchor_idx, T_anchor_to_glob);
    loadPose(poses, target_idx, T_target_to_glob);

    // Inverse of these are also required
    float T_glob_to_target[16];
    computePoseInverse(T_target_to_glob, T_glob_to_target);

    // Get the observation in the anchor
    float p_in_anchor[3] = { observations[anchor_idx][feat_idx][0][0], observations[anchor_idx][feat_idx][0][1], 1.0f };


    // Compute t_feat_in_anchor
    float t_feat_in_anchor[4];
    float alpha = inverse_depths[feat_idx];
    MatrixMultiplication(Kl_inv, p_in_anchor, t_feat_in_anchor, 3, 3, 1);

    t_feat_in_anchor[0] = t_feat_in_anchor[0] / alpha;
    t_feat_in_anchor[1] = t_feat_in_anchor[1] / alpha;
    t_feat_in_anchor[2] = t_feat_in_anchor[2] / alpha;
    t_feat_in_anchor[3] = 1.0f;


    for(int state_idx = anchor_idx; state_idx < (poses.size(0)-1); state_idx++){
        // Load state to global pose
        loadPose(poses, state_idx, T_state_to_glob);

        // Compute 
        float T_glob_to_state[16];
        computePoseInverse(T_state_to_glob, T_glob_to_state);

        // Compute T_state_to_target
        float T_state_to_target[16];
        MatrixMultiplication(T_glob_to_target, T_state_to_glob, T_state_to_target, 4, 4, 4);

        // Compute T_anchor_to_state
        float T_anchor_to_state[16];
        MatrixMultiplication(T_glob_to_state, T_anchor_to_glob, T_anchor_to_state, 4, 4, 4);

        // Compute t_feat_in_state
        float t_feat_in_state[4];
        MatrixMultiplication(T_anchor_to_state, t_feat_in_anchor, t_feat_in_state, 4, 4, 4);

        // Compute del_tn_del_xi
        float del_tn_del_xi[18];
        compute_del_tn_del_xi(T_state_to_target, t_feat_in_state, del_tn_del_xi);
    }
    
    // Compute the Jacobians of the left camera frame

    // del_pn_del_bn
    float del_pn_del_bn[9];
    for(int idx=0; idx<9; idx++){
        del_pn_del_bn[idx] = Kl[idx];
    }

    // del_bn_del_tn 
    float t_feat_cn[3];  // Compute this anyhow
    t_feat_cn[0] = 1.0f;
    t_feat_cn[1] = 1.0f;
    t_feat_cn[2] = 1.0f;

    float del_bn_del_tn[9];

    compute_del_bn_del_tn(t_feat_cn, del_bn_del_tn);
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

    // Compute the poses first
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);;
    torch::Tensor poses = torch::zeros({incremental_poses.size(0)+1, 4, 4}, options); 
    
    // Set the first pose to identity
    poses[0] = torch::eye(4, options);

    /*
    incremental poses are T_curr_to_next
    poses are T_curr_to_global
    */

    // Compute global poses by chaining transforms
    for (int i = 0; i < incremental_poses.size(0); ++i) {
        // Inverse of T_curr_to_next
        torch::Tensor T_next_to_curr = torch::linalg_inv(incremental_poses[i]);

        // poses[i+1] = poses[i] @ inv(T_curr_to_next)
        poses[i+1] = poses[i].matmul(T_next_to_curr);
    }

    // Compute the Jacobians and residual
    JacobianAndResidualKernel<<<NUM_BLOCKS(anchor_frame_id.size(0)), NUM_THREADS>>>(
        observations.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        poses.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        inverse_depths.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        T_r_to_l.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        anchor_frame_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
        target_frame_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
        feat_glob_id.packed_accessor32<int,1,torch::RestrictPtrTraits>()
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