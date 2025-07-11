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


    // Get global poses
    float T_anchor_to_glob[16];
    float T_target_to_glob[16];
    float T_anchor_to_target[16];
    
    for(int row_idx=0; row_idx<4; row_idx++){
        for(int col_idx=0; col_idx<4; col_idx++){
            T_anchor_to_glob[row_idx*4 + col_idx] = lm_var->_global_left_poses[anchor_idx*16 + row_idx*4 + col_idx];
            if(cam_idx == 0)
            { // Load left camera pose
                T_target_to_glob[row_idx*4 + col_idx] = lm_var->_global_left_poses[target_idx*16 + row_idx*4 + col_idx];
            }else
            { // Load right camera pose
                T_target_to_glob[row_idx*4 + col_idx] = lm_var->_global_right_poses[target_idx*16 + row_idx*4 + col_idx];
            }
        }
    }
    

    // Inverse of these are also required
    float T_glob_to_target[16];
    InvertPose4x4(T_target_to_glob, T_glob_to_target);

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

    // Compute t_feat_in_target
    MatrixMultiplication(T_glob_to_target, T_anchor_to_glob, T_anchor_to_target, 4, 4, 4);
    float t_feat_in_target[4];
    MatrixMultiplication(T_anchor_to_target, t_feat_in_anchor, t_feat_in_target, 4, 4, 1);


    // Compute estimated observation
    float b_feat_in_target[3];
    float p_in_target_est[3];
    b_feat_in_target[0] = t_feat_in_target[0] / t_feat_in_target[2]; 
    b_feat_in_target[1] = t_feat_in_target[1] / t_feat_in_target[2];
    b_feat_in_target[2] = 1.0;

    if(cam_idx == 0){     // If the observation belongs to left camera
        MatrixMultiplication(Kl, b_feat_in_target, p_in_target_est, 3, 3, 1);
    }else{
        MatrixMultiplication(Kr, b_feat_in_target, p_in_target_est, 3, 3, 1);
    }

    // del_pn_del_bn
    float del_pn_del_bn[9];
    if(cam_idx == 0){     // If the observation belongs to left camera
        for(int idx=0; idx<9; idx++){
            del_pn_del_bn[idx] = Kl[idx];
        }
    }else{
        for(int idx=0; idx<9; idx++){
            del_pn_del_bn[idx] = Kr[idx];
        }
    }

    // del_bn_del_tn 
    float del_bn_del_tn[9];
    compute_del_bn_del_tn(t_feat_in_target, del_bn_del_tn);

    float del_pn_del_tn[9];
    MatrixMultiplication(del_pn_del_bn, del_bn_del_tn, del_pn_del_tn, 3, 3, 3);
    
    // Compute J_alpha
    float del_tn_del_ta[9];
    getRotFromT(T_anchor_to_target, del_tn_del_ta);

    float del_ta_del_alpha[3];
    del_ta_del_alpha[0] = - t_feat_in_anchor[0] / alpha;
    del_ta_del_alpha[1] = - t_feat_in_anchor[1] / alpha;
    del_ta_del_alpha[2] = - t_feat_in_anchor[2] / alpha;

    
    float del_pn_del_ta[9];
    float del_pn_del_alpha[3];

    MatrixMultiplication(del_pn_del_tn, del_tn_del_ta, del_pn_del_ta, 3, 3, 3);
    MatrixMultiplication(del_pn_del_ta, del_ta_del_alpha, del_pn_del_alpha, 3, 3, 1);
    float del_fxy_del_alpha[2];

    // First cauchy weight
    float res_mag_square{0.0};
    for(int xy_idx=0; xy_idx<2; xy_idx++){
        float res = observations[target_idx][feat_idx][cam_idx][xy_idx] - p_in_target_est[xy_idx];
        res_mag_square += res * res;
    }
    
    // Compute cauchy weight
    float cauchy_weight = sqrtf( 1.0f / ( 1.0 + (res_mag_square / lm_var->_cauchy_constant_square) ));


    for(int xy_idx=0; xy_idx<2; xy_idx++){

        int J_alpha_row_idx = 4 * measurement_idx + xy_idx + 2 * cam_idx;

        
        // Compute the residual
        lm_var->d_r[J_alpha_row_idx] = cauchy_weight * ( observations[target_idx][feat_idx][cam_idx][xy_idx] - p_in_target_est[xy_idx] );
        
        // Fill H_aa and g_alpha
        atomicAdd(&lm_var->d_C[feat_idx],   cauchy_weight * cauchy_weight * del_pn_del_alpha[xy_idx] * del_pn_del_alpha[xy_idx]);
        atomicAdd(&lm_var->d_g_a[feat_idx], cauchy_weight * del_pn_del_alpha[xy_idx] * lm_var->d_r[J_alpha_row_idx]);

        del_fxy_del_alpha[xy_idx] = cauchy_weight * del_pn_del_alpha[xy_idx];
    }


    // Compute B = J_T and J_T^T @ J_alpha
    for(int state_idx = anchor_idx; state_idx < target_idx; state_idx++){
        // Load state to global pose
        float T_state_to_glob[16];
        // loadPose(poses, state_idx, 0, T_state_to_glob);
        for(int row_idx=0; row_idx<16; row_idx++){
            for(int col_idx=0; col_idx<16; col_idx++){
                T_state_to_glob[row_idx*4 + col_idx] = lm_var->_global_left_poses[state_idx*16 + row_idx*4 + col_idx];
            }
        }

        // Compute 
        float T_glob_to_state[16];
        InvertPose4x4(T_state_to_glob, T_glob_to_state);

        // Compute T_state_to_target
        float T_state_to_target[16];
        MatrixMultiplication(T_glob_to_target, T_state_to_glob, T_state_to_target, 4, 4, 4);

        // Compute T_anchor_to_state
        float T_anchor_to_state[16];
        MatrixMultiplication(T_glob_to_state, T_anchor_to_glob, T_anchor_to_state, 4, 4, 4);

        // Compute t_feat_in_state
        float t_feat_in_state[4];
        MatrixMultiplication(T_anchor_to_state, t_feat_in_anchor, t_feat_in_state, 4, 4, 1);

        // Compute del_tn_del_xi
        float del_tn_del_xi[18];
        compute_del_tn_del_xi(T_state_to_target, t_feat_in_state, del_tn_del_xi);

        // Compute del_pn_del_xi
        float del_pn_del_xi[18];
        MatrixMultiplication(del_pn_del_tn, del_tn_del_xi, del_pn_del_xi, 3, 3, 6);
        
        // Fill the Jacobian
        for(int xy_idx=0; xy_idx<2; xy_idx++){
            for(int col_idx=0; col_idx<6; col_idx++){
                int J_T_row_idx = 4 * measurement_idx + xy_idx + 2 * cam_idx;
                int J_T_col_idx = 6 * state_idx + col_idx;
                lm_var->d_J_T[J_T_row_idx * lm_var->_number_of_pose_params + J_T_col_idx] = cauchy_weight * del_pn_del_xi[6 * xy_idx + col_idx];
            }
        }
    }

    // Finally, compute B
    for(int state_idx = anchor_idx; state_idx < target_idx; state_idx++){
        // Fill the Jacobian
        for(int xy_idx=0; xy_idx<2; xy_idx++){
            for(int col_idx=0; col_idx<6; col_idx++){
                int B_row_idx = 6 * state_idx + col_idx;
                int B_col_idx = feat_idx;
                
                int JT_row_idx = 4 * measurement_idx + 2 * cam_idx + xy_idx;
                int JT_col_idx = 6 * state_idx + col_idx;

                float del_f_del_t = lm_var->d_J_T[JT_row_idx * lm_var->_number_of_pose_params + JT_col_idx];
                float del_f_del_a = del_fxy_del_alpha[xy_idx];

                atomicAdd(&lm_var->d_B[B_row_idx * lm_var->_number_of_landmarks + B_col_idx], del_f_del_t * del_f_del_a);
            }
        }
    }
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

    lm_var->d_B_C_inv[thread_global_idx] = lm_var->d_B[thread_global_idx] / (lm_var->d_C[col_idx] + lm_var->_eps);
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
    lm_var->d_delta_a[thread_global_idx] = (lm_var->d_g_a[thread_global_idx] - lm_var->d_B_T_delta_T[thread_global_idx]) / (lm_var->d_C[thread_global_idx] + lm_var->_eps);
}



__global__ void loadIncrementalPoses(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> incremental_poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> T_r_to_l,
    LMvariables *lm_var
)
{   
    // Get the size of the alpha vector
    int number_of_poses = lm_var->_number_of_poses;
    int max_item        = number_of_poses * 16;

    // Get the global thread index
    int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    // Load incremental poses
    if(thread_global_idx < max_item){
        int incremental_pose_index = thread_global_idx / 16;
        int index_of_item          = thread_global_idx % 16;
        int row_idx                = index_of_item / 4;
        int col_idx                = index_of_item % 4;

        lm_var->_incremental_poses[thread_global_idx] = incremental_poses[incremental_pose_index][row_idx][col_idx];
    } 
    else if(thread_global_idx == max_item){
        // Initalize the poses at first time instant for right
        for(int row_idx=0; row_idx<4; row_idx++){
            for(int col_idx=0; col_idx<4; col_idx++){
                lm_var->d_T_r_to_l[row_idx*4 + col_idx] = T_r_to_l[row_idx][col_idx];
            }
        }
    } 
}




__global__ void loadIncrementalPosesBack(
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> incremental_poses,
    LMvariables *lm_var
)
{   
    // Get the size of the alpha vector
    int number_of_poses = lm_var->_number_of_poses;
    int max_item        = number_of_poses * 16;

    // Get the global thread index
    int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    // Load incremental poses
    if(thread_global_idx < max_item){
        int incremental_pose_index = thread_global_idx / 16;
        int index_of_item          = thread_global_idx % 16;
        int row_idx                = index_of_item / 4;
        int col_idx                = index_of_item % 4;

        incremental_poses[incremental_pose_index][row_idx][col_idx] = lm_var->_incremental_poses[thread_global_idx];
    } 
}



__global__ void computeGlobalPoses(
    LMvariables *lm_var
)
{   
    // Get the size of the alpha vector
    int number_of_poses = lm_var->_number_of_poses;

    // Get the global thread index
    int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the global poses.
    // This process has to be sequential, nothing to do
    if(thread_global_idx == 0){
        // Initalize the poses at first time instant for left
        for(int row_idx=0; row_idx<4; row_idx++){
            for(int col_idx=0; col_idx<4; col_idx++){
                if(row_idx==col_idx){
                    lm_var->_global_left_poses[row_idx*4 + col_idx] = 1.0;
                }else{
                    lm_var->_global_left_poses[row_idx*4 + col_idx] = 0.0;
                }
            }
        }

        // Initalize the poses at first time instant for right
        for(int row_idx=0; row_idx<4; row_idx++){
            for(int col_idx=0; col_idx<4; col_idx++){
                lm_var->_global_right_poses[row_idx*4 + col_idx] = lm_var->d_T_r_to_l[row_idx*4 + col_idx];
            }
        }

        // Compute global poses
        for(int pose_idx=0; pose_idx<number_of_poses; pose_idx++){
            float T_next_to_curr[16];
            InvertPose4x4(
                &lm_var->_incremental_poses[pose_idx * 16],
                T_next_to_curr
            );
            MatrixMultiplication(
                &lm_var->_global_left_poses[pose_idx * 16],       
                T_next_to_curr,                            
                &lm_var->_global_left_poses[(pose_idx + 1) * 16],
                4, 4, 4
            );

            MatrixMultiplication(
                &lm_var->_global_left_poses[(pose_idx + 1) * 16],
                lm_var->d_T_r_to_l,                          
                &lm_var->_global_right_poses[(pose_idx + 1) * 16], 
                4, 4, 4
            );   
        }
    } 
}



void __global__ updateEstimation(
    LMvariables *lm_var,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> inverse_depths,
    const float step_size
){
    // Get the size of the alpha vector
    int number_of_poses      = lm_var->_number_of_poses;
    int number_of_landmarks  = lm_var->_number_of_landmarks;

    // Get the global thread index
    int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;


    
    if(thread_global_idx < number_of_poses){
        // Update incremental pose
        int pose_idx = thread_global_idx;
        float updated_incremental_pose[16];
        float state_innovation[16];

        for(int idx=0; idx<6; ++idx){
            lm_var->d_g_schur[pose_idx * 6 + idx] = step_size * lm_var->d_g_schur[pose_idx * 6 + idx];
        }

        ExpSE3(
            &lm_var->d_g_schur[pose_idx * 6],
            state_innovation
        );
        
        MatrixMultiplication(
            &lm_var->_incremental_poses[pose_idx * 16],
            state_innovation,                          
            updated_incremental_pose, 
            4, 4, 4
        );

        // Rewrite the incremental state
        for(int idx=0; idx<16; idx++){
            lm_var->_incremental_poses[pose_idx * 16 + idx] = updated_incremental_pose[idx];
        }


    }
    else if(thread_global_idx < number_of_poses + number_of_landmarks ){
        // Update estimated depth
        int landmark_idx = thread_global_idx - number_of_poses;

        // Update inverse depth
        float updated_inverse_depth = inverse_depths[landmark_idx] + lm_var->d_delta_a[landmark_idx];
        updated_inverse_depth = fminf(fmaxf(updated_inverse_depth, lm_var->_min_inv_depth), lm_var->_max_inv_depth);

        // Rewrite the inverse depth
        inverse_depths[landmark_idx] += step_size * lm_var->d_delta_a[landmark_idx];

        inverse_depths[landmark_idx] = updated_inverse_depth;
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
    const torch::Tensor feat_glob_id,
    const int num_of_poses,
    const int num_of_landmark,
    const int measurement_count,
    const int iterations,
    const float step_size
){


    // 

    LMvariables h_lm_var;
    h_lm_var.allocateMemory(num_of_poses, num_of_landmark, measurement_count);

    LMvariables* d_lm_var;
    cudaMalloc((void**)&d_lm_var, sizeof(LMvariables));

    // We need a pointer to our variables in cuda. 
    cudaMemcpy(d_lm_var, &h_lm_var, sizeof(LMvariables), cudaMemcpyHostToDevice);

    loadIncrementalPoses<<<NUM_BLOCKS(h_lm_var._number_of_poses*16+1), NUM_THREADS>>>(
        incremental_poses.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        T_r_to_l.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        d_lm_var
    );

    // Wait untill kernel is completed
    cudaDeviceSynchronize();

    for(int it_idx=0; it_idx<iterations; it_idx++){
        // auto start = std::chrono::high_resolution_clock::now();

        // Compute Global Poses First
        computeGlobalPoses<<<NUM_BLOCKS(1), NUM_THREADS>>>(d_lm_var);
    
        // Compute the A, B, C and g_a, g_T
        LevenbergMarquardt<<<NUM_BLOCKS(h_lm_var._measurement_count * 2), NUM_THREADS>>>(
            observations.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            inverse_depths.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            anchor_frame_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            target_frame_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            feat_glob_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            d_lm_var
        );


        // Wait untill kernel is completed
        cudaDeviceSynchronize();
        
        // Compute the inverse of BC_inv
        compute_B_C_inv<<<NUM_BLOCKS(h_lm_var._number_of_pose_params * h_lm_var._number_of_landmarks), NUM_THREADS>>>(d_lm_var);

        
        
        cudaError_t err = cudaGetLastError ();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
        }


        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        cudaStream_t cublas_stream;
        cublasGetStream(handle, &cublas_stream);  // Get cuBLAS internal stream

        // J_T and J_alpha are row major. But cuBLAS requires column major indexing. 
        // Hence, consider J_T and J_alpha as J_T.transpose and J_alpha.transpose are
        float alpha{1.0}, beta{0.0};
        cublasStatus_t err_cublas = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            static_cast<size_t>(h_lm_var._number_of_pose_params), static_cast<size_t>(h_lm_var._number_of_pose_params), static_cast<size_t>(h_lm_var._measurement_size),
            &alpha,
            h_lm_var.d_J_T, h_lm_var._number_of_pose_params,     
            h_lm_var.d_J_T, h_lm_var._number_of_pose_params,
            &beta,
            h_lm_var.d_A, h_lm_var._number_of_pose_params
        );
        // Output is symmetric, hence, it does not matter if it is row major or column major.

        if (err_cublas != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasSgemm failed: " << err_cublas << std::endl;
        }
        cudaStreamSynchronize(cublas_stream);

        err_cublas = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<size_t>(h_lm_var._number_of_pose_params), static_cast<size_t>(1), static_cast<size_t>(h_lm_var._measurement_size),
            &alpha,
            h_lm_var.d_J_T, h_lm_var._number_of_pose_params,
            h_lm_var.d_r,   h_lm_var._measurement_size,
            &beta,
            h_lm_var.d_g_T, h_lm_var._number_of_pose_params
        );
        // Output is a vector, hence, it does not matter if it is row major or column major.
        if (err_cublas != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasSgemm failed: " << err_cublas << std::endl;
        }

        cudaStreamSynchronize(cublas_stream);

        // h_lm_var.J_T_to_txt();
        // h_lm_var.A_to_txt();
        // h_lm_var.r_to_txt();
        // h_lm_var.g_T_to_txt();

        cudaStreamSynchronize(cublas_stream);     // Synchronize on that stream
        cudaDeviceSynchronize();

        
        err = cudaGetLastError ();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
        }

        // Now we are ready to compute the rest
        // DON'T FORGET THAT, I have adopted the row-major array indexing for representing a matrix as an array.
        // However, cuBLAS assumes column-major indexing. 
        // Hence, cuBLAS percieves my arrays as their transpose

        // Compute B_C_inv_B_T
        err_cublas = cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<size_t>(h_lm_var._number_of_pose_params), static_cast<size_t>(h_lm_var._number_of_pose_params), static_cast<size_t>(h_lm_var._number_of_landmarks),
            &alpha,
            h_lm_var.d_B_C_inv, h_lm_var._number_of_landmarks,     
            h_lm_var.d_B, h_lm_var._number_of_landmarks,
            &beta,
            h_lm_var.d_B_C_inv_B_T, h_lm_var._number_of_pose_params
        );
        // Output is symmetric, hence, it does not matter if it is row major or column major.


        err_cublas = cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<size_t>(h_lm_var._number_of_pose_params), static_cast<size_t>(1), static_cast<size_t>(h_lm_var._number_of_landmarks),
            &alpha,
            h_lm_var.d_B_C_inv, h_lm_var._number_of_landmarks,     
            h_lm_var.d_g_a, h_lm_var._number_of_landmarks,
            &beta,
            h_lm_var.d_B_C_inv_g_a, h_lm_var._number_of_pose_params
        );

        
        cudaStreamSynchronize(cublas_stream);

        int H_schur_size = h_lm_var._number_of_pose_params * h_lm_var._number_of_pose_params;
        elementwiseSubtractionKernel<<<NUM_BLOCKS(H_schur_size), NUM_THREADS>>>(
            h_lm_var.d_A,
            h_lm_var.d_B_C_inv_B_T,
            h_lm_var.d_H_schur,
            H_schur_size
        );

        elementwiseSubtractionKernel<<<NUM_BLOCKS(h_lm_var._number_of_pose_params), NUM_THREADS>>>(
            h_lm_var.d_g_T,
            h_lm_var.d_B_C_inv_g_a,
            h_lm_var.d_g_schur,
            h_lm_var._number_of_pose_params
        );

        cudaStreamSynchronize(cublas_stream);
        cudaDeviceSynchronize();

        // Compute delta T
        h_lm_var.solve_Eigen();

        // Compute delta alpha
        err_cublas = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<size_t>(h_lm_var._number_of_landmarks), static_cast<size_t>(1), static_cast<size_t>(h_lm_var._number_of_pose_params),
            &alpha,
            h_lm_var.d_B,       h_lm_var._number_of_landmarks,     
            h_lm_var.d_g_schur, h_lm_var._number_of_pose_params,
            &beta,
            h_lm_var.d_B_T_delta_T, h_lm_var._number_of_landmarks
        );

        cudaStreamSynchronize(cublas_stream);
        cudaDeviceSynchronize();

        compute_delta_a<<<NUM_BLOCKS(h_lm_var._number_of_landmarks), NUM_THREADS>>>(d_lm_var);

        cudaDeviceSynchronize();


        
        // Finally, update the estimation
        updateEstimation<<<NUM_BLOCKS(h_lm_var._number_of_poses + h_lm_var._number_of_landmarks), NUM_THREADS>>>(
            d_lm_var,
            inverse_depths.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            step_size
        );

        cudaDeviceSynchronize();

        err = cudaGetLastError ();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
        }
        cublasDestroy(handle);

        h_lm_var.resetMiddleVariables();
        cudaDeviceSynchronize();

        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // std::cout << "Elapsed time :" << duration.count() << " milliseconds" << std::endl;
        
    }

    // Load the new poses back to tensor array
    loadIncrementalPosesBack<<<NUM_BLOCKS(h_lm_var._number_of_poses*16), NUM_THREADS>>>(
        incremental_poses.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        d_lm_var
    );
    cudaDeviceSynchronize();

    
    
    cudaFree(d_lm_var);
    h_lm_var.freeAll();
}