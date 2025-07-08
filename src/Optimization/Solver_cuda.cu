#include "Solver_cuda.h"
#include "LMutils.h"



torch::Tensor invertSE3(const torch::Tensor& T) {
    // Assumes T is a 4x4 SE(3) matrix
    TORCH_CHECK(T.sizes() == std::vector<int64_t>({4, 4}), "Input must be 4x4");

    // Extract rotation and translation
    torch::Tensor R = T.slice(0, 0, 3).slice(1, 0, 3); // top-left 3x3
    torch::Tensor t = T.slice(0, 0, 3).slice(1, 3, 4); // top-right 3x1

    // Inverse rotation = transpose
    torch::Tensor R_inv = R.transpose(0, 1); // 3x3
    torch::Tensor t_inv = -R_inv.matmul(t);  // 3x1

    // Construct inverse transformation matrix
    torch::Tensor T_inv = torch::eye(4, T.options());
    T_inv.slice(0, 0, 3).slice(1, 0, 3) = R_inv;
    T_inv.slice(0, 0, 3).slice(1, 3, 4) = t_inv;

    return T_inv;
}

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


inline __device__ void getTranspose(
    float* in,
    float* out,
    int height,
    int width
) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int inIdx  = row * width + col;
            int outIdx = col * height + row;
            out[outIdx] = in[inIdx];
        }
    }
}

inline __device__ void applyRigidTransformation(
    float* T_a_to_b,
    float* t_in_a,
    float* t_in_b
) {
    t_in_b[0] = T_a_to_b[0] * t_in_a[0] + T_a_to_b[1] * t_in_a[1] + T_a_to_b[2]  * t_in_a[2] + T_a_to_b[3];
    t_in_b[1] = T_a_to_b[4] * t_in_a[0] + T_a_to_b[5] * t_in_a[1] + T_a_to_b[6]  * t_in_a[2] + T_a_to_b[7];
    t_in_b[2] = T_a_to_b[8] * t_in_a[0] + T_a_to_b[9] * t_in_a[1] + T_a_to_b[10] * t_in_a[2] + T_a_to_b[11];
}

inline __device__ void projection(
    float* K,
    float* vec_in,
    float* vec_out
) 
{
    float bearing[3];

    bearing[0] = vec_in[0] / vec_in[2];
    bearing[1] = vec_in[1] / vec_in[2];
    bearing[2] = 1.0;

    MatrixMultiplication(K, bearing, vec_out, 3, 3, 1);
}

inline __device__ void jacobianOfProjection(
    float *K,
    float* vec,
    float* J
) 
{
    // First ne weed to compute a middle variable
    float del_b_del_vec[9];
    del_b_del_vec[0] = 1.0 / vec[2];
    del_b_del_vec[1] = 0.0;
    del_b_del_vec[2] = - vec[0] / (vec[2]*vec[2]);

    del_b_del_vec[3] = 0.0;
    del_b_del_vec[4] = 1.0 / vec[2];
    del_b_del_vec[5] = - vec[1] / (vec[2]*vec[2]);

    MatrixMultiplication(K, del_b_del_vec, J, 2, 3, 3);
}

inline __device__ void compute_del_tnl_del_xn(
    float* t_feat_in_cnl,
    float* del_tnl_del_xn
) 
{

    // First Row
    del_tnl_del_xn[0] = 0.0;
    del_tnl_del_xn[1] = -t_feat_in_cnl[2];
    del_tnl_del_xn[2] = t_feat_in_cnl[1];

    del_tnl_del_xn[3] = -1.0;
    del_tnl_del_xn[4] = 0.0;
    del_tnl_del_xn[5] = 0.0;

    // Second Row
    del_tnl_del_xn[6]  = t_feat_in_cnl[2];
    del_tnl_del_xn[7]  = 0.0;
    del_tnl_del_xn[8]  = -t_feat_in_cnl[0];

    del_tnl_del_xn[9]  = 0.0;
    del_tnl_del_xn[10] = -1.0;
    del_tnl_del_xn[11] = 0.0;

    // Third Row
    del_tnl_del_xn[12] = -t_feat_in_cnl[1];
    del_tnl_del_xn[13] = t_feat_in_cnl[0];
    del_tnl_del_xn[14] = 0.0;

    del_tnl_del_xn[15] = 0.0;
    del_tnl_del_xn[16] = 0.0;
    del_tnl_del_xn[17] = -1.0;
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
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> T_l_to_r_torch,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> T_r_to_l_torch,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> anchor_frame_ids,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> target_frame_ids,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> feat_glob_ids,
    LMvariables *lm_var
){
    __shared__ float Kl[9];
    __shared__ float Kl_inv[9];
    __shared__ float Kr[9];
    __shared__ float T_l_to_r[16];
    __shared__ float T_r_to_l[16];
    __shared__ int thread_max_limit[1];

    // First load the data which is commonly used by all threads to the shared memory
    if (threadIdx.x == 0) {      
        // Load left camera intrinsics
        Kl[0] = intrinsics[0][0]; // fx
        Kl[1] = 0.0f; 
        Kl[2] = intrinsics[0][2]; // cx

        Kl[3] = 0.0f; 
        Kl[4] = intrinsics[0][1]; // fy
        Kl[5] = intrinsics[0][3]; // cy

        Kl[6] = 0.0f; 
        Kl[7] = 0.0f; 
        Kl[8] = 1.0f;  

        // Load invers of left camera parameters
        Kl_inv[0] = 1.0f / intrinsics[0][0]; //  1 / fx
        Kl_inv[1] = 0.0f; 
        Kl_inv[2] = - intrinsics[0][2] / intrinsics[0][0]; //  -cx / fx

        Kl_inv[3] = 0.0f; 
        Kl_inv[4] = 1.0f / intrinsics[0][1]; //  1 / fy
        Kl_inv[5] = - intrinsics[0][3] / intrinsics[0][1]; //  -cy / fy

        Kl_inv[6] = 0.0f; 
        Kl_inv[7] = 0.0f; 
        Kl_inv[8] = 1.0f; 

        // Load right camera intrinsics
        Kr[0] = intrinsics[1][0]; // fx
        Kr[1] = 0.0f; 
        Kr[2] = intrinsics[1][2]; // cx

        Kr[3] = 0.0f; 
        Kr[4] = intrinsics[1][1]; // fy
        Kr[5] = intrinsics[1][3]; // cy

        Kr[6] = 0.0f; 
        Kr[7] = 0.0f; 
        Kr[8] = 1.0f; 

        // Load extrinsic calibrations
        for(int row_idx=0; row_idx<4; row_idx++){
            for(int col_idx=0; col_idx<4; col_idx++){
                T_l_to_r[row_idx*4 + col_idx] = T_l_to_r_torch[row_idx][col_idx];
            }
        }

        for(int row_idx=0; row_idx<4; row_idx++){
            for(int col_idx=0; col_idx<4; col_idx++){
                T_r_to_l[row_idx*4 + col_idx] = T_r_to_l_torch[row_idx][col_idx];
            }
        }

        // Get max thread index
        thread_max_limit[0] = lm_var->_measurement_count * 2;
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

    // Get the observation in the anchor
    float p_in_anchor[3] = { observations[anchor_idx][feat_idx][0][0], observations[anchor_idx][feat_idx][0][1], observations[anchor_idx][feat_idx][0][2] };

    // Get the estimation parameters
    float T_ca_to_g[16];  loadPose(poses, anchor_idx, T_ca_to_g);
    float T_cnl_to_g[16]; loadPose(poses, target_idx, T_cnl_to_g);
    float alpha = inverse_depths[feat_idx];

    // We will also need T_g_to_cnl
    float T_g_to_cnl[16]; InvertPose4x4(T_cnl_to_g, T_g_to_cnl);

    // Compute position of feature in anchor camera frame
    float t_feat_in_ca[3]; MatrixMultiplication(Kl_inv, p_in_anchor, t_feat_in_ca, 3, 3, 1);
    t_feat_in_ca[0] = t_feat_in_ca[0] / alpha;
    t_feat_in_ca[1] = t_feat_in_ca[1] / alpha;
    t_feat_in_ca[2] = t_feat_in_ca[2] / alpha;

    // Compute T_ca_to_cnl
    float T_ca_to_cnl[16];
    MatrixMultiplication(T_g_to_cnl, T_ca_to_g, T_ca_to_cnl, 4, 4, 4);

    // Compute position of the feature in target left camera frame
    float t_feat_in_cnl[3]; applyRigidTransformation(T_ca_to_cnl, t_feat_in_ca, t_feat_in_cnl);

    if(cam_idx == 0) // The observation belongs to left camera
    {

        // Compute the residual first
        float pnl_estimated[3];
        projection(Kl, t_feat_in_cnl, pnl_estimated);

        float residual[2];
        residual[0] = observations[target_idx][feat_idx][cam_idx][0] - pnl_estimated[0];
        residual[1] = observations[target_idx][feat_idx][cam_idx][1] - pnl_estimated[1];
        
        // Compute del_pnl_del_tnl in the first place
        float del_pnl_del_tnl[6]; jacobianOfProjection(Kl, t_feat_in_cnl, del_pnl_del_tnl);
        
        if(target_idx>anchor_idx){  // Pose is updated only if the measurement belongs to a new frame
            // Compute del_tnl_del_xn
            float del_tnl_del_xn[18]; compute_del_tnl_del_xn(t_feat_in_cnl, del_tnl_del_xn);

            // Compute del_pnl_del_xn
            float del_pnl_del_xn[12];
            MatrixMultiplication(del_pnl_del_tnl, del_tnl_del_xn, del_pnl_del_xn, 2, 3, 6);

            // Compute H_TT and g_TT
            float del_pnl_del_xn_transpose[12]; getTranspose(del_pnl_del_xn, del_pnl_del_xn_transpose, 2, 6);
            float h_T[36]; MatrixMultiplication(del_pnl_del_xn_transpose, del_pnl_del_xn, h_T, 6, 2, 6);
            float g_T[6];  MatrixMultiplication(del_pnl_del_xn_transpose, residual, g_T, 6, 2, 1);
            
            for(int row_idx=0; row_idx<6; row_idx++){
                // First fil the H_T
                for(int col_idx=0; col_idx<6; col_idx++){
                    int h_T_idx = row_idx * 6 + col_idx;
                    int H_T_idx = 6 * target_idx * lm_var->_number_of_pose_params 
                                + 6 * target_idx 
                                + row_idx * lm_var->_number_of_pose_params 
                                + col_idx;

                                atomicAdd(&(lm_var->d_H_T[H_T_idx]), h_T[h_T_idx]);
                }

                // Fill g_T
                int g_T_idx = 6 * target_idx + row_idx;
                atomicAdd(&(lm_var->d_g_T[g_T_idx]), g_T[row_idx]);
            }
        }
        
        
    }
    else // The observation belongs to left camera
    {
        // Compute the pose of the landmark in the rigth camera first
        float t_feat_in_cnr[3]; applyRigidTransformation(T_l_to_r, t_feat_in_cnl, t_feat_in_cnr); 

        // Compute the residual
        float pnr_estimated[3];
        projection(Kr, t_feat_in_cnr, pnr_estimated);

        float residual[2];
        residual[0] = observations[target_idx][feat_idx][cam_idx][0] - pnr_estimated[0];
        residual[1] = observations[target_idx][feat_idx][cam_idx][1] - pnr_estimated[1];

        // Compute del_pnr_del_tnr in the first place
        float del_pnr_del_tnr[6]; jacobianOfProjection(Kr, t_feat_in_cnr, del_pnr_del_tnr);

        if(target_idx>anchor_idx){  // Pose is updated only if the measurement belongs to a new frame
            // Compute del_tnr_del_xn
            float del_tnr_del_tnl[9]; getRotFromT(T_l_to_r, del_tnr_del_tnl);
            float del_tnl_del_xn[18]; compute_del_tnl_del_xn(t_feat_in_cnl, del_tnl_del_xn);
            float del_tnr_del_xn[18]; MatrixMultiplication(del_tnr_del_tnl, del_tnl_del_xn, del_tnr_del_xn, 3, 3, 6);

            // Compute del_pnr_del_xn
            float del_pnr_del_xn[12];
            MatrixMultiplication(del_pnr_del_tnr, del_tnr_del_xn, del_pnr_del_xn, 2, 3, 6);

            // Compute H_TT and g_TT
            float del_pnr_del_xn_transpose[12]; getTranspose(del_pnr_del_xn, del_pnr_del_xn_transpose, 2, 6);
            float h_T[36]; MatrixMultiplication(del_pnr_del_xn_transpose, del_pnr_del_xn, h_T, 6, 2, 6);
            float g_T[6];  MatrixMultiplication(del_pnr_del_xn_transpose, residual, g_T, 6, 2, 1);

            if(feat_idx==0 && target_idx==1){
                printf("g_T : \n");
                for(int row_idx = 0; row_idx<6; row_idx++){
                    printf("%f, ", g_T[row_idx]);
                }
                printf("\n");
                printf("\n");
                
                printf("residual : \n");
                for(int row_idx = 0; row_idx<2; row_idx++){
                    printf("%f, ", residual[row_idx]);
                }
                printf("\n");
            } 
            
            for(int row_idx=0; row_idx<6; row_idx++){
                // First fil the H_T
                for(int col_idx=0; col_idx<6; col_idx++){
                    int h_T_idx = row_idx * 6 + col_idx;
                    int H_T_idx = 6 * target_idx * lm_var->_number_of_pose_params 
                                + 6 * target_idx 
                                + row_idx * lm_var->_number_of_pose_params 
                                + col_idx;

                    atomicAdd(&(lm_var->d_H_T[H_T_idx]), h_T[h_T_idx]);
                }

                // Fill g_T
                int g_T_idx = 6 * target_idx + row_idx;
                atomicAdd(&(lm_var->d_g_T[g_T_idx]), g_T[row_idx]);
                printf("row_idx : %d, target_idx : %d, lm_var->d_g_T[g_T_idx] : %f\n", row_idx, target_idx, lm_var->d_g_T[g_T_idx]);
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

    // Also compute the inverse of the extrinsics which will be used
    torch::Tensor T_l_to_r = invertSE3(T_r_to_l);

    for(int it_idx=0; it_idx<iterations; it_idx++){
        // Compute the A, B, C and g_a, g_T
        LevenbergMarquardt<<<NUM_BLOCKS(h_lm_var._measurement_count * 2), NUM_THREADS>>>(
            observations.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            poses.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            inverse_depths.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            T_l_to_r.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            T_r_to_l.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            anchor_frame_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            target_frame_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            feat_glob_id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            d_lm_var
        );

        // Wait untill kernel is completed
        cudaDeviceSynchronize();

        // print the Hessian for varification
        h_lm_var.H_T_to_txt();
        h_lm_var.g_T_to_txt();
    }

    cudaDeviceSynchronize();
    cudaFree(d_lm_var);
    h_lm_var.freeAll();
}