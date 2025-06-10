#include "LMutils.h"


void LMvariables::allocateMemory(int num_of_poses, int num_of_landmarks, int measurement_count){

    _number_of_pose_params = num_of_poses * 6;
    _num_of_landmarks      = num_of_landmarks;
    _measurement_size      = measurement_count * 4;

    _measurement_count     = measurement_count;
    _num_of_poses          = num_of_poses;



    cudaMalloc((void**)&d_J_T, _measurement_size * _number_of_pose_params * sizeof(float));
    cudaMemset(d_J_T, 0,       _measurement_size * _number_of_pose_params * sizeof(float)); 

    cudaMalloc((void**)&d_r,   _measurement_size * sizeof(float));
    cudaMemset(d_r, 0,         _measurement_size * sizeof(float)); 

    cudaMalloc((void**)&d_A,   _number_of_pose_params * _number_of_pose_params * sizeof(float));
    cudaMemset(d_A, 0,         _number_of_pose_params * _number_of_pose_params * sizeof(float)); 

    cudaMalloc((void**)&d_g_T, _number_of_pose_params * sizeof(float));
    cudaMemset(d_g_T, 0,       _number_of_pose_params * sizeof(float)); 
       
    cudaMalloc((void**)&d_C,     num_of_landmarks * sizeof(float));
    cudaMemset(d_C, 0,           num_of_landmarks * sizeof(float)); 

    cudaMalloc((void**)&d_C_inv, num_of_landmarks * sizeof(float));
    cudaMemset(d_C_inv, 0,       num_of_landmarks * sizeof(float)); 


    cudaMalloc((void**)&d_g_a,   num_of_landmarks * sizeof(float));
    cudaMemset(d_g_a, 0,         num_of_landmarks * sizeof(float)); 

    cudaMalloc((void**)&d_B, _number_of_pose_params * num_of_landmarks * sizeof(float));
    cudaMemset(d_B, 0,       _number_of_pose_params * num_of_landmarks * sizeof(float)); 
};

void LMvariables::freeAll(){
    cudaFree(d_J_T);
    cudaFree(d_r);
    
    cudaFree(d_A);
    cudaFree(d_g_T);

    cudaFree(d_C);
    cudaFree(d_C_inv);
    cudaFree(d_g_a);

    cudaFree(d_B);

    _number_of_pose_params = 0;
    _num_of_landmarks      = 0;
    _measurement_size      = 0;
    _measurement_count     = 0;
    _num_of_poses          = 0;
};


void LMvariables::J_T_to_txt(){
    // First load the matrix C into CPU
    float *h_J_T;
    h_J_T =(float *) malloc( _measurement_size * _number_of_pose_params * sizeof(float));
    cudaMemcpy(h_J_T, d_J_T, _measurement_size * _number_of_pose_params * sizeof(float), cudaMemcpyDeviceToHost);

    // Create and open a text file
    std::ofstream txt_file("J_T.txt");

    // Write the C matrix into the txt
    for(int row_idx=0; row_idx<_measurement_size; row_idx++){
        for(int col_idx=0; col_idx<_number_of_pose_params; col_idx++){
            txt_file << h_J_T[row_idx * _number_of_pose_params + col_idx];
            if(col_idx<(_number_of_pose_params-1)){
                txt_file << ", ";
            }
        }
        txt_file << std::endl;
    }
    

    // Close the file
    txt_file.close();

    free(h_J_T);
}

void LMvariables::r_to_txt(){
    // First load the matrix C into CPU
    float *h_r;
    h_r =(float *) malloc( _measurement_size * sizeof(float));
    cudaMemcpy(h_r, d_r,   _measurement_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Create and open a text file
    std::ofstream txt_file("r.txt");

    // Write the C matrix into the txt
    for(int row_idx=0; row_idx<_measurement_size; row_idx++){
        txt_file << h_r[row_idx];
        if(row_idx<(_measurement_size-1)){
            txt_file << ", ";
        }
    }
    

    // Close the file
    txt_file.close();

    free(h_r);
}

void LMvariables::C_to_txt(){
    // First load the matrix C into CPU
    float *h_C;
    h_C =(float *) malloc( _num_of_landmarks * sizeof(float));
    cudaMemcpy(h_C, d_C, _num_of_landmarks * sizeof(float), cudaMemcpyDeviceToHost);

    // Create and open a text file
    std::ofstream txt_file("C.txt");

    // Write the C matrix into the txt
    for(int idx=0; idx<_num_of_landmarks; idx++){
        txt_file << h_C[idx];
        if(idx<(_num_of_landmarks-1)){
            txt_file << ", ";
        }
    }
    txt_file << std::endl;

    // Close the file
    txt_file.close();

    free(h_C);

};


void LMvariables::C_inv_to_txt(){
    // First load the matrix C into CPU
    float *h_C_inv;
    h_C_inv =  (float *) malloc( _num_of_landmarks * sizeof(float));
    cudaMemcpy(h_C_inv, d_C_inv, _num_of_landmarks * sizeof(float), cudaMemcpyDeviceToHost);

    // Create and open a text file
    std::ofstream txt_file("C_inv.txt");

    // Write the C matrix into the txt
    for(int idx=0; idx<_num_of_landmarks; idx++){
        txt_file << h_C_inv[idx];
        if(idx<(_num_of_landmarks-1)){
            txt_file << ", ";
        }
    }
    txt_file << std::endl;

    // Close the file
    txt_file.close();

    free(h_C_inv);

};

void LMvariables::g_a_to_txt(){
    // First load the matrix C into CPU
    float *h_g_a;
    h_g_a =(float *) malloc( _num_of_landmarks * sizeof(float));
    cudaMemcpy(h_g_a, d_g_a, _num_of_landmarks * sizeof(float), cudaMemcpyDeviceToHost);

    // Create and open a text file
    std::ofstream txt_file("g_a.txt");

    // Write the C matrix into the txt
    for(int row_idx=0; row_idx<_num_of_landmarks; row_idx++){
        txt_file << h_g_a[row_idx];
        if(row_idx<(_num_of_landmarks-1)){
            txt_file << ", ";
        }
    }
    

    // Close the file
    txt_file.close();

    free(h_g_a);
}





void LMvariables::A_to_txt(){
    // First load the matrix C into CPU
    float *h_A;
    h_A =(float *) malloc( _number_of_pose_params * _number_of_pose_params * sizeof(float));
    cudaMemcpy(h_A, d_A,   _number_of_pose_params * _number_of_pose_params * sizeof(float), cudaMemcpyDeviceToHost);

    // Create and open a text file
    std::ofstream txt_file("A.txt");

    // Write the C matrix into the txt
    for(int row_idx=0; row_idx<_number_of_pose_params; row_idx++){
        for(int col_idx=0; col_idx<_number_of_pose_params; col_idx++){
            txt_file << h_A[row_idx * _number_of_pose_params + col_idx];
            if(col_idx<(_number_of_pose_params-1)){
                txt_file << ", ";
            }
        }
        txt_file << std::endl;
    }
    

    // Close the file
    txt_file.close();

    free(h_A);
}

void LMvariables::g_T_to_txt(){
    // First load the matrix C into CPU
    float *h_g_T;
    h_g_T =(float *) malloc( _number_of_pose_params * sizeof(float));
    cudaMemcpy(h_g_T, d_g_T, _number_of_pose_params * sizeof(float), cudaMemcpyDeviceToHost);

    // Create and open a text file
    std::ofstream txt_file("g_T.txt");

    // Write the C matrix into the txt
    for(int row_idx=0; row_idx<_number_of_pose_params; row_idx++){
        txt_file << h_g_T[row_idx];
        if(row_idx<(_number_of_pose_params-1)){
            txt_file << ", ";
        }
    }
    

    // Close the file
    txt_file.close();

    free(h_g_T);
}

void LMvariables::B_to_txt(){
    // First load the matrix C into CPU
    printf("CP1\n");

    float *h_B;
    h_B =(float *) malloc( _number_of_pose_params * _num_of_landmarks * sizeof(float));
    cudaMemcpy(h_B, d_B,   _number_of_pose_params * _num_of_landmarks * sizeof(float), cudaMemcpyDeviceToHost);

    // Create and open a text file
    std::ofstream txt_file("B.txt");

    // Write the C matrix into the txt
    for(int row_idx=0; row_idx<_number_of_pose_params; row_idx++){
        for(int col_idx=0; col_idx<_num_of_landmarks; col_idx++){
            txt_file << h_B[row_idx * _num_of_landmarks + col_idx];
            if(col_idx<(_num_of_landmarks-1)){
                txt_file << ", ";
            }
        }
        txt_file << std::endl;
    }
    

    // Close the file
    txt_file.close();

    free(h_B);
}


__device__ void getRotFromT(const float *T, float *R){
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

__device__ void compute_del_tn_del_xi(const float *T_state_to_target, const float *t_feat_in_state, float *del_tn_del_xi){
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

__device__ void computePoseInverse(const float* T, float* T_inv) {
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


__device__ void compute_del_bn_del_tn(float *t_feat_cn, float *del_bn_del_tn){
    del_bn_del_tn[0] = 1.0f / t_feat_cn[2];
    del_bn_del_tn[1] = 0.0f;
    del_bn_del_tn[2] = -t_feat_cn[0] / (t_feat_cn[2]*t_feat_cn[2]);

    del_bn_del_tn[3] = 0.0f;
    del_bn_del_tn[4] = 1.0f / t_feat_cn[2];
    del_bn_del_tn[5] = -t_feat_cn[1] / (t_feat_cn[2]*t_feat_cn[2]);

    del_bn_del_tn[6] = 0.0f;
    del_bn_del_tn[7] = 0.0f;
    del_bn_del_tn[8] = 0.0f;
}
