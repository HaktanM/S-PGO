#include <iostream>
#include <fstream> 
#include "lie.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>    // cuSOLVER API




struct LMvariables{
public:

    float *_global_left_poses  = NULL;

    float *d_H_T            = NULL;
    float *d_g_T            = NULL;
    
    float *d_H_a            = NULL;
    float *d_g_a            = NULL;

    float *d_B              = NULL;

    float *d_B_C_inv        = NULL;
    float *d_B_C_inv_B_T    = NULL;
    float *d_B_C_inv_g_a    = NULL;

    float *d_H_schur        = NULL;
    float *d_g_schur        = NULL;

    float *d_B_T_delta_T    = NULL;
    float *d_delta_T        = NULL;
    float *d_delta_a        = NULL;

    float *d_T_r_to_l       = NULL;
    float *d_T_l_to_r       = NULL;

    int _number_of_pose_params{0};
    int _number_of_landmarks{0};
    int _measurement_size{0};

    int _measurement_count{0};
    int _number_of_poses{0};

    float _cauchy_constant_square{9.0};

    float _eps{0.0001};   // During the division, _eps is used for numeric stability

    float _lambda_schur{10.0};   // During the division, _eps is used for numeric stability

    float _min_inv_depth{1.0/30.0};
    float _max_inv_depth{1.0/0.2};

    void allocateMemory(int num_of_pose_params, int num_of_landmarks, int measurement_size);

    void solve_Cholesky();
    void solve_SVD();
    void solve_Eigen();

    void resetMiddleVariables();
    void freeAll();


    void H_a_to_txt();
    void g_a_to_txt();
    
    void H_T_to_txt();
    void g_T_to_txt();

    void B_to_txt();
    void B_C_inv_to_txt();
    void B_C_inv_B_T_to_txt();

    void H_schur_to_txt();
    void g_schur_to_txt();
    void delta_pose_to_txt();

    void d_B_T_delta_T_to_txt();
    void d_delta_a_to_txt();
};


__device__ void getRotFromT(const float *T, float *R);
__device__ void compute_del_tn_del_xi(const float *T_state_to_target, const float *t_feat_in_state, float *del_tn_del_xi);
__device__ void computePoseInverse(const float* T, float* T_inv);
__device__ void compute_del_bn_del_tn(float *t_feat_cn, float *del_bn_del_tn);