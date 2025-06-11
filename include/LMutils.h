#include <cuda_runtime.h>
#include <iostream>
#include <fstream> 
#include "lie.h"

struct LMvariables{
public:
    float* d_J_T            = NULL;
    float *d_r              = NULL;

    float *d_A              = NULL;
    float *d_g_T            = NULL;
    
    float *d_C              = NULL;
    float *d_g_a            = NULL;

    float *d_B              = NULL;

    float *d_B_C_inv        = NULL;
    float *d_B_C_inv_B_T    = NULL;
    float *d_B_C_inv_g_a    = NULL;

    float *d_H_schur        = NULL;
    float *d_g_schur        = NULL;

    int _number_of_pose_params{0};
    int _number_of_landmarks{0};
    int _measurement_size{0};

    int _measurement_count{0};
    int _num_of_poses{0};

    float _cauchy_constant_square{9.0};

    float _eps{0.0001};   // During the division, _eps is used for numeric stability

    void allocateMemory(int num_of_pose_params, int num_of_landmarks, int measurement_size);

    void freeAll();

    
    void J_T_to_txt();
    void r_to_txt();

    void C_to_txt();
    void g_a_to_txt();
    
    void A_to_txt();
    void g_T_to_txt();

    void B_to_txt();
    void B_C_inv_to_txt();
    void B_C_inv_B_T_to_txt();

    void H_schur_to_txt();
    void g_schur_to_txt();
};


__device__ void getRotFromT(const float *T, float *R);
__device__ void compute_del_tn_del_xi(const float *T_state_to_target, const float *t_feat_in_state, float *del_tn_del_xi);
__device__ void computePoseInverse(const float* T, float* T_inv);
__device__ void compute_del_bn_del_tn(float *t_feat_cn, float *del_bn_del_tn);