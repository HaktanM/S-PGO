#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#define NUMERICAL_TOLERANCE 1e-4f

#include <cuda_runtime.h>
#include <iostream>

#include <assert.h>
#include <cmath> 

// General Utils
__device__ void MatrixMultiplication(const float *A, const float *B, float *C, int HEIGHT, int HIDDEN, int WIDTH);
__device__ void InvertPose4x4(const float* input, float* output);
__device__ void skew(const float *s, float *S);

// SO3 tools
__device__ void ExpSO3(const float *psi, float *R);
__device__ void LogSO3(const float *R, float *psi);
__device__ void LeftJacobianSO3(const float *psi, float *Jl);
__device__ void RightJacobianSO3(const float *psi, float *Jr);
__device__ void InvLeftJacobianSO3(const float *psi, float *Jr_inv);
__device__ void InvRightJacobianSO3(const float *psi, float *Jr_inv);

// SE3 tools
__device__ void ExpSE3(const float *xi, float *T);
__device__ void LogSE3(const float *T, float *xi);
__device__ void LeftJacobianSE3(const float *psi, float *Jl);
__device__ void RightJacobianSE3(const float *psi, float *Jr);

///////// Test Tools /////////
// SO3
void TestExpSO3_d(const float *h_psi, float *h_R);
void TestLogSO3_d(const float *d_R, float *d_psi);
void TestJacobiansSO3_d(const float *d_psi, float *d_Jl, float *d_Jr);
void TestInvJacobiansSO3_d(const float *d_psi, float *d_Jl_inv, float *d_Jr_inv);
// SE3
void TestExpSE3_d(const float *d_xi, float *d_T);
void TestLogSE3_d(const float *d_T, float *d_xi);
void TestJacobiansSE3_d(const float *d_xi, float *d_Jl, float *d_Jr);

#endif