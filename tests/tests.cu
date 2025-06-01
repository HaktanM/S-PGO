#include "lie.h"


// ---------------- Test Utils for ExpSO3 ----------------
__global__ void ExpSO3Kernel(const float *psi, float *R) {
    ExpSO3(psi, R);
}
void TestExpSO3_d(const float *d_psi, float *d_R) {
    // Compute the rotation matrix
    ExpSO3Kernel<<<1, 1>>>(d_psi, d_R); // Launch the kernel
    cudaDeviceSynchronize(); // Ensure the kernel execution is complete
}
// ------------------------------------------------


// ---------------- Test Utils for LogSO3 ----------------
__global__ void LogSO3Kernel(const float *R, float *psi) {
    LogSO3(R, psi);
}

void TestLogSO3_d(const float *d_R, float *d_psi) {
    LogSO3Kernel<<<1, 1>>>(d_R, d_psi); // Launch the kernel
    cudaDeviceSynchronize(); // Ensure the kernel execution is complete
}
// ------------------------------------------------

// ---------------- Test Utils for JacobiansSO3Kernel ----------------
__global__ void JacobiansSO3Kernel(const float *psi, float *Jl, float *Jr) {
    if (threadIdx.x == 0){
        LeftJacobianSO3(psi, Jl);
    } else{
        RightJacobianSO3(psi, Jr);
    }
    
}

void TestJacobiansSO3_d(const float *d_psi, float *d_Jl, float *d_Jr) {
    JacobiansSO3Kernel<<<1, 2>>>(d_psi, d_Jl, d_Jr); // Launch the kernel
    cudaDeviceSynchronize(); // Ensure the kernel execution is complete
}
// ------------------------------------------------



// ---------------- Test Utils for Inverse JacobiansSO3Kernel ----------------
__global__ void InvJacobiansSO3Kernel(const float *psi, float *Jl_inv, float *Jr_inv) {
    if (threadIdx.x == 0){
        InvLeftJacobianSO3(psi, Jl_inv);
    } else{
        InvRightJacobianSO3(psi, Jr_inv);
    }
    
}

void TestInvJacobiansSO3_d(const float *d_psi, float *d_Jl_inv, float *d_Jr_inv) {
    InvJacobiansSO3Kernel<<<1, 2>>>(d_psi, d_Jl_inv, d_Jr_inv); // Launch the kernel
    cudaDeviceSynchronize(); // Ensure the kernel execution is complete
}
// ------------------------------------------------


////////////////////////// SE3 //////////////////////////

// ---------------- Test Utils for ExpSE3 ----------------
__global__ void ExpSE3Kernel(const float *xi, float *T) {
    ExpSE3(xi, T);
}
void TestExpSE3_d(const float *d_xi, float *d_T) {
    ExpSE3Kernel<<<1, 1>>>(d_xi, d_T); // Launch the kernel
    cudaDeviceSynchronize(); // Ensure the kernel execution is complete
}
// ------------------------------------------------


// ---------------- Test Utils for LogSE3 ----------------
__global__ void LogSE3Kernel(const float *T, float *xi) {
    LogSE3(T, xi);
}
void TestLogSE3_d(const float *d_T, float *d_xi) {
    LogSE3Kernel<<<1, 1>>>(d_T, d_xi); // Launch the kernel
    cudaDeviceSynchronize(); // Ensure the kernel execution is complete
}
// ------------------------------------------------

// ---------------- Test Utils for Jacobians SE3 ----------------
__global__ void JacobiansSE3Kernel(const float *xi, float *Jl, float *Jr) {
    if (threadIdx.x == 0){
        LeftJacobianSE3(xi, Jl);
    } else{
        RightJacobianSE3(xi, Jr);
    }
}
void TestJacobiansSE3_d(const float *d_xi, float *d_Jl, float *d_Jr) {
    JacobiansSE3Kernel<<<1, 2>>>(d_xi, d_Jl, d_Jr); 
    cudaDeviceSynchronize(); // Ensure the kernel execution is complete
}
// ------------------------------------------------