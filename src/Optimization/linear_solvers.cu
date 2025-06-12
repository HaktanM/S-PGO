#include "LMutils.h"
#include <Eigen/Dense>

void LMvariables::solve_Cholesky(){
    // N is the number of rows and columns of matrix d_H
    int N = _number_of_pose_params;

    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    int workspace_size;
    float* d_workspace;
    int* devInfo;

    cudaMalloc(&devInfo, sizeof(int));
    cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, N, d_H_schur, N, &workspace_size);
    cudaMalloc(&d_workspace, workspace_size * sizeof(float));

    // Cholesky factorization: H = L * L^T (in-place on d_H_schur)
    cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, N, d_H_schur, N, d_workspace, workspace_size, devInfo);
    cudaDeviceSynchronize();

    // Check for Cholesky factorization success
    int h_devInfo = 0;
    cudaMemcpy(&h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_devInfo != 0) {
        std::cerr << "Cholesky factorization failed. devInfo = " << h_devInfo << std::endl;
        // You might want to return or handle this case
    }


    // Solve L * L^T * x = g => x stored in d_g
    cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, N, 1, d_H_schur, N, d_g_schur, N, devInfo);
    // d_g_schur now contains ∆x
    cudaDeviceSynchronize();

    cudaMemcpy(&h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_devInfo != 0) {
        std::cerr << "Failed to solve Schur complement = " << h_devInfo << std::endl;
        // You might want to return or handle this case
    }

    // Clean up
    cudaFree(d_workspace);
    cudaFree(devInfo);
    cusolverDnDestroy(handle);
};



void LMvariables::solve_SVD(){
    int N = _number_of_pose_params;

    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // SVD: H = U * S * V^T
    float* d_U;         // N x N
    float* d_VT;        // N x N
    float* d_S;         // N (singular values)
    float* d_work;
    int* devInfo;

    cudaMalloc(&d_U, N * N * sizeof(float));
    cudaMalloc(&d_VT, N * N * sizeof(float));
    cudaMalloc(&d_S, N * sizeof(float));
    cudaMalloc(&devInfo, sizeof(int));

    // Query workspace size
    int lwork = 0;
    cusolverDnSgesvd_bufferSize(handle, N, N, &lwork);
    cudaMalloc(&d_work, lwork * sizeof(float));

    // Copy d_H_schur into a temp matrix because SVD destroys input
    float* d_H_copy;
    cudaMalloc(&d_H_copy, N * N * sizeof(float));
    cudaMemcpy(d_H_copy, d_H_schur, N * N * sizeof(float), cudaMemcpyDeviceToDevice);

    // For gesvd: inputs
    signed char jobu = 'A';  // All columns of U
    signed char jobvt = 'A'; // All rows of VT

    float* d_rwork = nullptr;  // Not needed for single precision
    float* h_g = new float[N];
    float* h_x = new float[N];  // Solution

    cusolverDnSgesvd(
        handle, jobu, jobvt,
        N, N,
        d_H_copy, N,
        d_S,
        d_U, N,
        d_VT, N,
        d_work, lwork,
        d_rwork,
        devInfo
    );
    cudaDeviceSynchronize();

    int h_devInfo;
    cudaMemcpy(&h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_devInfo != 0) {
        std::cerr << "SVD failed. devInfo = " << h_devInfo << std::endl;
        // Handle or exit
    }

    // Bring singular values and g vector to host
    float* h_U = new float[N * N];
    float* h_VT = new float[N * N];
    float* h_S = new float[N];

    cudaMemcpy(h_U, d_U, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_VT, d_VT, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_S, d_S, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, d_g_schur, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Solve using x = V * S^-1 * U^T * g
    // 1. Compute U^T * g
    float* Ut_g = new float[N]();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Ut_g[i] += h_U[j * N + i] * h_g[j];  // U^T * g

    // 2. Scale by 1 / singular value (with threshold)
    float tol = 1e-6f;
    for (int i = 0; i < N; ++i) {
        if (h_S[i] > tol)
            Ut_g[i] /= h_S[i];
        else
            Ut_g[i] = 0.0f;  // Truncate small singular values
    }

    // 3. Multiply V * result
    for (int i = 0; i < N; ++i) {
        h_x[i] = 0.0f;
        for (int j = 0; j < N; ++j)
            h_x[i] += h_VT[j * N + i] * Ut_g[j];  // V * (S^-1 * U^T * g)
    }

    // Copy back solution to d_g_schur (∆x)
    cudaMemcpy(d_g_schur, h_x, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Clean up
    delete[] h_U;
    delete[] h_VT;
    delete[] h_S;
    delete[] h_g;
    delete[] h_x;
    delete[] Ut_g;

    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_S);
    cudaFree(d_work);
    cudaFree(devInfo);
    cudaFree(d_H_copy);

    cusolverDnDestroy(handle);
}



void LMvariables::solve_Eigen() {
    int N = _number_of_pose_params;

    // Allocate host memory
    std::vector<float> h_H_schur(N * N);
    std::vector<float> h_g_schur(N);

    cudaError_t err;

    err = cudaMemcpy(h_H_schur.data(), d_H_schur, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H failed: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaMemcpy(h_g_schur.data(), d_g_schur, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy g failed: " << cudaGetErrorString(err) << std::endl;
    }

    // Map to Eigen types
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> H(h_H_schur.data(), N, N);
    Eigen::Map<Eigen::VectorXf> g(h_g_schur.data(), N);

    // Solve using SVD: H * x = g
    Eigen::VectorXf delta_x = H.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(g);

    // Copy result back to GPU
    cudaMemcpy(d_g_schur, delta_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
}
