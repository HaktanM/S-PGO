#include "test_utils.h"



void TestExpSE3_h(const float *h_xi, float *h_T) {

    // Allocate memory from the device
    float *d_xi, *d_T;
    cudaMalloc(&d_xi, 6 * sizeof(float));
    cudaMalloc(&d_T, 16 * sizeof(float));
    
    // Copy vector psi vector to device
    cudaMemcpy(d_xi, h_xi, 6 * sizeof(float), cudaMemcpyHostToDevice);

    TestExpSE3_d(d_xi, d_T);

    // Copy rotation matrix to host
    cudaMemcpy(h_T, d_T, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_xi);
    cudaFree(d_T);
}

void TestLogSE3_h(const float *h_T, float *h_xi) {
    // Allocate memory from the device
    float *d_xi, *d_T;
    cudaMalloc(&d_xi, 6 * sizeof(float));
    cudaMalloc(&d_T, 16 * sizeof(float));
    
    // Copy vector psi vector to device
    cudaMemcpy(d_T, h_T, 16 * sizeof(float), cudaMemcpyHostToDevice);

    TestLogSE3_d(d_T, d_xi);

    // Copy rotation matrix to host
    cudaMemcpy(h_xi, d_xi, 6 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_xi);
    cudaFree(d_T);
}



void TestJacobiansSE3_h(const float *h_xi, float *h_Jl, float *h_Jr) {

    // Allocate memory from the device
    float *d_xi, *d_Jl, *d_Jr;
    cudaMalloc(&d_xi, 6 * sizeof(float));
    cudaMalloc(&d_Jl, 36 * sizeof(float));
    cudaMalloc(&d_Jr, 36 * sizeof(float));
    
    // Copy vector psi vector to device
    cudaMemcpy(d_xi, h_xi, 6 * sizeof(float), cudaMemcpyHostToDevice);

    TestJacobiansSE3_d(d_xi, d_Jl, d_Jr);

    // Copy Jacobians to CPU
    cudaMemcpy(h_Jl, d_Jl, 36 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Jr, d_Jr, 36 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_xi);
    cudaFree(d_Jl);
    cudaFree(d_Jr);
}



bool SE3_Reconstruction_Test(const cv::Mat xi){
    // xi to float array
    float* h_xi = (float*)xi.data;
    float h_T[16];

    // Call CUDA kernel to obtain rigid transformation
    TestExpSE3_h(h_xi, h_T);
    cudaDeviceSynchronize();
    cv::Mat T(4, 4, CV_32F, h_T);  // Store rigid transformation in a matrix

    /////////// Reconstruction test ///////////
    // Get the reconstructed xi
    float h_xi_reconstructed[6];
    TestLogSE3_h(h_T, h_xi_reconstructed);
    cudaDeviceSynchronize();

    // Using the reconstructed xi, obtain T_reconstructed
    float h_T_reconstructed[16];
    TestExpSE3_h(h_xi_reconstructed, h_T_reconstructed);
    cudaDeviceSynchronize();
    cv::Mat T_reconstructed(4, 4, CV_32F, h_T_reconstructed);  // Store rigid transformation in a matrix

    // Compute the discreapancy
    cv::Mat T_disc = T * T_reconstructed.inv();
    T_disc.convertTo(T_disc, CV_32F);  // Make sure that the data format is float

    float h_xi_disc[6];
    float* h_T_disc = (float*)T_disc.data;

    TestLogSE3_h(h_T_disc, h_xi_disc);
    cudaDeviceSynchronize();
    cv::Mat xi_disc(6, 1, CV_32F, h_xi_disc);  // Store rigid transformation in a matrix
    
    // Check the reconstruction error
    if(cv::norm(xi_disc, cv::NORM_INF) > 1e-4){
        std::cout << "xi_disc : " << std::endl << xi_disc << std::endl;
        return false;
    }
    return true;
}

bool SE3_Jacobian_Test(const cv::Mat xi){
    // xi to float array
    float* h_xi = (float*)xi.data;
    float h_T[16];

    // Call CUDA kernel to obtain rigid transformation
    TestExpSE3_h(h_xi, h_T);
    cudaDeviceSynchronize();
    cv::Mat T(4, 4, CV_32F, h_T);  // Store rigid transformation in a matrix

    /////////// Jacobian test ///////////
    cv::Mat Jl_numeric = cv::Mat::zeros(6, 6, CV_64F);
    cv::Mat Jr_numeric = cv::Mat::zeros(6, 6, CV_64F);

    float eps = 1e-4;

    for(int idx=0; idx<6; ++idx){
        float h_xi_perturbed[6];
        std::memcpy(h_xi_perturbed, h_xi, 6 * sizeof(float)); // Copy memory
        h_xi_perturbed[idx] += eps;

        // Compute the corresponding rigid transformation
        float h_T_perturbed[16];
        TestExpSE3_h(h_xi_perturbed, h_T_perturbed);
        cv::Mat T_perturbed(4, 4, CV_32F, h_T_perturbed);  // Store rigid transformation in a matrix

        // Compute the Jacobian numerically
        cv::Mat T_left  = T_perturbed * T.inv();
        cv::Mat T_right = T.inv() * T_perturbed;
        float *h_T_left  = (float*)T_left.data; 
        float *h_T_right = (float*)T_right.data;

        float h_Jl_col[6], h_Jr_col[6];
        TestLogSE3_h(h_T_left, h_Jl_col);
        TestLogSE3_h(h_T_right, h_Jr_col);

        for (int col_idx = 0; col_idx < 6; ++col_idx) {
            Jl_numeric.at<double>(col_idx, idx) = h_Jl_col[col_idx] / eps;  // Divide by eps for finite differences
            Jr_numeric.at<double>(col_idx, idx) = h_Jr_col[col_idx] / eps;
        }
    }
    // Compute the Jacobian analytically
    float h_Jl_analytic[36], h_Jr_analytic[36];
    TestJacobiansSE3_h(h_xi, h_Jl_analytic, h_Jr_analytic);
    cv::Mat Jl_analytic(6, 6, CV_32F, h_Jl_analytic);
    cv::Mat Jr_analytic(6, 6, CV_32F, h_Jr_analytic);

    std::cout << "Jl_numeric :" << std::endl << Jl_numeric << std::endl;
    std::cout << "Jl_analytic :" << std::endl << Jl_analytic << std::endl;
    std::cout << "Jr_analytic :" << std::endl << Jr_analytic << std::endl;

    Jl_analytic.convertTo(Jl_analytic, CV_32F);
    Jl_numeric.convertTo(Jl_numeric, CV_32F);
    std::cout << std::endl << "Jl_analytic - Jl_numeric :" << std::endl << Jl_analytic-Jl_numeric << std::endl;

    return false;

}




// Multiple tests to check if the implementation of ExpSO3 is correct.
bool MultipleTestsSE3() {
    cv::RNG rng(cv::getTickCount());  // Seed RNG
    std::cout << "Testing if the implementation of ExpSE3 is correct, please wait..." << std::endl;
    for (int i = 0; i < 100000; i++) {  // Test 10 random vectors
        // std::cout << "Test " << i + 1 << ":\n";
        

        // Step 1: Generate a random 3×1 vector using OpenCV
        cv::Mat psi = cv::Mat(3, 1, CV_32F);  
        rng.fill(psi, cv::RNG::UNIFORM, -1.0f, 1.0f);  // Random values in [-1, 1]
        float norm = cv::norm(psi);
        float modifiedNorm = norm;  // Start with the original norm

        // Arrange the norm of the vector
        float randomChance = rng.uniform(0.0f, 1.0f);
   
        if (randomChance < 0.1) {
            modifiedNorm = (float)M_PI;
            std::cout << "i: " << i << ": Norm is set to be PI" << std::endl;
        } else if (randomChance < 0.2) {
            modifiedNorm = (float)(M_PI + rng.uniform(0.1f, 1.0f));  
            std::cout << "i: " << i << ": Norm is set to be larger than PI" << std::endl;
        } else if (randomChance < 0.25) {
            std::cout << "i: " << i << ": Norm is set to be 0" << std::endl;
            modifiedNorm = 0.0f;
        } else if (randomChance < 0.30) {
            std::cout << "i: " << i << ": Norm is set to be close to 0" << std::endl;
            modifiedNorm = (float)(rng.uniform(1e-10, 1e-8));
        } else{
            std::cout << "i: " << i << std::endl;
        }

        psi = psi * (modifiedNorm / norm);

        // Also sample the rho
        cv::Mat rho(3, 1, CV_32F);
        rng.fill(rho, cv::RNG::UNIFORM, -10.0f, 10.0f);

        // Finally, obtain
        cv::Mat xi;
        cv::vconcat(psi, rho, xi);
        xi.convertTo(xi, CV_32F);
    
        // Step 2: Allocate memory for CUDA computation
        bool reconstruction_SUCCESS = SE3_Reconstruction_Test(xi);
        if(!reconstruction_SUCCESS){
            std::cout << "Reconstruction test has failed..." << std::endl;
            return false;
        }

        bool Jacobians_SUCCESS = SE3_Jacobian_Test(xi);
        if(!Jacobians_SUCCESS){
            std::cout << "Jacobians test has failed..." << std::endl;
            return false;
        }
    }

    return true;
}