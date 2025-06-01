#include "test_utils.h"

void TestExpSO3_h(const float *h_psi, float *h_R) {

    // Allocate memory from the device
    float *d_psi, *d_R;
    cudaMalloc(&d_psi, 3 * sizeof(float));
    cudaMalloc(&d_R, 9 * sizeof(float));
    
    // Copy vector psi vector to device
    cudaMemcpy(d_psi, h_psi, 3 * sizeof(float), cudaMemcpyHostToDevice);

    TestExpSO3_d(d_psi, d_R);

    // Copy rotation matrix to host
    cudaMemcpy(h_R, d_R, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_psi);
    cudaFree(d_R);
}




// Multiple tests to check if the implementation of ExpSO3 is correct.
bool MultipleTestsExpSO3() {
    cv::RNG rng(cv::getTickCount());  // Seed RNG
    std::cout << "Testing if the implementation of ExpSO3 is correct, please wait..." << std::endl;
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
            // std::cout << "i: " << i << ": Norm is set to be PI" << std::endl;
        } else if (randomChance < 0.2) {
            modifiedNorm = (float)(M_PI + rng.uniform(0.1f, 1.0f));  
            // std::cout << "i: " << i << ": Norm is set to be larger than PI" << std::endl;
        } else if (randomChance < 0.25) {
            // std::cout << "i: " << i << ": Norm is set to be 0" << std::endl;
            modifiedNorm = 0.0f;
        } else if (randomChance < 0.30) {
            // std::cout << "i: " << i << ": Norm is set to be close to 0" << std::endl;
            modifiedNorm = (float)(rng.uniform(1e-10, 1e-8));
        }

        psi = psi * (modifiedNorm / norm);

        // std::cout << "Random Input Vector (psi):\n" << psi << "\n";
    
        // Step 2: Allocate memory for CUDA computation
        float h_vec[3] = {psi.at<float>(0, 0), psi.at<float>(1, 0), psi.at<float>(2, 0)};
        float h_R[9];  // 3x3 Rotation Matrix

        // Step 3: Compute Rotation Matrix using OpenCV's Rodrigues Formula
        cv::Mat R_cv;
        cv::Rodrigues(psi, R_cv);  // Convert axis-angle to rotation matrix

        // Step 4: Call CUDA Kernel
        TestExpSO3_h(h_vec, h_R);
        cudaDeviceSynchronize();


        // Step 5: Compare matrices
        if (compareMatrices(h_R, R_cv)) {
            // std::cout << "Test Passed: CUDA and OpenCV matrices match.\n";
        } else {
            std::cout << "Test Failed: CUDA and OpenCV matrices do not match.\n";
            return false;
        }
    }

    return true;
}