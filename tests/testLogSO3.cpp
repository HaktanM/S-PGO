#include "test_utils.h"

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void TestLogSO3_h(const float *h_R, float *h_psi) {

    // Allocate memory from the device
    float *d_psi, *d_R;
    cudaMalloc(&d_psi, 3 * sizeof(float));
    cudaMalloc(&d_R, 9 * sizeof(float));

    // Copy vector psi vector to device
    cudaMemcpy(d_R, h_R, 9 * sizeof(float), cudaMemcpyHostToDevice);
    TestLogSO3_d(d_R, d_psi);

    // Copy rotation matrix to host
    cudaMemcpy(h_psi, d_psi, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_psi);
    cudaFree(d_R);
}



// Multiple tests to check if the implementation of ExpSO3 is correct.
bool MultipleTestsLogSO3() {
    cv::RNG rng(cv::getTickCount());  // Seed RNG
    std::cout << "Testing if the implementation of LogSO3 is correct, please wait..." << std::endl;
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
            // std::cout << "Norm is set to be PI" << std::endl;
        } else if (randomChance < 0.2) {
            modifiedNorm = (float)(M_PI + rng.uniform(-0.1f, 0.1f));  
            // std::cout << "Norm is around PI" << std::endl;
        } else if (randomChance < 0.201) {
            modifiedNorm = 0.0f;
            // std::cout << "modifiedNorm : " << modifiedNorm << std::endl;
        } else if (randomChance < 0.30) {
            modifiedNorm = (float)(rng.uniform(-1e-1, 1e-1));
            // std::cout << "modifiedNorm : " << modifiedNorm << std::endl;
        }

        psi = psi * (modifiedNorm / norm);

        // Step 2: Obtain the correponding rotation matrix.
        cv::Mat R_cv;
        cv::Rodrigues(psi, R_cv);  // Convert axis-angle to rotation matrix
        R_cv.convertTo(R_cv, CV_32F);

        // Convert rotation matrix to float array
        float* h_R = (float*)R_cv.data;
        float h_vec[3];

        // Step 3: Call CUDA Kernel
        TestLogSO3_h(h_R, h_vec);
        cudaDeviceSynchronize();
        cv::Mat reconstructed_psi(3, 1, CV_32F, h_vec);

        // Step 4 : Calculate the difference between psi and reconstructed_psi
        cv::Mat diff = psi - reconstructed_psi;

        // Check if the vectors are "close enough" based on the norm
        if (cv::norm(diff, cv::NORM_L2) < 1e-6) {
            // std::cout << "The vectors are the same (within the tolerance)." << std::endl;
        } else {
            // We need to check if the constructed rotation matrices are the same
            cv::Mat R_cv_reconstructed;
            cv::Rodrigues(reconstructed_psi, R_cv_reconstructed);  // Convert axis-angle to rotation matrix

            cv::Mat R_discrepancy = R_cv * R_cv_reconstructed.t();
            cv::Mat psi_discrepancy;
            cv::Rodrigues(R_discrepancy, psi_discrepancy);  // Convert axis-angle to rotation matrix

            // std::cout << "cv::norm(R_diff, cv::NORM_L2) : " << cv::norm(R_diff, cv::NORM_L2) << std::endl;

            if (cv::norm(psi_discrepancy, cv::NORM_L2) < 5e-4) {
                // std::cout << "Reconstructed rotation matrices are the same (within the tolerance)." << std::endl;
            } else {
                std::cout << "Test Failed: CUDA and OpenCV matrices do not match. Error Norm : " << cv::norm(psi_discrepancy, cv::NORM_L2) << std::endl;
                cv::Mat psi_discrepancy2;
                cv::Rodrigues(R_cv * R_cv_reconstructed, psi_discrepancy2);
                std::cout << "R_cv : " << std::endl << R_cv << std::endl << std::endl;
                std::cout << "R_cv_reconstructed : " << std::endl << R_cv_reconstructed << std::endl << std::endl;
                std::cout << "psi : " << std::endl << psi << std::endl << std::endl;
                std::cout << "reconstructed_psi : " << std::endl << reconstructed_psi << std::endl << std::endl;
                std::cout << "modifiedNorm : " << modifiedNorm << std::endl << std::endl;
                std::cout << "cv::norm(psi_discrepancy, cv::NORM_L2) : " << cv::norm(psi_discrepancy, cv::NORM_L2) << std::endl << std::endl;
                
                return false;
            }
        }
    }

    return true;
}