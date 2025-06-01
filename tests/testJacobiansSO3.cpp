#include "test_utils.h"

void NumericLeftRightJacobiansSO3Kernel(cv::Mat psi, cv::Mat &JlPtr, cv::Mat &JrPtr) {
    
    //Make sure that data type of psi is double
    psi.convertTo(psi, CV_64F);

    double eps = 1e-4;

    cv::Mat R;
    cv::Rodrigues(psi, R);  // Convert axis-angle to rotation matrix
    
    cv::Mat Jl = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat Jr = cv::Mat::zeros(3, 3, CV_64F);

    for (int idx = 0; idx < 3; ++idx) {
        cv::Mat psi_disturbed = psi.clone();
        psi_disturbed.at<double>(idx, 0) += eps;

        cv::Mat R_disturbed;
        cv::Rodrigues(psi_disturbed, R_disturbed);  // Convert axis-angle to rotation matrix

        // Compute left Jacobian
        cv::Mat psi_new_left;
        cv::Rodrigues(R_disturbed * R.t(), psi_new_left);

        for (int row = 0; row < 3; ++row) {  // Assign element-wise
            Jl.at<double>(row, idx) = psi_new_left.at<double>(row, 0) / eps;
        }
        
        // std::cout << "psi_new_left.at<double>(row, 0) / eps : " << std::endl << psi_new_left / eps << std::endl;
        // Compute right Jacobian
        cv::Mat psi_new_right;
        cv::Rodrigues(R.t() * R_disturbed, psi_new_right);


    
        for (int row = 0; row < 3; ++row) {  // Assign element-wise
            Jr.at<double>(row, idx) = psi_new_right.at<double>(row, 0) / eps;
        }
    }

    JlPtr = Jl;
    JrPtr = Jr;
}

void TestJacobiansSO3Kernel_h(const float *h_psi, float *h_Jl, float *h_Jr) {

    // Allocate memory from the device
    float *d_psi, *d_Jl, *d_Jr;
    cudaMalloc(&d_psi, 3 * sizeof(float));
    cudaMalloc(&d_Jl, 9 * sizeof(float));
    cudaMalloc(&d_Jr, 9 * sizeof(float));

    // Copy vector psi vector to device
    cudaMemcpy(d_psi, h_psi, 3 * sizeof(float), cudaMemcpyHostToDevice);
    TestJacobiansSO3_d(d_psi, d_Jl, d_Jr);

    // Copy rotation matrix to host
    cudaMemcpy(h_Jl, d_Jl, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Jr, d_Jr, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_psi);
    cudaFree(d_Jl);
    cudaFree(d_Jr);
}


void TestInvJacobiansSO3Kernel_h(const float *h_psi, float *h_Jl_inv, float *h_Jr_inv) {

    // Allocate memory from the device
    float *d_psi, *d_Jl_inv, *d_Jr_inv;
    cudaMalloc(&d_psi, 3 * sizeof(float));
    cudaMalloc(&d_Jl_inv, 9 * sizeof(float));
    cudaMalloc(&d_Jr_inv, 9 * sizeof(float));

    // Copy vector psi vector to device
    cudaMemcpy(d_psi, h_psi, 3 * sizeof(float), cudaMemcpyHostToDevice);
    TestInvJacobiansSO3_d(d_psi, d_Jl_inv, d_Jr_inv);

    // Copy rotation matrix to host
    cudaMemcpy(h_Jl_inv, d_Jl_inv, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Jr_inv, d_Jr_inv, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_psi);
    cudaFree(d_Jl_inv);
    cudaFree(d_Jr_inv);
}



// Multiple tests to check if the implementation of ExpSO3 is correct.
bool MultipleTestsJacobiansSO3Kernel() {
    cv::RNG rng(cv::getTickCount());  // Seed RNG
    std::cout << "Testing if the implementation of Jacobians of SO3 is correct, please wait..." << std::endl;
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
        } else if (randomChance < 0.5) {
            modifiedNorm = (float)(M_PI + rng.uniform(-0.1f, 0.1f));  
            // std::cout << "Norm is around PI" << std::endl;
        } else if (randomChance < 0.6) {
            modifiedNorm = 0.0f;
            // std::cout << "modifiedNorm : " << modifiedNorm << std::endl;
        } else if (randomChance < 0.9) {
            modifiedNorm = (float)(rng.uniform(-1e-1, 1e-1));
            // std::cout << "modifiedNorm : " << modifiedNorm << std::endl;
        }

        psi = psi * (modifiedNorm / norm);

        // Step 2: Obtain the correponding left, right Jacobians and also inverses
        float* h_psi = (float*)psi.data;
        float h_Jl[9], h_Jr[9];
        float h_Jl_inv[9], h_Jr_inv[9];

        // Step 3: Call CUDA Kernel
        TestJacobiansSO3Kernel_h(h_psi, h_Jl, h_Jr);
        TestInvJacobiansSO3Kernel_h(h_psi, h_Jl_inv, h_Jr_inv);
        cudaDeviceSynchronize();    

        // Convert the result into a CV matrix

        cv::Mat Jl_analytical(3, 3, CV_32F, h_Jl);
        cv::Mat Jr_analytical(3, 3, CV_32F, h_Jr);

        cv::Mat Jl_inv_analytical(3, 3, CV_32F, h_Jl_inv);
        cv::Mat Jr_inv_analytical(3, 3, CV_32F, h_Jr_inv);
        
        // Step 3: Numerically compute the left and right Jacobians for comparsion.
        cv::Mat Jl_numeric = cv::Mat::zeros(3, 3, CV_32F);
        cv::Mat Jr_numeric = cv::Mat::zeros(3, 3, CV_32F);

        NumericLeftRightJacobiansSO3Kernel(psi, Jl_numeric, Jr_numeric);

        Jl_numeric.convertTo(Jl_numeric, CV_32F);
        Jr_numeric.convertTo(Jr_numeric, CV_32F);

        // Compute the error
        cv::Mat Jl_disp = Jl_numeric - Jl_analytical;
        cv::Mat Jr_disp = Jr_numeric - Jr_analytical;

    
        if ((cv::norm(Jl_disp, cv::NORM_INF) < 5e-5) && (cv::norm(Jr_disp, cv::NORM_INF) < 5e-5)) {
            // std::cout << "Reconstructed rotation matrices are the same (within the tolerance)." << std::endl;
        } else {
            std::cout << "cv::norm(Jl_disp, cv::NORM_INF) : " << cv::norm(Jl_disp, cv::NORM_INF) << std::endl;
            std::cout << "cv::norm(Jr_disp, cv::NORM_INF) : " << cv::norm(Jr_disp, cv::NORM_INF) << std::endl;
            return false;
        }

        // Also check the if inverse Jacobians are correct
        cv::Mat Jl_Jl_inv = Jl_analytical * Jl_inv_analytical;
        cv::Mat Jr_Jr_inv = Jr_analytical * Jr_inv_analytical;

        cv::Mat identity = cv::Mat::eye(3, 3, CV_32F);

        if ((cv::norm(Jl_Jl_inv - identity, cv::NORM_INF) < 5e-5) && (cv::norm(Jr_Jr_inv - identity, cv::NORM_INF) < 5e-5)) {
            // std::cout << "Inverse Jacobians are correct (within the tolerance)." << std::endl;
        } else {
            std::cout << "cv::norm(Jl_Jl_inv - identity, cv::NORM_INF) : " << cv::norm(Jl_Jl_inv - identity, cv::NORM_INF) << std::endl;
            std::cout << "cv::norm(Jr_Jr_inv - identity, cv::NORM_INF) : " << cv::norm(Jr_Jr_inv - identity, cv::NORM_INF) << std::endl << std::endl;


            std::cout << "modifiedNorm : " << modifiedNorm << std::endl << std::endl;

            std::cout << "psi : " << psi << std::endl << std::endl;

            std::cout << "Jl_analytical : " << std::endl << Jl_analytical << std::endl;
            std::cout << "Jl_inv_analytical : " << std::endl << Jl_inv_analytical << std::endl << std::endl;

            std::cout << "Jl_Jl_inv : " << std::endl << Jl_Jl_inv << std::endl << std::endl;

            std::cout << "Jl_numeric * Jl_inv_analytical : " << std::endl << Jl_numeric * Jl_inv_analytical << std::endl << std::endl;

            std::cout << "Jl_analytical - Jl_numeric : " << std::endl << Jl_analytical - Jl_numeric << std::endl;

            return false;
        }
    }

    return true;
}