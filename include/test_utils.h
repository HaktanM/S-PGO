#ifndef CUDA_TESTS_H
#define CUDA_TESTS_H

#include <iostream>
#include <opencv2/opencv.hpp>  // OpenCV for random vector & Rodrigues formula
#include <cuda_runtime.h>
#include "lie.h"
#include <cstring>  // For memcpy

// Function to compare two matrices
inline bool compareMatrices(const float* mat1, const cv::Mat& mat2, float threshold = 1e-5) {
    for (int i = 0; i < 9; i++) {
        if (std::abs(mat1[i] - mat2.at<float>(i / 3, i % 3)) > threshold) {
            return false;  // If any element is outside the threshold, matrices are different
        }
    }
    return true;  // Matrices are considered the same
}

// SO3 Tools
bool MultipleTestsExpSO3();
bool MultipleTestsLogSO3();
bool MultipleTestsJacobiansSO3Kernel();

// SE3 Tools
bool MultipleTestsSE3();
#endif