#include "lie.h"



// __device__ function to multiply a HEIGHTxHIDDEN and a HIDDENxWIDTH matrix using 1D arrays
__device__ void MatrixMultiplication(const float *A, const float *B, float *C, int HEIGHT, int HIDDEN, int WIDTH) {
    // Iterate over rows of A
    for (int row_idx = 0; row_idx < HEIGHT; row_idx++) {
        // Iterate over columns of B 
        for (int col_idx = 0; col_idx < WIDTH; col_idx++) {
            C[row_idx * WIDTH + col_idx] = 0;  // Initialize the result element
            // Perform the dot product of the i-th row of A and the j-th column of B
            for (int k = 0; k < HIDDEN; k++) {
                C[row_idx * WIDTH + col_idx] += A[row_idx * HIDDEN + k] * B[k * WIDTH + col_idx];
            }
        }
    }
}

// Get the skew symmetric matrix form of a 3x1 vector s.
__device__ void skew(const float *s, float *S) {
    // First row
    S[0] = 0.0;    
    S[1] = -s[2];
    S[2] = s[1];

    // Second Raw
    S[3] = s[2];
    S[4] = 0.0;
    S[5] = -s[0];

    // Third Raw
    S[6] = -s[1];
    S[7] = s[0];
    S[8] = -0.0;
}

__device__ void InvertPose4x4(const float* input, float* output) {
    // Extract rotation matrix R (3x3) and translation vector T (3x1)
    float R[9];
    float T[3];
    
    // Row-major layout
    R[0] = input[0]; R[1] = input[1]; R[2] = input[2];
    R[3] = input[4]; R[4] = input[5]; R[5] = input[6];
    R[6] = input[8]; R[7] = input[9]; R[8] = input[10];

    T[0] = input[3];
    T[1] = input[7];
    T[2] = input[11];

    // Compute transpose of R (i.e., R^T)
    float Rt[9];
    Rt[0] = R[0]; Rt[1] = R[3]; Rt[2] = R[6];
    Rt[3] = R[1]; Rt[4] = R[4]; Rt[5] = R[7];
    Rt[6] = R[2]; Rt[7] = R[5]; Rt[8] = R[8];

    // Compute -R^T * T
    float T_new[3];
    T_new[0] = -(Rt[0] * T[0] + Rt[1] * T[1] + Rt[2] * T[2]);
    T_new[1] = -(Rt[3] * T[0] + Rt[4] * T[1] + Rt[5] * T[2]);
    T_new[2] = -(Rt[6] * T[0] + Rt[7] * T[1] + Rt[8] * T[2]);

    // Write the 4x4 inverse matrix in row-major format
    output[0] = Rt[0]; output[1] = Rt[1]; output[2] = Rt[2]; output[3] = T_new[0];
    output[4] = Rt[3]; output[5] = Rt[4]; output[6] = Rt[5]; output[7] = T_new[1];
    output[8] = Rt[6]; output[9] = Rt[7]; output[10] = Rt[8]; output[11] = T_new[2];
    output[12] = 0.0f; output[13] = 0.0f; output[14] = 0.0f; output[15] = 1.0f;
}