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
