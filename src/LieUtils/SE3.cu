#include "lie.h"

__device__ void ExpSE3(const float *xi, float *T) {
    // Decompose xi into psi (rotation) and rho 
    float psi[3], rho[3];

    for (int i = 0; i < 3; i++) {
        psi[i] = xi[i];
        rho[i] = xi[i+3];
    }

    float R[9];
    ExpSO3(psi, R);

    float Jl[9];
    LeftJacobianSO3(psi, Jl);

    float t[3];
    MatrixMultiplication(Jl, rho, t, 3, 3, 1);

    for(int row_idx=0; row_idx<3; ++row_idx){
        for(int col_idx=0; col_idx<3; ++col_idx){
            int idx_T = 4 * row_idx + col_idx;
            int idx_R = 3 * row_idx + col_idx;
            T[idx_T]  = R[idx_R];
        }
    }

    for(int row_idx=0; row_idx<3; ++row_idx){
        int idx_T = 4 * row_idx + 3;
        T[idx_T]  = t[row_idx];
    }

    T[12] = 0.0;
    T[13] = 0.0;
    T[14] = 0.0;
    T[15] = 1.0;
}



__device__ void LogSE3(const float *T, float *xi) {
    // First get the rotation matrix:
    float R[9];
    for(int row_idx=0; row_idx<3; ++row_idx){
        for(int col_idx=0; col_idx<3; ++col_idx){
            int idx_T = 4 * row_idx + col_idx;
            int idx_R = 3 * row_idx + col_idx;
            R[idx_R] = T[idx_T];
        }
    }

    // Get the transition vector:
    float t[3];
    for(int row_idx=0; row_idx<3; ++row_idx){
        int idx_T = 4 * row_idx + 3;
        t[row_idx] = T[idx_T];
    }

    // Get the psi vector
    float psi[3];
    LogSO3(R, psi);

    // Get the rho vector
    float Jl_inv[9];
    InvLeftJacobianSO3(psi, Jl_inv);
    float rho[3];
    MatrixMultiplication(Jl_inv, t, rho, 3, 3, 1);

    // Finally get the xi
    for (int i = 0; i < 3; i++) {
        xi[i]   = psi[i];
        xi[i+3] = rho[i];
    }
}


__device__ void Q_SE3(const float *xi, float *Q){
    // Decompose xi into psi (rotation) and rho 
    float psi[3], rho[3];

    for (int i = 0; i < 3; i++) {
        psi[i] = xi[i];
        rho[i] = xi[i+3];
    }

    // Compute the psi_norm
    float psi_norm = sqrt( psi[0]*psi[0] + psi[1]*psi[1] + psi[2]*psi[2] );

    // Compute the necessary terms
    float psi_skew[9], rho_skew[9], psi_rho[9], rho_psi[9], psi_rho_psi[9], psi2_rho[9], rho_psi2[9], psi_rho_psi2[9], psi2_rho_psi[9];
    skew(psi, psi_skew);
    skew(rho, rho_skew);
    MatrixMultiplication(psi_skew, rho_skew, psi_rho, 3, 3, 3);
    MatrixMultiplication(rho_skew, psi_skew, rho_psi, 3, 3, 3);
    MatrixMultiplication(psi_skew, rho_psi, psi_rho_psi, 3, 3, 3);
    MatrixMultiplication(psi_skew, psi_rho, psi2_rho, 3, 3, 3);
    MatrixMultiplication(rho_psi, psi_skew, rho_psi2, 3, 3, 3);
    MatrixMultiplication(psi_skew, rho_psi2, psi_rho_psi2, 3, 3, 3);
    MatrixMultiplication(psi2_rho, psi_skew, psi2_rho_psi, 3, 3, 3);

    // Compute the constants
    float c[4];
    c[0] = 0.5f;

    if(psi_norm > NUMERICAL_TOLERANCE){
        // General terms
        float sin_psi_norm = sinf(psi_norm);
        float cos_psi_norm = cosf(psi_norm);
        float psi_norm2 = psi_norm * psi_norm;
        float psi_norm3 = psi_norm2 * psi_norm;
        float psi_norm4 = psi_norm3 * psi_norm;
        float psi_norm5 = psi_norm4 * psi_norm;

        // constant 1
        c[1] = (psi_norm - sin_psi_norm) / psi_norm3;
        
        // constant 2
        c[2] = - (1.0f - 0.5*psi_norm2 - cos_psi_norm) / psi_norm4;

        // constant 3
        c[3] = 0.5 * (c[2] + 3*( (psi_norm-sin_psi_norm-(psi_norm3/6.0f)) / psi_norm5) );
    
    } else{
        // General terms
        float psi_norm2 =  psi_norm * psi_norm;
        // factorials
        float fac3_inv{1.0/6.0}, fac4_inv{1.0/24.0}, fac5_inv{1.0/120.0}, fac6_inv{1.0/720.0}, fac7_inv{1.0/5040.0};

        // constant 1
        c[1] = fac3_inv - fac5_inv*psi_norm2;

        // constant 2
        c[2] = fac4_inv - fac6_inv*psi_norm2;

        // constant 2
        c[3] = 0.5 * (c[2] + 3*( - fac5_inv + ( fac7_inv*psi_norm2 ) ) );
    }

    // Now, we are ready to compute the Q
    for(int idx=0; idx<9; ++idx){
        float terms[4];
        terms[0] = rho_skew[idx];
        terms[1] = psi_rho[idx] + rho_psi[idx] + psi_rho_psi[idx];
        terms[2] = psi2_rho[idx] + rho_psi2[idx] - 3*psi_rho_psi[idx];
        terms[3] = psi_rho_psi2[idx] + psi2_rho_psi[idx];

        Q[idx] = c[0]*terms[0] + c[1]*terms[1] + c[2]*terms[2] + c[3]*terms[3];
    }
}


__device__ void LeftJacobianSE3(const float *xi, float *Jl){
    // Decompose xi into psi (rotation) and rho 
    float psi[3];

    for (int i = 0; i < 3; i++) {
        psi[i] = xi[i];
    }

    // Get the left Jacobian for the SO3
    float Jl_SO3[9];
    LeftJacobianSO3(psi, Jl_SO3);

    // Get the Q
    float Q[9];
    Q_SE3(xi, Q);

    // Fill the Jacobian
    for(int row_idx=0; row_idx<3; ++row_idx){
        for(int col_idx=0; col_idx<3; ++col_idx){
            int item_idx   = 3 * row_idx + col_idx;
            int Jl_Jl_idx0 = 6 * row_idx + col_idx;
            int Jl_Jl_idx1 = 6 * (row_idx+3) + (col_idx + 3);
            int Jl_Q_idx   = 6 * (row_idx+3) + col_idx;
            
            Jl[Jl_Jl_idx0] = Jl_SO3[item_idx];
            Jl[Jl_Jl_idx1] = Jl_SO3[item_idx];
            Jl[Jl_Q_idx]   = Q[item_idx];
        }
    }
}

__device__ void RightJacobianSE3(const float *xi, float *Jr){
    // Jr(psi) = Jl(-psi)
    float xi_neg[6];
    for (int i = 0; i < 6; i++) {
        xi_neg[i] = - xi[i];
    }

    LeftJacobianSE3(xi_neg, Jr);    
}