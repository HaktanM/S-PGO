#include "lie.h"



__device__ void ExpSO3(const float *psi, float *R) {
    float S[9], S2[9]; 

    skew(psi, S);                               // Compute skew-symmetric matrix S
    MatrixMultiplication(S, S, S2, 3, 3, 3);    // Compute S * S  (matrix multiplication)

    // Norm of psi
    double angle = sqrt(psi[0] * psi[0] + psi[1] * psi[1] + psi[2] * psi[2]);

    // -------------------------------------- Finally compute the rotation matrix --------------------------------------
    if (angle > NUMERICAL_TOLERANCE) {
        double c1 = sinf(angle) / angle;
        double c2 = (1 - cosf(angle)) / (angle * angle);
        for (int i = 0; i < 9; i++) {
            R[i] =  c1 * S[i] + c2 * S2[i];
        }
    } else {
        for (int i = 0; i < 9; i++) {
            R[i] = S[i] + 0.5f * S2[i];
        }
    }
    
    // Add Identity
    R[0] += 1.0f;
    R[4] += 1.0f;
    R[8] += 1.0f;
    // -------------------------------------- -------------------------------------- --------------------------------------
}



__device__ void LogSO3(const float *R, float *psi) {
    
    // First compute the norm of the psi vector 
    double trace_R = (double) ((double)R[0] + (double)R[4] + (double)R[8]);
    double cos_norm_psi = 0.5 * trace_R - 0.5;
    cos_norm_psi = fmax(-1.0, fmin(cos_norm_psi, 1.0)); // Clamp cos_norm_psi to the range [-1, 1] before calling acos for numerical stability
    double norm_psi =  acos(cos_norm_psi);

    // TO DO : When norm is PI, numerical accuracy is not enough !!!
    if(norm_psi < (M_PI - (5e-1))){ // This is the standard formula
        double c;
        if(norm_psi < NUMERICAL_TOLERANCE){
            c = 0.5;   // TO DO: Check if a better approximation available
        } else{ 
            c = 0.5 * norm_psi / sin(norm_psi);
        }

        psi[0] = - c * (R[5]-R[7]);
        psi[1] =   c * (R[2]-R[6]);
        psi[2] = - c * (R[1]-R[3]);

    } else { // if (norm_psi < (M_PI + NUMERICAL_TOLERANCE)) { // This is the case when norm_psi is around PI       
        double norms[3] = {
            norm_psi * sqrt( ( R[0] - cos_norm_psi ) / ( 1.0 - cos_norm_psi ) ),
            norm_psi * sqrt( ( R[4] - cos_norm_psi ) / ( 1.0 - cos_norm_psi ) ),
            norm_psi * sqrt( ( R[8] - cos_norm_psi ) / ( 1.0 - cos_norm_psi ) )
        };

        // Now, you can find the sign of each item.
        double signs[3];

        // First find the sign of the first item:        
        signs[0] = copysign(1.0, copysign(1.0f, R[7] - R[5]) * copysign(1.0, sin(norm_psi)));
        signs[1] = copysign(1.0, signs[0] * (R[1] + R[3]) );
        signs[2] = copysign(1.0, signs[0] * (R[2] + R[6]) );
        
        // Make sure that all the conditions are satisfied
        assert(fabs(signs[1] * signs[2] - copysign(1.0, R[5] + R[7])) < NUMERICAL_TOLERANCE);
        
        // Finally, we obtain the psi
        for(int idx = 0; idx < 3; ++idx){
            psi[idx] = norms[idx] * signs[idx];
        }
    } 
}

__device__ void LeftJacobianSO3(const float *psi, float *Jl) {
    // Get skew symmetric form of the psi
    float S[9], S2[9];
    skew(psi, S);
    MatrixMultiplication(S, S, S2, 3, 3, 3);
    
    // Compute the norm of the input vector
    double psi_norm = (double)sqrtf( psi[0] * psi[0] + psi[1] * psi[1] + psi[2] * psi[2] );
    double c1, c2;
    if(psi_norm > (NUMERICAL_TOLERANCE)){     // If the norm is large enough, use the closed form expression
        c1 = ( 1.0 - cos(psi_norm) ) / ( psi_norm * psi_norm );
        c2 = (psi_norm - sin(psi_norm)) / ( psi_norm * psi_norm * psi_norm );
    } else {                                // If the norm is not large enough, ignore the higher order terms.                         
        c1 = 0.5;
        c2 = 1.0f / 6.0f;
    }

    for (int i = 0; i < 9; i++) {
        Jl[i] = (float)c1 * S[i] + (float)c2 * S2[i];
    }

    // Finally, add itentity component
    Jl[0] += 1.0f;
    Jl[4] += 1.0f;
    Jl[8] += 1.0f;
}

__device__ void RightJacobianSO3(const float *psi, float *Jr) {
    // Jr(psi) = Jl(-psi)
    float psi_neg[3];
    for (int i = 0; i < 3; i++) {
        psi_neg[i] = - psi[i];
    }

    LeftJacobianSO3(psi_neg, Jr);    
}

__device__ void InvLeftJacobianSO3(const float *psi, float *Jl_inv){
    // Get skew symmetric form of the psi
    float S[9], S2[9];
    skew(psi, S);
    MatrixMultiplication(S, S, S2, 3, 3, 3);

    // Compute the norm of the input vector
    double psi_norm = (double)sqrtf( psi[0] * psi[0] + psi[1] * psi[1] + psi[2] * psi[2] );

    double c1, c2;
    if(psi_norm > NUMERICAL_TOLERANCE){         // If the norm is large enough, use the closed form expression
        c1 = - 0.5;
        c2 = ( 1.0 / (psi_norm * psi_norm) ) - ( 0.5 * cos(psi_norm / 2.0) / ( psi_norm * sin(psi_norm / 2.0) ) );                
    } else {                                    // If the norm is not large enough, ignore the higher order terms.
        c1 = -0.5;
        c2 =  1.0 / 6.0 / 2.0;
    }

    for (int i = 0; i < 9; i++) {
        Jl_inv[i] = c1 * S[i] + c2 * S2[i];
    }

    // Finally, add itentity component
    Jl_inv[0] += 1.0f;
    Jl_inv[4] += 1.0f;
    Jl_inv[8] += 1.0f;
}


__device__ void InvRightJacobianSO3(const float *psi, float *Jr_inv){
    // Jr_inv(psi) = Jl_inv(-psi)
    float psi_neg[3];
    for (int i = 0; i < 3; i++) {
        psi_neg[i] = - psi[i];
    }

    InvLeftJacobianSO3(psi_neg, Jr_inv);    
}


