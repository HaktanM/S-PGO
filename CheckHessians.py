from PythonManager import Manager
from PythonUtils.visualization_utils import visualize_jacobian_and_residual_to_cv, visualize_hessian_and_g
import numpy as np
import time
import threading

if __name__ == "__main__":
    manager = Manager()

    H_TT, g_TT, H_aa, g_aa, BB = manager.optimizer.getHessians(observations=manager.simulator.observations)
    J_T, J_a, r                = manager.optimizer.getJacobians(observations=manager.simulator.observations)

    H_T = J_T.T @ J_T 
    g_T = J_T.T @ r

    H_a = J_a.T @ J_a 
    g_a = J_a.T @ r

    B = J_T.T @ J_a

    visualize_jacobian_and_residual_to_cv(J_a,r)


    visualize_hessian_and_g(H_TT, g_TT)
    visualize_hessian_and_g(H_T, g_T)

    visualize_hessian_and_g(np.diag(H_aa.reshape(-1)), g_aa)
    visualize_hessian_and_g(H_a, g_a)

    visualize_hessian_and_g(BB, g_aa)
    visualize_hessian_and_g(B, g_a)
