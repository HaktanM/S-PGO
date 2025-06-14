from build import Solver
import numpy as np

from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import map_value_to_index

import time

from PythonUtils.visualization_utils import visualize_jacobian_and_residual_to_cv, visualize_hessian_and_g

from PythonUtils.Optimizer import Optimizer


import matplotlib.pyplot as plt



class Manager():
    def __init__(self):

        # Number of keyframes
        self.n = 12

        # NUmber of landmarks per frame
        self.m = 96

        # Number of total landmarks 
        self.M = self.n * self.m

        # Initialize the simulator
        self.simulator = Simulator(n=self.n+1, m=self.M)

        # Initilize the CUDA solver
        self.solver = Solver.CudaSolver(self.n, self.m)

        # Initialize the optimizer to compare two results
        self.optimizer = Optimizer(n = self.n, m=self.M)
        self.optimizer.initialize_estimated_incremental_poses(self.simulator.incremental_poses)

        # Load calibration to solver
        self.loadCalibration()

        # Load Observations to solver
        self.loadObservations()

        # Load estimated inverse depths to the solver
        self.loadInverseDepths()

        # Load estimated incremental poses to solver
        self.loadIncrementalPoses()

    def loadObservations(self):
        for landmark_idx in range(self.M): 
            anchor_idx = map_value_to_index(v=landmark_idx, x=self.M, n=self.n)
            for projection_idx in range(anchor_idx, self.n+1):
                if True: # self.simulator.validty[projection_idx][landmark_idx][False] and self.simulator.validty[projection_idx][landmark_idx][True]:
                    left_obs_py  = self.simulator.observations[projection_idx][landmark_idx][False].reshape(3).astype(np.float32)
                    right_obs_py = self.simulator.observations[projection_idx][landmark_idx][True].reshape(3).astype(np.float32)
                    self.solver.writeObservations(anchor_idx, projection_idx, landmark_idx, left_obs_py, right_obs_py)
                else:
                    print("Non Valid Observation has been detected")

    def loadIncrementalPoses(self):
        for idx, incremental_pose in enumerate(self.optimizer.estimated_incremental_poses):
            self.solver.writeIncrementalPose(idx, incremental_pose.reshape(16))

    def loadCalibration(self):
        intrinsics = np.array([
            self.simulator.cam.Kl[0,0], self.simulator.cam.Kl[1,1], self.simulator.cam.Kl[0,2], self.simulator.cam.Kl[1,2],
            self.simulator.cam.Kr[0,0], self.simulator.cam.Kr[1,1], self.simulator.cam.Kr[0,2], self.simulator.cam.Kr[1,2]
        ]).reshape(8).astype(np.float32)

        T_r_to_l = self.simulator.cam.T_r_l.reshape(16).astype(np.float32)

        self.solver.loadCalibration(intrinsics, T_r_to_l)

    def loadInverseDepths(self):
        inverse_depths = np.array(self.optimizer.estimated_inverse_depths).reshape(-1).astype(np.float32)
        self.solver.loadInverseDepths(inverse_depths)

if __name__ == "__main__":
    manager = Manager()

    

    # print(f"Total elapsed_time : { (t_stop - t_start) * (1e-6) } milliseconds")


    J_T_o, J_alpha_o, r_o = manager.optimizer.getJacobiansAndResidual(manager.simulator.observations)
    H_TT_o = J_T_o.transpose() @ J_T_o
    g_T_o  = J_T_o.transpose() @ r_o
    
    H_aa_o = J_alpha_o.transpose() @ J_alpha_o
    g_a_o  = J_alpha_o.transpose() @ r_o
    B_o = J_T_o.transpose() @ J_alpha_o
    B_C_inv_o = B_o @ np.linalg.inv(H_aa_o + np.eye(H_aa_o.shape[0])*0.0001)
    B_C_inv_B_T_o = B_C_inv_o @ B_o.transpose()

    H_schur_o = H_TT_o - B_C_inv_B_T_o 
    g_schur_o = g_T_o  - B_C_inv_o @ g_a_o

    delta_pose_o  = np.linalg.solve(H_schur_o, g_schur_o)

    B_T_delta_T_o = B_o.transpose() @ delta_pose_o
    delta_a_o     = np.linalg.inv(H_aa_o + np.eye(H_aa_o.shape[0])*0.0001) @ (g_a_o - B_T_delta_T_o)

    innovation = delta_pose_o[:6,:]
    update = manager.optimizer.LU.Exp_SE3(innovation)
    
    incremental_pose = manager.solver.getIncrementalPose(0)
    updated_state = incremental_pose @ update

    # print("H_schur_o")
    # print(H_schur_o)

    # print("g_schur_o")
    # print(g_schur_o)

    # print("delta_a_o")
    # print(delta_a_o)

    t_start = time.monotonic_ns()
    manager.solver.step(1)
    t_stop  = time.monotonic_ns()

    H_schur     = np.loadtxt("H_schur.txt", delimiter=",")
    g_schur     = np.loadtxt("g_schur.txt", delimiter=",")
    delta_pose  = np.loadtxt("delta_pose.txt", delimiter=",")

    # visualize_hessian_and_g(H_schur, g_schur)

    # ev, _ = np.linalg.eig(H_schur)
    # ev_o, _ = np.linalg.eig(H_schur_o)

    # print(ev)
    # print("---------")
    # print(ev_o)

    delta_pose_co  = np.linalg.solve(H_schur, g_schur)

    fig, axes = plt.subplots(4, 1, figsize=(8, 8))  # Adjust figsize as needed
        # Example: Plot something in each subplot
    axes[0].plot(H_schur_o.flatten(), color="green")
    axes[0].plot(H_schur.flatten()  , "--", color="red")

    axes[1].plot(g_schur_o.flatten(), color="green")
    axes[1].plot(g_schur.flatten()  , "--",   color="red")

    axes[2].plot(delta_pose_o.flatten(), color="green")
    axes[2].plot(delta_pose.flatten()  , "--", color="red")
    axes[2].plot(delta_pose_co.flatten(), "--", color="blue")

    axes[3].plot((H_schur_o - H_schur).flatten(), color="green")

    plt.show()



    

    