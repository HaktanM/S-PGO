from build import Solver
import numpy as np

from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import map_value_to_index

import time

from PythonUtils.visualization_utils import visualize_jacobian_and_residual_to_cv

from PythonUtils.Optimizer import Optimizer

class Manager():
    def __init__(self):

        # Number of keyframes
        self.n = 12

        # NUmber of landmarks per frame
        self.m = 100

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

    # for idx in range(manager.n):
    #     incremental_pose = manager.solver.getIncrementalPose(idx)
    #     print(incremental_pose)

    intrinsics, extrinsics = manager.solver.getCalibration()
 
    manager.solver.step(1)
    J_T, J_alpha, r = manager.solver.getJacobiansAndResidual()


    start_time = time.time()

    J_T_o, J_alpha_o, r_o = manager.optimizer.getJacobiansAndResidual(manager.simulator.observations)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for one step: {elapsed_time:.4f} seconds")


    print(J_T.shape)
    print(J_T_o.shape)


    # Compute percentage error safely (avoid division by zero)
    def percentage_error(A, B):
        # Add small epsilon to denominator to avoid division by zero
        epsilon = 1e-4
        return np.abs(A - B) / (np.abs(B) + epsilon) * 100

    # For J_T
    pct_err_JT = percentage_error(J_T, J_T_o[:J_T.shape[0], :])
    max_idx_pct_JT = np.unravel_index(np.argmax(pct_err_JT), pct_err_JT.shape)
    max_pct_JT = pct_err_JT[max_idx_pct_JT]

    print(f"Max percentage error in J_T at index {max_idx_pct_JT}: {max_pct_JT:.6f}%")
    print(f"J_T value: {J_T[max_idx_pct_JT]}, J_T_o value: {J_T_o[max_idx_pct_JT]}")

    # For J_alpha
    pct_err_Jalpha = percentage_error(J_alpha, J_alpha_o[:J_alpha.shape[0], :])
    max_idx_pct_Jalpha = np.unravel_index(np.argmax(pct_err_Jalpha), pct_err_Jalpha.shape)
    max_pct_Jalpha = pct_err_Jalpha[max_idx_pct_Jalpha]

    print(f"Max percentage error in J_alpha at index {max_idx_pct_Jalpha}: {max_pct_Jalpha:.6f}%")
    print(f"J_alpha value: {J_alpha[max_idx_pct_Jalpha]}, J_alpha_o value: {J_alpha_o[max_idx_pct_Jalpha]}")



    # print(J_alpha[:3,:3])
    # print(J_alpha_o[:3,:3])

    visualize_jacobian_and_residual_to_cv(J_T, r)
    visualize_jacobian_and_residual_to_cv(J_T_o[:J_T.shape[0],:], r_o[:r.shape[0],:])

    visualize_jacobian_and_residual_to_cv(J_alpha, r)
    visualize_jacobian_and_residual_to_cv(J_alpha_o[:J_alpha.shape[0],:], r_o[:r.shape[0],:])

    # # C = J_alpha.T @ J_alpha

    # # print(C)
