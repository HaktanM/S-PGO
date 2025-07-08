from build import Solver
from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import Optimizer
from PythonUtils.SceneRenderer import Renderer
from PythonUtils.LieUtils import LieUtils
from PythonUtils.Optimizer import map_value_to_index
from PythonUtils.visualization_utils import visualize_hessian_and_g

import time
import threading
import numpy as np


class Manager():
    def __init__(self):

        # Number of keyframes
        self.n = 2

        # NUmber of landmarks per frame
        self.m = 2

        # Number of total landmarks 
        self.M = self.n * self.m

        # Initialize the simulator
        self.simulator = Simulator(n=self.n+1, m=self.M)

        # Initilize the CUDA solver
        self.solver    = Solver.CudaSolver(self.n, self.m)

        # We have a Python Implementation for comparison
        self.optimizer = Optimizer(n=self.n, m=self.M)
        self.optimizer.initialize_estimated_poses_with_identity()

        # Lie Algebra Utils
        self.LU = LieUtils()

        # Load calibration to solver
        self.loadCalibration()

        # Load Observations to solver
        self.loadObservations()

        # Load estimated inverse depths to the solver
        self.loadInverseDepths()

        # Load estimated incremental poses to solver
        self.loadPoses()

        # Finally, initialize the visualizer
        self.initialize_the_visualizer()

    def initialize_the_visualizer(self):
        # Initialize the visualizer
        self.visualizer = Renderer()

        # Add our frames to visualizer
        for actual_pose in self.simulator.poses:
            self.visualizer.cam_frames.append(actual_pose)

        # Add landmarks to our visualizer
        for point in self.simulator.points:
            self.visualizer.landmarks.append(point)

        # Add estimated poses as es well
        for _ in range(self.n):
            self.visualizer.estimated_cam_frames.append(np.eye(4))

    def loadObservations(self):
        for landmark_idx in range(self.M): 
            anchor_idx = map_value_to_index(v=landmark_idx, x=self.M, n=self.n)
            for projection_idx in range(anchor_idx, self.n+1):
                if True: #self.simulator.validty[projection_idx][landmark_idx][False] and self.simulator.validty[projection_idx][landmark_idx][True]:
                    left_obs_py  = self.simulator.observations[projection_idx][landmark_idx][False].reshape(3).astype(np.float32)
                    right_obs_py = self.simulator.observations[projection_idx][landmark_idx][True].reshape(3).astype(np.float32)
                    self.solver.writeObservations(anchor_idx, projection_idx, landmark_idx, left_obs_py, right_obs_py)

    def loadPoses(self):
        for idx in range(self.n):
            self.solver.writePose(idx, self.optimizer.estimated_poses[idx].reshape(16).astype(np.float32))

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


    def compute_estimation_errors(self):
        errors = []
        for idx in range(self.n):
            actual_T = self.simulator.poses[idx]
            estim_T  = self.solver.getPose(idx)
            error_T = np.linalg.inv(actual_T) @ estim_T
            xi = self.LU.Log_SE3(error_T)
            errors.append(np.linalg.norm(xi))
        return errors

    def get_estimated_global_poses(self,T_c0_g=None):
        # If we don't have any initial condition, initialize with identity
        if T_c0_g is None:
            T_c0_g = np.eye(4)

        # Compute the global poses and append
        estimated_global_poses = []
        estimated_global_poses.append(T_c0_g)
        for T_c_c0 in self.estimated_poses:
            T_c_g = T_c0_g @ T_c_c0
            estimated_global_poses.append(T_c_g)
        return estimated_global_poses

if __name__ == "__main__":
    manager = Manager()

    # Get the optimization parameters from Python
    H_TT, g_TT, H_aa, g_aa, BB = manager.optimizer.getHessians(observations=manager.simulator.observations)

    # Get Hessian from CUDA
    manager.solver.step(1)
    H_T = np.loadtxt("H_T.txt", delimiter=",")
    g_T = np.loadtxt("g_T.txt", delimiter=",")

    # Compare two results
    visualize_hessian_and_g(H_T, g_T)
    visualize_hessian_and_g(H_TT, g_TT)

    # print(0.01 * H_TT[6:, 6:])