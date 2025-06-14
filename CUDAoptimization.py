from build import Solver
import numpy as np

from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import map_value_to_index

import time

from PythonUtils.visualization_utils import visualize_jacobian_and_residual_to_cv, visualize_hessian_and_g

from PythonUtils.Optimizer import Optimizer
from PythonUtils.SceneRenderer import Renderer

import threading

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

        # # Load estimated inverse depths to the solver
        # self.loadInverseDepths()

        # Load estimated incremental poses to solver
        self.loadIncrementalPoses()

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
        estimated_poses = self.optimizer.get_estimated_global_poses(T_curr_global=self.simulator.poses[0])
        for estimated_pose in estimated_poses:
            self.visualizer.estimated_cam_frames.append(estimated_pose)

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
            self.solver.writeIncrementalPose(idx, incremental_pose.astype(np.float32).reshape(16))

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
            actual_T = self.simulator.incremental_poses[idx]
            estim_T  = self.solver.getIncrementalPose(idx)

            error_T = np.linalg.inv(actual_T) @ estim_T
            xi = self.optimizer.LU.Log_SE3(error_T)
            errors.append(np.linalg.norm(xi))
        return errors

    def update_visualizer(self, T_curr_global = None):
        # If we don't have any initial condition, initialize with identity
        if T_curr_global is None:
            T_curr_global = np.eye(4)

        print(T_curr_global)
        # Compute the global poses and append
        for idx in range(self.n):

            T_curr_next = self.solver.getIncrementalPose(idx)
            T_curr_global = T_curr_global @ np.linalg.inv(T_curr_next)
            self.visualizer.estimated_cam_frames[idx] = T_curr_global

    def optimization_loop(self): 
        time.sleep(3)
        while True:
            self.solver.step(1)
            errors = self.compute_estimation_errors()
            print(np.array(errors))
            self.update_visualizer(T_curr_global = self.simulator.poses[0])
            time.sleep(1)


if __name__ == "__main__":
    manager = Manager()


    worker = threading.Thread(
        target=manager.optimization_loop,
        daemon=True            # ensures the thread won’t block process exit
    )
    worker.start()

    manager.visualizer.start_rendering()

    worker.join()

        

    