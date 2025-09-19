from build_gp import Solver
from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import Optimizer
from PythonUtils.SceneRenderer import Renderer
from PythonUtils.LieUtils import LieUtils
from PythonUtils.Optimizer import map_value_to_index
from PythonUtils.visualization_utils import visualize_hessian_and_g
from PythonUtils.visualization_utils import visualize_depth_estimation
import time
import threading
import numpy as np


class Manager():
    def __init__(self, n=5, m=30):

        # Number of keyframes
        self.n = n

        # NUmber of landmarks per frame
        self.m = m 

        # Number of total landmarks 
        self.M = self.n * self.m

        # Initialize the simulator
        self.simulator = Simulator(n=self.n, m=self.M)

        # Initilize the CUDA solver
        self.solver    = Solver.CudaSolver(self.n, self.m)

        # We have a Python Implementation for comparison
        self.PyOptimizer = Optimizer(n=self.n, m=self.M)
        self.PyOptimizer.initialize_estimated_poses_with_identity()

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

        counter = 0
        for landmark_idx in range(self.M): 
            anchor_idx = map_value_to_index(v=landmark_idx, x=self.M, n=self.n)
            for projection_idx in range(anchor_idx, self.n):
                if True: # self.simulator.validty[projection_idx][landmark_idx][False] and self.simulator.validty[projection_idx][landmark_idx][True]:
                    left_obs_py  = self.simulator.observations[projection_idx][landmark_idx][False].reshape(3).astype(np.float32)
                    right_obs_py = self.simulator.observations[projection_idx][landmark_idx][True].reshape(3).astype(np.float32)
                    self.solver.writeObservations(anchor_idx, projection_idx, landmark_idx, left_obs_py, right_obs_py)
                    counter += 1

        self.meas_size = counter * 4

        print(f"Counter : {counter}")
    def loadPoses(self):
        for idx in range(self.n):
            self.solver.writePose(idx, self.PyOptimizer.estimated_poses[idx].reshape(16).astype(np.float32))

    def loadCalibration(self):
        intrinsics = np.array([
            self.simulator.cam.Kl[0,0], self.simulator.cam.Kl[1,1], self.simulator.cam.Kl[0,2], self.simulator.cam.Kl[1,2],
            self.simulator.cam.Kr[0,0], self.simulator.cam.Kr[1,1], self.simulator.cam.Kr[0,2], self.simulator.cam.Kr[1,2]
        ]).reshape(8).astype(np.float32)

        T_r_to_l = self.simulator.cam.T_r_l.reshape(16).astype(np.float32)

        self.solver.loadCalibration(intrinsics, T_r_to_l)

    def loadInverseDepths(self):
        inverse_depths = np.array(self.PyOptimizer.estimated_inverse_depths).reshape(-1).astype(np.float32)
        self.solver.loadInverseDepths(inverse_depths)


    def compute_estimation_errors(self):
        errors = []
        for idx in range(self.n):
            actual_T = self.simulator.poses[idx]
            estim_T  = self.simulator.poses[0] @ self.solver.getPose(idx)
            error_T = np.linalg.inv(actual_T) @ estim_T
            xi = self.LU.Log_SE3(error_T)
            errors.append(np.linalg.norm(xi))
        return errors
    
    def compute_depth_error(self):
        actuals = []
        estimates = []
        errors = []

        estimated_inverse_depths = self.solver.getInverseDepths()
        for idx in range(self.M):
            # Extract scalar values safely
            anchor_idx = map_value_to_index(v=idx, x=self.PyOptimizer.number_of_landmarks, n=self.PyOptimizer.number_of_keyframes)
            actual_depth    = 1 / self.simulator.actual_depths[anchor_idx][idx][False]
            estimated_depth = 1 / estimated_inverse_depths[idx]

            actuals.append(actual_depth)
            estimates.append(estimated_depth)
            errors.append(abs(estimated_depth - actual_depth))

        visualize_depth_estimation(actual_depths=actuals, estimated_depths=estimates)

        # # Print total error
        # total_error = np.sum(errors)
        # print("Total depth estimation error:", total_error)

    def visualize_estimated_poses(self):
        # Compute the global poses and append
        for idx in range(self.n):
            estim_T  = self.simulator.poses[0] @ self.solver.getPose(idx)
            self.visualizer.estimated_cam_frames[idx] = estim_T

    def optimization_loop(self): 
        counter = 0
        while True:
            self.solver.step(1)
            errors = self.compute_estimation_errors()
            print(np.array(errors).sum())
            self.visualize_estimated_poses()
            time.sleep(0.5)
            # if counter<50:
            #     counter += 1
            # elif counter == 50:
            #     print("Setting The Step Size")
            #     self.solver.setStepSize(1.0)
            #     counter += 1

if __name__ == "__main__":
    manager = Manager()

    # Get Hessian from CUDA
    
    manager.compute_depth_error()
    time.sleep(2)
    
    for _ in range(100):

        manager.compute_depth_error()
        time.sleep(2)

        t_start = time.monotonic_ns()
        manager.solver.step(10)
        t_stop = time.monotonic_ns()
        elsapsed_time = (t_stop - t_start) * 1e-6
        print(f"elsapsed_time : {elsapsed_time} ms")
        errors = manager.compute_estimation_errors()
        print(np.array(errors).max())

        manager.compute_depth_error()
        time.sleep(1)
