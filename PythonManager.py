from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import Optimizer
from PythonUtils.SceneRenderer import Renderer
from PythonUtils.visualization_utils import visualize_depth_estimation
import numpy as np

from Optimizer import map_value_to_index
import threading

class Manager():
    def __init__(self):

        # Number of keyframes
        self.n = 10

        # Number of landmarks 
        self.m = 30 * self.n

        # Initialize the simulator
        self.simulator = Simulator(n=self.n+1, m=self.m)
        
        # Initialize the optimizer
        self.optimizer = Optimizer(n=self.n, m=self.m)
        self.optimizer.initialize_estimated_incremental_poses(actual_incremental_poses=self.simulator.incremental_poses)

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

    def compute_estimation_errors(self):
        errors = []
        for idx in range(self.n):
            actual_T = self.simulator.incremental_poses[idx]
            estim_T  = self.optimizer.estimated_incremental_poses[idx]

            error_T = np.linalg.inv(actual_T) @ estim_T
            xi = self.optimizer.LU.Log_SE3(error_T)
            errors.append(np.linalg.norm(xi))
        return errors
    
    def compute_depth_error(self):
        actuals = []
        estimates = []
        errors = []

        for idx in range(self.m):
            # Extract scalar values safely
            anchor_idx = map_value_to_index(v=idx, x=self.optimizer.number_of_landmarks, n=self.optimizer.number_of_keyframes)
            actual_depth    = 1 / self.simulator.actual_depths[anchor_idx][idx][False]
            estimated_depth = 1 / self.optimizer.estimated_inverse_depths[idx]

            actuals.append(actual_depth)
            estimates.append(estimated_depth)
            errors.append(abs(estimated_depth - actual_depth))

        # Print total error
        total_error = np.sum(errors)
        print("Total depth estimation error:", total_error)

   
if __name__ == "__main__":
    manager = Manager()
    manager.visualizer.start_rendering()


