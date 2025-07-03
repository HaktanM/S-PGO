from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import Optimizer
from PythonUtils.SceneRenderer import Renderer
from PythonUtils.visualization_utils import visualize_depth_estimation
import numpy as np
import time

from PythonUtils.Optimizer import map_value_to_index
import threading

class Manager():
    def __init__(self):

        # Number of keyframes
        self.n = 10

        # Number of landmarks 
        self.m = 30 * self.n

        # Initialize the simulator
        self.simulator = Simulator(n=self.n, m=self.m)
        
        # Initialize the optimizer
        self.optimizer = Optimizer(n=self.n, m=self.m)
        self.optimizer.initialize_estimated_poses(actual_poses=self.simulator.poses)

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
        estimated_poses = self.optimizer.get_estimated_global_poses(T_c0_g=self.simulator.poses[0])
        for estimated_pose in estimated_poses:
            self.visualizer.estimated_cam_frames.append(estimated_pose)

    def compute_estimation_errors(self):
        errors = []
        for idx in range(self.n):
            actual_T = self.simulator.poses[idx]
            estim_T  = self.simulator.poses[0] @ self.optimizer.estimated_poses[idx]

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

        visualize_depth_estimation(actual_depths=actuals, estimated_depths=estimates)

        # Print total error
        total_error = np.sum(errors)
        print("Total depth estimation error:", total_error)

    def pose_only_optimization_loop(self):
        while True:
            self.optimizer.step_pose_only(observations=self.simulator.observations, actual_depths=self.simulator.actual_depths)
            errors = self.compute_estimation_errors()
            print(np.array(errors))
            time.sleep(1.0)

            # Update the visualization as well
            estimated_poses = self.optimizer.get_estimated_global_poses(T_c0_g=self.simulator.poses[0])
            for idx in range(len(self.visualizer.estimated_cam_frames)):
                self.visualizer.estimated_cam_frames[idx] = estimated_poses[idx]


    def optimization_loop(self):
        while True:
            self.optimizer.step(observations=self.simulator.observations)
            errors = self.compute_estimation_errors()
            print(np.array(errors))
            time.sleep(1.0)

            # Update the visualization as well
            estimated_poses = self.optimizer.get_estimated_global_poses(T_c0_g=self.simulator.poses[0])
            for idx in range(len(self.visualizer.estimated_cam_frames)):
                self.visualizer.estimated_cam_frames[idx] = estimated_poses[idx]

            self.compute_depth_error()
   
if __name__ == "__main__":
    manager = Manager()

    # Create and start the background thread
    worker = threading.Thread(
        target=manager.pose_only_optimization_loop,
        daemon=True            # ensures the thread won’t block process exit
    )
    worker.start()

    manager.visualizer.start_rendering()

    worker.join()

