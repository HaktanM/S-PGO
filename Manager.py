from Simulator import Simulator
from Optimizer import Optimizer
from SceneRenderer import Renderer
from visualization_utils import visualize_depth_estimation
import numpy as np

import threading

class Manager():
    def __init__(self):

        # Number of keyframes
        self.n = 1

        # Number of landmarks 
        self.m = 10 * self.n

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

    def step_pose_only(self):
        self.optimizer.step_pose_only(observations=self.simulator.observations, actual_depths=self.simulator.actual_depths)

        errors = self.compute_estimation_errors()
        msg = 'Errors : ' + ', '.join([f"{item:.06f}" for item in errors])
        print(msg)

        # Update the visualization as well
        estimated_poses = self.optimizer.get_estimated_global_poses(T_curr_global=self.simulator.poses[0])
        for idx in range(len(self.visualizer.estimated_cam_frames)):
            self.visualizer.estimated_cam_frames[idx] = estimated_poses[idx]

    def pose_only_optimization_loop(self):
        while True:
            self.step_pose_only()
            import time
            time.sleep(0.01)


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
            actual_depth    = 1 / self.simulator.actual_depths[0][idx][False]
            estimated_depth = 1 / self.optimizer.estimated_inverse_depths[idx]

            actuals.append(actual_depth)
            estimates.append(estimated_depth)
            errors.append(abs(estimated_depth - actual_depth))

        # Print total error
        total_error = np.sum(errors)
        print("Total depth estimation error:", total_error)

        visualize_depth_estimation(actual_depths=actuals, estimated_depths=estimates)



    def step_depth_only(self):
        self.optimizer.step_depth_only(observations=self.simulator.observations, actual_poses=self.simulator.poses, actual_depths=self.simulator.actual_depths)


# if __name__ == "__main__":
#     manager = Manager()

#     # Create and start the background thread
#     worker = threading.Thread(
#         target=manager.pose_only_optimization_loop,
#         daemon=True            # ensures the thread won’t block process exit
#     )
#     worker.start()

#     manager.visualizer.start_rendering()

#     worker.join()


if __name__ == "__main__":
    manager = Manager()

    for idx in range(manager.m):
        actual_inverse_depth = manager.simulator.actual_depths[0][idx][False]
        actual_depth = 1 / actual_inverse_depth
        noisy_depth = actual_depth + np.random.uniform(10.0, 12.5)
        noisy_depth = max(0.001, noisy_depth)
        manager.optimizer.estimated_inverse_depths[idx] = 1 / noisy_depth
    while True:
        manager.step_depth_only()
        manager.compute_depth_error()

        # for item in manager.optimizer.estimated_inverse_depths:
        #     print(item)