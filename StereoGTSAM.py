# This code script is adopted from the official example implementation 
# You can find further deails in https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/StereoVOExample.ipynb

import gtsam
from gtsam.symbol_shorthand import X, L

import numpy as np

from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import map_value_to_index
import time

class GTSAMmanager():
    def __init__(self, n, m):
        # Number of keyframes
        self.n = n

        # Number of landmarks per frame
        self.m = m

        # Total number of landmarks
        self.M = self.n * self.m

        ### Inverse depth limits
        self.min_depth = 0.01
        self.max_depth = 20
        self.min_alpha = 1.0 / self.max_depth
        self.max_alpha = 1.0 / self.min_depth

        # Initialize the simulator
        self.simulator = Simulator(n=self.n, m=self.M)
        
        # Get intrinsics and extrinsics
        # Here we strictly assume that the left and right camera frames are rectified. 
        fx, fy, cx, cy = self.simulator.cam.Kl[0,0], self.simulator.cam.Kl[1,1], self.simulator.cam.Kl[0,2], self.simulator.cam.Kl[1,2]
        baseline = -self.simulator.cam.T_l_r[0,3]
        self.K = gtsam.Cal3_S2Stereo(fx, fy, 0, cx, cy, baseline)

        # Define the measurement noise
        self.mu_z = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        
        # Initialize the graph
        self.graph = gtsam.NonlinearFactorGraph()

        

        # Get the observations
        self.loadObservations()

        # Initialize the estimation parameters
        self.initializeEstimates()

        params = gtsam.LevenbergMarquardtParams()
        # params.setDiagonalDamping(0.1)
        # params.setlambdaInitial(0.5)
        # params.setlambdaUpperBound(2e10)
        # params.setlambdaLowerBound(0.1)
        # params.setlambdaFactor(1.2)
        # params.setVerbosity("SILENT")
        # params.setErrorTol(0)
        # params.setRelativeErrorTol(0)

        params.setlambdaInitial(10000.0)
        params.setlambdaUpperBound(10000.0)
        params.setlambdaLowerBound(10000.0)
        params.setlambdaFactor(1.0)
        params.setVerbosity("SILENT")
        params.setErrorTol(0)
        params.setRelativeErrorTol(0)

        # params.setErrorTol(0.0)
        # params.setAbsoluteErrorTol(0)
        params.setMaxIterations(1)

        # params.print()
        # print("----------")
        #params.setAbsoluteErrorTol(0)
        # Set up the 
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, params=params)

    def loadObservations(self):
        # Add prior factor
        T_cam_to_glob = self.simulator.poses[0]
        self.first_pose = gtsam.Pose3(gtsam.Rot3(T_cam_to_glob[0,0], T_cam_to_glob[0,1], T_cam_to_glob[0,2], T_cam_to_glob[1,0], T_cam_to_glob[1,1], T_cam_to_glob[1,2], T_cam_to_glob[2,0], T_cam_to_glob[2,1], T_cam_to_glob[2,2]), gtsam.Point3(T_cam_to_glob[0,3],T_cam_to_glob[1,3],T_cam_to_glob[2,3]))
        self.graph.add(gtsam.NonlinearEqualityPose3(X(1), self.first_pose))


        counter = 0
        for landmark_idx in range(self.M): 
            anchor_idx = map_value_to_index(v=landmark_idx, x=self.M, n=self.n)
            for projection_idx in range(anchor_idx, self.n):
                if True: # self.simulator.validty[projection_idx][landmark_idx][False] and self.simulator.validty[projection_idx][landmark_idx][True]:

                    left_obs_py  = self.simulator.observations[projection_idx][landmark_idx][False].reshape(3).astype(np.float32)
                    right_obs_py = self.simulator.observations[projection_idx][landmark_idx][True].reshape(3).astype(np.float32)
                    
                    # print(f"{left_obs_py[0]}, {right_obs_py[0]}, {left_obs_py[1]}")
                    self.graph.add(gtsam.GenericStereoFactor3D(
                        gtsam.StereoPoint2(left_obs_py[0], right_obs_py[0], left_obs_py[1]), self.mu_z, X(projection_idx+1), L(landmark_idx+1), self.K
                    ))
                    gtsam.StereoPoint2()
                    self.meas_size = counter * 4

    def initializeEstimates(self):
        self.initial_estimate = gtsam.Values()


        for pose_idx in range(self.n):
            # Get pose
            T_cam_to_glob = self.simulator.poses[pose_idx]
            # self.initial_estimate.insert(X(pose_idx+1), gtsam.Pose3(gtsam.Rot3(T_cam_to_glob[0,0], T_cam_to_glob[0,1], T_cam_to_glob[0,2], T_cam_to_glob[1,0], T_cam_to_glob[1,1], T_cam_to_glob[1,2], T_cam_to_glob[2,0], T_cam_to_glob[2,1], T_cam_to_glob[2,2]), gtsam.Point3(T_cam_to_glob[0,3],T_cam_to_glob[1,3],T_cam_to_glob[2,3])))
            self.initial_estimate.insert(X(pose_idx+1), self.first_pose)


        for landmark_idx in range(self.M):
            t_land_in_glob = self.simulator.points[landmark_idx]
            # print(gtsam.Point3(t_land_in_glob[0], t_land_in_glob[1], t_land_in_glob[2]))
            self.initial_estimate.insert(L(landmark_idx+1), gtsam.Point3(t_land_in_glob[0], t_land_in_glob[1], t_land_in_glob[2]))
            # self.initial_estimate.insert(L(landmark_idx+1), gtsam.Point3())

        




if __name__ == "__main__":
    

    # gtsam_manager = GTSAMmanager(n=3, m=50)
    # result = gtsam_manager.optimizer.optimize()

    # optimized_pose1 = gtsam_manager.initial_estimate.atPose3(X(1))
    # print("\nOptimized Pose X(1):\n", optimized_pose1)

    # print(f"Initial Error: {gtsam_manager.graph.error(gtsam_manager.initial_estimate):.4f}")
    # print(f"Final Error  : {gtsam_manager.graph.error(result):.4f}")
    # print(f"Iterations   : {gtsam_manager.optimizer.iterations()}")

    # # Define the test range

    # How many run to test
    number_of_trials = 25

    # Number of keyframes
    number_of_poses = [3,  6,  9,  12]

    # Number of landmarks per keyframe
    number_of_feats = [24, 48, 72, 96]


    # GTSAM Times will be stored here
    gtsam_times = {}
    for n in number_of_poses:
        gtsam_times.update({n:{}})
        for m in number_of_feats:
            gtsam_manager = GTSAMmanager(n=n, m=n)

            gtsam_times[n].update({m:{}})
            gtsam_times[n][m].update({"meas_size":gtsam_manager.meas_size})
            gtsam_times[n][m].update({"elapsed_times":[]})

            
    
            for _ in range(number_of_trials):
                gtsam_manager = GTSAMmanager(n=n, m=n)

                t_start = time.monotonic_ns()
                gtsam_manager.optimizer.optimize()
                t_stop = time.monotonic_ns()

                # print(f"Initial Error: {gtsam_manager.graph.error(gtsam_manager.initial_estimate):.4f}")
                # print(f"Final Error  : {gtsam_manager.graph.error(result):.4f}")
                
                elapsed_time = (t_stop-t_start)

                print(f"{n} {m} {elapsed_time} {gtsam_manager.optimizer.iterations()}")

                gtsam_times[n][m]["elapsed_times"].append(elapsed_time)


    line = ["Elapsed Time GTSAM:"]
    for n in number_of_poses:
        for m in number_of_feats:
            elapsed_time = np.median(np.array(gtsam_times[n][m]["elapsed_times"])) * 1e-6
            elapsed_time = f"{elapsed_time:.2f}"
            line = line + ["& " + elapsed_time + " "]


    print("".join(line))