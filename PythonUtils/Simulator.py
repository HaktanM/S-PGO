import pypose as pp
import numpy as np
import pypose as pp
import torch
import sys

from PythonUtils.StereoSetup import StereoSetup
from PythonUtils.utils import SamplePoses, get_point_samples



class Simulator():
    def __init__(self, n=10, m = 20):
        """
        n: number of camera frames
        m: number of landmarks
        """

        self.n = n
        self.m = m

        self.cam    = StereoSetup()
        self.poses  = SamplePoses(n=self.n).poses
        self.points = get_point_samples(m=self.m)

        self.pixel_noise_std = 2.0
        
        # Compute the observations
        self.compute_observations()

        # Compute the incremental poses
        self.incremental_poses = []
        self.compute_the_incremental_poses()

        

    def compute_observations(self):
        """
        Calculate the observations and add noise to create noisy observations.
        """
        self.observations  = {}
        self.actual_depths = {}
        self.validty = {}

        self.noisy_observations  = {}
        for right in [False, True]:
            for cam_idx in range(self.n):
                for landmark_idx in range(self.m):
                    p, alpha = self.cam.project(T_cam_in_global=self.poses[cam_idx], t_feat_in_global=self.points[landmark_idx], right=right)
                    self.observations.setdefault(cam_idx, {}).setdefault(landmark_idx, {})[right] = p
                    self.actual_depths.setdefault(cam_idx, {}).setdefault(landmark_idx, {})[right] = alpha

                    depth = 1 / alpha
                    if depth<0.2 or p[0,0]>self.cam.width or p[1,0]>self.cam.height:
                        self.validty.setdefault(cam_idx, {}).setdefault(landmark_idx, {})[right] = False
                    else:
                        self.validty.setdefault(cam_idx, {}).setdefault(landmark_idx, {})[right] = True

                    # Add Gaussian noise to simulate noisy observation
                    noise = np.random.normal(loc=0.0, scale=self.pixel_noise_std, size=p.shape)
                    p_noisy = p.copy()
                    p_noisy[:2] = p_noisy[:2] + noise[:2]

                    self.noisy_observations.setdefault(cam_idx, {}).setdefault(landmark_idx, {})[right] = p_noisy


    
    def compute_the_incremental_poses(self):
        T_curr_glob = np.eye(4)
        skip_first = True
        for pose in self.poses:
            if skip_first:
                T_curr_glob = pose
                skip_first=False
                continue
            T_next_glob = pose
            T_curr_next = np.linalg.inv(T_next_glob) @ T_curr_glob
            T_curr_glob = T_next_glob
            self.incremental_poses.append(T_curr_next)
        self.check_incremental_poses()

    def check_incremental_poses(self):
        T_curr_glob = self.poses[0]
        for idx, T_curr_next in enumerate(self.incremental_poses):
            T_curr_glob = T_curr_glob @ np.linalg.inv(T_curr_next)
            discrepeancy = T_curr_glob - self.poses[idx+1]
            error = np.linalg.norm(discrepeancy)
            if error > 1e-4:
                print(f"Error is larger than threshold : {error}/{1e-4}")
                sys.exit()



class DeterministicSimulator:
    def __init__(self):
        self.n = 3
        self.m = 1
        self.cam = StereoSetup()

        # Get the poses of the cameras
        pose_sampler  = SamplePoses()
        pose_sampler.create_samples_deterministic()
        self.poses    = pose_sampler.poses

        self.anchor_idx = 0

        self.pa_hom = np.array([
            320.0, 100.0, 1.0
        ]).reshape(3,1)

        self.actual_depth     = 5.0 # np.random.uniform(1.0, 10.0)
        self.actual_inv_depth = 1 / self.actual_depth

        T_ca_g = self.poses[self.anchor_idx]
        R_ca_g = T_ca_g[:3,:3]
        t_ca_g = T_ca_g[:3,3]

        t_feat_ca = self.cam.Kl_inv @ self.pa_hom / self.actual_inv_depth
        t_feat_g  = (R_ca_g @ t_feat_ca).reshape(3,1) + t_ca_g.reshape(3,1)

        self.points = []
        self.points.append(t_feat_g.reshape(-1))


        # Compute the observations
        self.pixel_noise_std = 2.0
        self.compute_observations()

        # Compute the incremental poses
        self.incremental_poses = []
        self.compute_the_incremental_poses()

    def compute_observations(self):
        """
        Calculate the observations and add noise to create noisy observations.
        """
        self.observations  = {}
        self.actual_depths = {}

        self.noisy_observations  = {}
        for right in [False, True]:
            for cam_idx in range(self.n):
                for landmark_idx in range(self.m):
                    p, alpha = self.cam.project(T_cam_in_global=self.poses[cam_idx], t_feat_in_global=self.points[landmark_idx], right=right)
                    self.observations.setdefault(cam_idx, {}).setdefault(landmark_idx, {})[right] = p
                    self.actual_depths.setdefault(cam_idx, {}).setdefault(landmark_idx, {})[right] = alpha

                    # Add Gaussian noise to simulate noisy observation
                    noise = np.random.normal(loc=0.0, scale=self.pixel_noise_std, size=p.shape)
                    p_noisy = p.copy()
                    p_noisy[:2] = p_noisy[:2] + noise[:2]

                    self.noisy_observations.setdefault(cam_idx, {}).setdefault(landmark_idx, {})[right] = p_noisy

    def compute_the_incremental_poses(self):
        T_curr_glob = np.eye(4)
        skip_first = True
        for pose in self.poses:
            if skip_first:
                T_curr_glob = pose
                skip_first=False
                continue
            T_next_glob = pose
            T_curr_next = np.linalg.inv(T_next_glob) @ T_curr_glob
            T_curr_glob = T_next_glob
            self.incremental_poses.append(T_curr_next)
        self.check_incremental_poses()

    def check_incremental_poses(self):
        T_curr_glob = self.poses[0]
        for idx, T_curr_next in enumerate(self.incremental_poses):
            T_curr_glob = T_curr_glob @ np.linalg.inv(T_curr_next)
            discrepeancy = T_curr_glob - self.poses[idx+1]
            error = np.linalg.norm(discrepeancy)
            if error > 1e-4:
                print(f"Error is larger than threshold : {error}/{1e-4}")
                sys.exit()

if __name__ == "__main__":
    simulator = Simulator(n=3, m=80)
    print(simulator.observations[0][0][False])
    print(simulator.observations[0][0][True])