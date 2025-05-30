import pypose as pp
import numpy as np
import sys
import pypose as pp
import torch
import numpy as np


from visualization_utils import *


def map_value_to_index(v, x, n):
    """
    Maps value `v` in range [0, x] to index in range [0, n-1].
    """
    v_clipped = max(0, min(v, x))  # clamp to [0, x]
    index = int((v_clipped / x) * n)
    return min(index, n - 1)  # ensure index is in [0, n-1]



from StereoSetup import StereoSetup
from LieUtils import LieUtils

class Optimizer():
    def __init__(self, n=10, m=80):

        # Get the camera setup, where intrinsics and extrinsics are stored
        self.cam = StereoSetup()

        ### Parameters to be tuned
        self.min_depth = 0.01
        self.max_depth = 20
        self.min_alpha = 1.0 / self.max_depth
        self.max_alpha = 1.0 / self.min_depth

        # Number of incremental poses to be estimated
        self.number_of_keyframes = n

        # Number of landmarks extracted from a single keyframe
        self.number_of_landmarks = m

        # Initialize the inverse depths
        self.estimated_inverse_depths = []
        
        for idx in range(self.number_of_landmarks):
            random_initial_depth = np.random.uniform(self.min_depth, self.max_depth)
            self.estimated_inverse_depths.append( 1 / random_initial_depth )
        

        # Lie Algebra Utils
        self.LU = LieUtils()


        # Optimization parameters
        self.step_size  = 0.5
        self.prev_norm  = 1e10

    
    def initialize_estimated_incremental_poses(self, actual_incremental_poses):
        """
        Initialize the estimated poses around the actual incremental poses
        """
        # Initialize the incremental poses 
        self.estimated_incremental_poses = []

        for idx in range(self.number_of_keyframes):
            noise   = pp.randn_se3(1) * 0.1
            T_noise = noise.Exp().matrix().cpu().numpy().reshape(4,4)

            T_curr_next = actual_incremental_poses[idx]
            T_curr_next_noisy = T_curr_next @ T_noise

            # self.estimated_incremental_poses.append(T_curr_next_noisy)
            self.estimated_incremental_poses.append(np.eye(4))

    def get_estimated_global_poses(self,T_curr_global=None):
        
        # If we don't have any initial condition, initialize with identity
        if T_curr_global is None:
            T_curr_global = np.eye(4)

        # Compute the global poses and append
        estimated_global_poses = []
        estimated_global_poses.append(T_curr_global)
        for T_curr_next in self.estimated_incremental_poses:
            T_curr_global = T_curr_global @ np.linalg.inv(T_curr_next)
            estimated_global_poses.append(T_curr_global)

        return estimated_global_poses
    

    def step_pose_only(self, observations:dict, actual_depths:dict):
        """
        Assume that the depths are given,
        we wish to optimize the incremental poses only for one step.
        """

        # State Dimension
        state_dim = self.number_of_keyframes    * 6


        # For a single landmark, we have more than a single observation
        """
        We have two camera frames. Left and right (x2)
        Each observation brings two measurements (x2)
        Right camera of the anchor frame brings (+1)
        """
        size_of_single_observation = ( 2 * self.number_of_keyframes + 1) * 2
        size_of_single_observation = self.number_of_keyframes * 2


        # Size of total observations
        observation_dimension = self.number_of_landmarks * size_of_single_observation

        # Get the estimated global poses
        estimated_global_poses = self.get_estimated_global_poses()
        

        # Initialize the Jacobian and residual matrices
        J = np.zeros((observation_dimension, state_dim))
        r = np.zeros((observation_dimension,1))

        # Get the pixel coordinates with respect to anchor frame
        J_row_idx = 0
        for right in [False]:
            start_idx = 0 if right else 1
            for projection_idx in range(start_idx, self.number_of_keyframes+1):
                for landmark_idx in range(self.number_of_landmarks):
                    
                    anchor_idx = map_value_to_index(v=landmark_idx, x=self.number_of_landmarks, n=self.number_of_keyframes)

                    # Pose of the anchor frame
                    T_ca_to_g = estimated_global_poses[anchor_idx]   

                    # Homogenous pixel coordinates of the landmark at the anchor frame
                    pa_hom = observations[anchor_idx][landmark_idx][False]
                    alpha  = actual_depths[anchor_idx][landmark_idx][False]  # Inverse depth

                    # Compute the location of the landmark in the global frame
                    t_feat_in_ca     = self.cam.Kl_inv @ pa_hom.reshape(3,1) / alpha
                    t_feat_in_ca_hom = np.append(t_feat_in_ca, 1).reshape(4, 1)

                    t_feat_in_g_hom  = T_ca_to_g @ t_feat_in_ca_hom
                    t_feat_in_g      = t_feat_in_g_hom[:3].reshape(3,1)

            
                    # Jacobian of the measurement with respect to which state
                    for i in range(self.number_of_keyframes):
                        del_pn_del_xi = self.del_d_pn_del_xi(pa_hom=observations[anchor_idx][landmark_idx][False], alpha=actual_depths[anchor_idx][landmark_idx][False], anchor_idx=anchor_idx, projection_idx=projection_idx, i=i, right=right)
                      
                        # Get the estimated camera pose with respect to global reference frame
                        T_cn_to_g = estimated_global_poses[projection_idx]    # Pose of the left cam at time projection_idx
                        
                        # Compute the residual
                        observation   = observations[projection_idx][landmark_idx][right]
                        estimation, _ = self.cam.project(T_cam_in_global=T_cn_to_g, t_feat_in_global=t_feat_in_g, right=right)
                        residual      = observation.reshape(3) - estimation.reshape(3)
                    
                        # Compute the column index
                        J_col_idx = 6 * i

                        
                        # if projection_idx > anchor_idx:
                        #     distance_weight = 1.0 / ((projection_idx - anchor_idx)**2)
                        # else:
                        #     distance_weight = 0.0
                        # residual = distance_weight * residual

                        # If the measurement is not in the image frame, ignore it
                        if observation[0]<0 or observation[1]<0 or observation[0]>self.cam.width or observation[1]>self.cam.height or alpha < 0.2:
                            w = 0.0
                        else:
                            w = 1.0

                        residual = residual * w


                        cauchy_c = 1.0  # Robustness parameter — tune this
                        norm = np.linalg.norm(residual)
                        weight = 1.0 / (1.0 + (norm / cauchy_c) ** 2)
                        sqrt_w = np.sqrt(weight)


                        # Fill the residual and Jacobian matrices
                        r[J_row_idx:J_row_idx+2,0] = w * sqrt_w * residual[:2]
                        J[J_row_idx:J_row_idx+2, J_col_idx:J_col_idx+6] = - w * sqrt_w * del_pn_del_xi[:2, :] 
                    J_row_idx += 2

        # zero_rows = np.sum(np.all(np.abs(J) < 10.0, axis=1))
        # print(f"Zero Jacobian rows: {zero_rows}/{J.shape[0]}")

        # Compute the incremental update to the state
        H = J.T @ J
        g = J.T @ r

        # lambda_damping = 1e-1  # or adapt based on convergence
        # H_damped = H + lambda_damping * np.eye(H.shape[0])

        # H += np.eye(H.shape[0]) * 1e-6 # Regularization term for numeric stability
        delta_state = - np.linalg.solve(H, g)
        delta_state = self.step_size * delta_state

        r_norm = np.linalg.norm(r)  
        if self.prev_norm > r_norm:
            self.step_size *= 1.01
        elif self.prev_norm < r_norm:
            self.step_size *= 0.9     

        self.step_size = max(self.step_size, 0.02)
        self.step_size = min(self.step_size, 10.0)
        print(f"step_size : {self.step_size}, r_norm : {r_norm}")
        self.prev_norm = r_norm

        if np.any(np.isnan(delta_state)):
            print("Divergence detected, resetting or skipping update.")
            return

        # Finally, update the estimated incremental states
        self.update_incremental_states(delta_state=delta_state)


        visualize_jacobian_and_residual_to_cv(J,r)


    def step_depth_only(self, observations:dict, actual_poses:list, actual_depths:dict):
        """
        Assume that the camera poses are given,
        we wish to optimize the inverse depth only for one step.
        """


        # For a single landmark, we have more than a single observation
        """
        We have two camera frames. Left and right (x2)
        Each observation brings two measurements (x2)
        Right camera of the anchor frame brings (+1)
        """
        size_of_single_observation = ( 2 * self.number_of_keyframes + 1) * 2
        size_of_single_observation = self.number_of_keyframes * 2

        # Size of total observations
        observation_dimension = self.number_of_landmarks * size_of_single_observation

        # State is only composed of depths for this case
        state_dim = len(self.estimated_inverse_depths)       

        # Initialize the Jacobian and residual matrices
        J = np.zeros((observation_dimension, state_dim))
        r = np.zeros((observation_dimension,1))



        J_row_idx = 0
        for landmark_idx in range(self.number_of_landmarks):
            
            anchor_idx = 0 # map_value_to_index(v=landmark_idx, x=self.number_of_landmarks, n=self.number_of_keyframes)

            # Pose of the anchor frame
            T_ca_to_g = actual_poses[anchor_idx]   

            # Homogenous pixel coordinates of the landmark at the anchor frame
            pa_hom = observations[anchor_idx][landmark_idx][False]
            alpha  = self.estimated_inverse_depths[landmark_idx]

            actual_alpha = actual_depths[anchor_idx][landmark_idx][False]

            # Compute the location of the landmark in the global frame
            t_feat_in_ca     = self.cam.Kl_inv @ pa_hom.reshape(3,1) / alpha
            t_feat_in_ca_hom = np.append(t_feat_in_ca, 1).reshape(4, 1)

            t_feat_in_g_hom  = T_ca_to_g @ t_feat_in_ca_hom
            t_feat_in_g      = t_feat_in_g_hom[:3].reshape(3,1)
            
            # Iterate through all the observations
            for right in [False]:
                start_idx = 0 if right else 1
                for projection_idx in range(start_idx, self.number_of_keyframes+1):
                    # Get the Jacobian with respect to inverse depth
                    del_pn_del_alpha = self.del_pn_del_alpha(pa_hom=pa_hom, alpha=alpha, poses=actual_poses, anchor_idx=anchor_idx, projection_idx=projection_idx, right=right)
                    

                    if right and projection_idx==0:
                        print("del_pn_del_alpha : \n",del_pn_del_alpha)
                    # Get the camera pose with respect to global reference frame
                    T_cn_to_g = actual_poses[projection_idx]    # Pose of the left cam at time projection_idx
                    
                    # Compute the residual
                    observation   = observations[projection_idx][landmark_idx][right]
                    estimation, _ = self.cam.project(T_cam_in_global=T_cn_to_g, t_feat_in_global=t_feat_in_g, right=right)
                    residual      = observation.reshape(3) - estimation.reshape(3)

                    # Compute the column index
                    J_col_idx = landmark_idx

                    # If the measurement is not in the image frame, ignore it
                    if observation[0]<0 or observation[1]<0 or observation[0]>self.cam.width or observation[1]>self.cam.height or actual_alpha>self.max_alpha or actual_alpha<self.min_alpha:
                        w = 0.0
                    else:
                        w = 1.0

                    residual = residual * w

                    sqrt_w = 1.0

                    cauchy_c = 10.0  # Robustness parameter — tune this
                    norm = np.linalg.norm(residual)
                    weight = 1.0 / (1.0 + (norm / cauchy_c) ** 2)
                    sqrt_w = np.sqrt(weight)

                    

                    # Fill the residual and Jacobian matrices
                    r[J_row_idx:J_row_idx+2,0] = w * sqrt_w * residual[:2]
                    J[J_row_idx:J_row_idx+2, J_col_idx:J_col_idx+1] = - w * sqrt_w * del_pn_del_alpha[:2, :] 

                    J_row_idx += 2
                
        # Compute the incremental update to the state
        H = J.T @ J
        g = J.T @ r

        H += np.eye(H.shape[0]) * 1e-3 # Required for stability
        delta_alpha = - np.linalg.solve(H, g)
        delta_alpha = self.step_size * delta_alpha

        # r_norm = np.linalg.norm(r)  
        # if self.prev_norm > r_norm:
        #     self.step_size *= 1.01
        # elif self.prev_norm < r_norm:
        #     self.step_size *= 0.9     

        # self.step_size = max(self.step_size, 0.02)
        # self.step_size = min(self.step_size, 10.0)
        # print(f"step_size : {self.step_size}, r_norm : {r_norm}")
        # self.prev_norm = r_norm

        if np.any(np.isnan(delta_alpha)):
            print("Divergence detected, resetting or skipping update.")
            return


        self.update_depths(delta_alpha=delta_alpha)
        visualize_jacobian_and_residual_to_cv(J,r)
        visualize_hessian_and_g(H, g)

    def update_incremental_states(self, delta_state):
        # print(f"Norm of update : {np.linalg.norm(delta_state)}")
        for state_idx in range(self.number_of_keyframes):
            row_idx = state_idx * 6
            
            # Get the incremental update 
            delta = delta_state[row_idx:row_idx+6]

            # Get the current state
            T_curr_next = self.estimated_incremental_poses[state_idx]

            # Update the state
            T_curr_next = T_curr_next @ self.LU.Exp_SE3(delta)

            # Rewrite to the array
            self.estimated_incremental_poses[state_idx] = T_curr_next

    def update_depths(self, delta_alpha):
        for idx in range(self.number_of_landmarks):
            self.estimated_inverse_depths[idx] += delta_alpha[idx]
            val = float(self.estimated_inverse_depths[idx])
            clipped_val = max(self.min_alpha, min(val, self.max_alpha))
            self.estimated_inverse_depths[idx] = clipped_val

    def del_d_pn_del_xi(self, pa_hom:np.array, alpha:float, anchor_idx:int, projection_idx:int, i:int, right=False):
        
        """
        Computes the Jacobian of the projected landmark (pn) with respect to the incremental pose (xi_i).

        Parameters:
        ----------
        pa_hom : np.array
            Homogeneous pixel coordinates of the landmark at the anchor frame

        alpha : float
            Inverse depth of the landmark relative to the anchor frame.
            
        anchor_idx : int
            Index of the anchor frame (the frame where the landmark was originally observed).
            
        projection_idx : int
            Index of the projection frame (the frame where the landmark is being reprojected).
            
        i : int
            Index of the pose increment with respect to which the derivative is being computed.
            
        right : bool, optional (default=False)
            If True, computes the derivative from the right side (used in right Jacobians).

        Returns:
        -------
        np.ndarray
            The Jacobian matrix of the landmark projection in the camera frame (pn) with respect to xi_i.
        """

        # The Jacobian is nonzero only if the incremental pose 'i' lies within the causal path 
        # from the anchor frame to the projection frame. 
        # This ensures that only poses that causally influence the landmark projection contribute to the Jacobian.
        is_causal = (anchor_idx <= i < projection_idx)
        if not is_causal:
            del_pn_del_xi = np.zeros((3,6))
            return del_pn_del_xi

        # Get the global poses
        estimated_global_poses = self.get_estimated_global_poses()
        T_ca_to_g = estimated_global_poses[anchor_idx]        # Pose of the anchor frame
        T_ci_to_g = estimated_global_poses[i]                 # Pose of the left cam at time i
        T_cn_to_g = estimated_global_poses[projection_idx]    # Pose of the left cam at time projection_idx

        if right:
            # Adjust for left-to-right baseline
            T_cn_to_g = T_cn_to_g @ self.cam.T_r_l                  

        # Compute the T_ca_to_ci 
        T_ca_to_ci = np.linalg.inv(T_ci_to_g) @ T_ca_to_g

        # Compute the T_ca_to_cn 
        T_ca_to_cn = np.linalg.inv(T_cn_to_g) @ T_ca_to_g

        # Compute the T_ci_to_cn 
        T_ci_to_cn = np.linalg.inv(T_cn_to_g) @ T_ci_to_g

        # Compute the t_feat_in_ca
        t_feat_in_ca     = self.cam.Kl_inv @ pa_hom.reshape(3,1) / alpha
        t_feat_in_ca_hom = np.append(t_feat_in_ca, 1).reshape(4, 1)

        # Compute the t_feat_in_cn and t_feat_in_ci
        t_feat_in_ci_hom = T_ca_to_ci @ t_feat_in_ca_hom
        t_feat_in_cn_hom = T_ca_to_cn @ t_feat_in_ca_hom

        t_feat_in_ci = t_feat_in_ci_hom[:3]
        t_feat_in_cn = t_feat_in_cn_hom[:3]

        # Compute b_feat_cn_estimated
        K = self.cam.Kr if right else self.cam.Kl
        b_feat_in_cn_estimated = K @ t_feat_in_cn.reshape(3,1)

        del_pn_del_b     = self.del_p_del_b(b=b_feat_in_cn_estimated)
        del_b_del_tcn    = self.del_b_del_tcn(right=right)
        del_tcn_del_xi   = self.del_tcn_del_xi(T_ci_cn=T_ci_to_cn, t_feat_ci=t_feat_in_ci)  

        # Finally, we are ready to compute the Jacobian
        del_pn_del_xi    = del_pn_del_b @ del_b_del_tcn @ del_tcn_del_xi
        return del_pn_del_xi


    def del_pn_del_alpha(self, pa_hom:np.array, alpha:float, poses:list, anchor_idx:int, projection_idx:int, right=False):
        
        # Get the global poses
        T_ca_to_g = poses[anchor_idx]        # Pose of the anchor frame
        T_cn_to_g = poses[projection_idx]    # Pose of the left cam at time projection_idx

        # Compute the T_ca_to_cn 
        T_ca_to_cn = np.linalg.inv(T_cn_to_g) @ T_ca_to_g
        R_ca_cn    = T_ca_to_cn[:3, :3].reshape(3,3)

        # Compute the t_feat_in_ca
        t_feat_in_ca     = self.cam.Kl_inv @ pa_hom.reshape(3,1) / alpha
        t_feat_in_ca_hom = np.append(t_feat_in_ca, 1).reshape(4, 1)

        # Compute the t_feat_in_cn
        t_feat_in_cn_hom = T_ca_to_cn @ t_feat_in_ca_hom
        t_feat_in_cn = t_feat_in_cn_hom[:3]

        # Compute b_feat_cn_estimated
        K = self.cam.Kr if right else self.cam.Kl
        b_feat_in_cn_estimated = K @ t_feat_in_cn.reshape(3,1)

        # Compute partial derivatives
        del_pn_del_b      = self.del_p_del_b(b=b_feat_in_cn_estimated)
        del_b_del_tcn     = self.del_b_del_tcn(right=right)
        del_tcn_del_tca   = R_ca_cn
        del_tca_del_alpha = - t_feat_in_ca / alpha

        # Finally, compute the Jacobian with respect to alpha
        del_pn_del_alpha  = del_pn_del_b @ del_b_del_tcn @ del_tcn_del_tca @ del_tca_del_alpha

        return del_pn_del_alpha
    
    def del_p_del_b(self, b:np.array):
        b = b.reshape(3)
        result = np.array([
            1.0/b[2], 0.0, -(b[0]/(b[2]*b[2])),
            0.0, 1.0/b[2], -(b[1]/(b[2]*b[2])),
            0.0,      0.0,                 0.0
        ]).reshape(3,3)
        return result
    
    def del_b_del_tcn(self, right=False):
        result = self.cam.Kr if right else self.cam.Kl
        return result
    
    def del_tcn_del_xi(self, T_ci_cn: np.array, t_feat_ci:np.array):
        # Fill the Jacobian matrix
        R_ci_cn = T_ci_cn[:3, :3].reshape(3,3)
        t_feat_ci = t_feat_ci.reshape(3,1)

        result = np.zeros((3,6))
        result[:3, :3] = - R_ci_cn @ self.LU.skew(t_feat_ci)
        result[:3, 3:] = R_ci_cn

        return result