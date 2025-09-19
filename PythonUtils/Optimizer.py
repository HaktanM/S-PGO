import pypose as pp
import numpy as np
import sys
import pypose as pp
import torch
import numpy as np


from PythonUtils.visualization_utils import *


def compute_cauchy_weigth(res:np.array):
    res = res.reshape(2)
    res_mag = res[0]*res[0] + res[1]*res[1]

    cauchy_weigth = np.sqrt( 1.0 / ( 1.0 + (res_mag / 9.0) ))

    return cauchy_weigth

def map_value_to_index(v, x, n):
    """
    Maps value `v` in range [0, x] to index in range [0, n-1].
    """
    v_clipped = max(0, min(v, x))  # clamp to [0, x]
    index = int((v_clipped / x) * n)
    return min(index, n - 1)  # ensure index is in [0, n-1]



from PythonUtils.StereoSetup import StereoSetup
from PythonUtils.LieUtils import LieUtils

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

        # Number of landmarks
        self.number_of_landmarks = m

        # Initialize the inverse depths
        self.estimated_inverse_depths = []
        
        for idx in range(self.number_of_landmarks):
            random_initial_depth = np.random.uniform(self.min_depth, self.max_depth)
            self.estimated_inverse_depths.append( 1 / random_initial_depth )

        # Lie Algebra Utils
        self.LU = LieUtils()

        # Optimization parameters
        self.step_size  = 1.0
        self.prev_norm  = 1e10
    
    def initialize_estimated_poses(self, actual_poses):
        """
        Initialize the estimated poses around the actual incremental poses
        """
        # Initialize the incremental poses 
        self.estimated_poses = []

        for idx in range(self.number_of_keyframes):
            noise   = pp.randn_se3(1) * 0.1
            T_noise = noise.Exp().matrix().cpu().numpy().reshape(4,4)

            T_c_g = actual_poses[idx]
            T_c_g_noisy = T_c_g @ T_noise

            # self.estimated_poses.append(T_c_g_noisy)
            self.estimated_poses.append(T_c_g_noisy)

    def initialize_estimated_poses_with_identity(self):
        """
        Initialize the estimated poses around the actual incremental poses
        """
        # Initialize the incremental poses 
        self.estimated_poses = []

        for idx in range(self.number_of_keyframes):
            self.estimated_poses.append(np.eye(4))

    def initalize_depth_with_disparity(self, observations:dict):
        
        # Get the extrinsics between stereo setup
        T_r_l  = self.cam.T_r_l
        R_r_l  = T_r_l[:3, :3]
        t_r_l  = T_r_l[:3, 3]

        for landmark_idx in range(self.number_of_landmarks):
            anchor_idx = map_value_to_index(v=landmark_idx, x=self.number_of_landmarks, n=self.number_of_keyframes)

            # Homogenous pixel coordinates of the landmark at the anchor frame
            pl_hom = observations[anchor_idx][landmark_idx][False]
            pr_hom = observations[anchor_idx][landmark_idx][True]

            b_f_l = self.cam.Kl_inv @ pl_hom
            b_f_r = self.cam.Kr_inv @ pr_hom

            N = self.LU.skew(R_r_l @ b_f_r)

            N_b_f_l = N @ b_f_l
            N_t_r_l  = N @ t_r_l

            num   = N_b_f_l.T @ N_t_r_l
            denum = N_b_f_l.T @ N_b_f_l

            depth = num / denum

            self.estimated_inverse_depths[landmark_idx] = 1 / depth

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
    
    def getHessians(self, observations:dict):
        # State Dimension
        state_dim = self.number_of_keyframes    * 6
        
        H_TT = np.zeros((state_dim, state_dim))
        g_TT = np.zeros((state_dim, 1))

        B    = np.zeros((state_dim, self.number_of_landmarks))
        H_aa = np.zeros((self.number_of_landmarks, 1)) # This is a diognal matrix, hence only store the diagonal items
        g_aa = np.zeros((self.number_of_landmarks, 1))

        for landmark_idx in range(self.number_of_landmarks):
                    
            anchor_idx = map_value_to_index(v=landmark_idx, x=self.number_of_landmarks, n=self.number_of_keyframes)

            # Pose of the anchor frame
            T_ca_to_g = self.estimated_poses[anchor_idx]   

            # Homogenous pixel coordinates of the landmark at the anchor frame
            pa_hom = observations[anchor_idx][landmark_idx][False]
            alpha  = self.estimated_inverse_depths[landmark_idx]

            # Compute the location of the landmark in the global frame
            t_feat_in_ca     = self.cam.Kl_inv @ pa_hom.reshape(3,1) / alpha
            t_feat_in_ca_hom = np.append(t_feat_in_ca, 1).reshape(4, 1)

            t_feat_in_g_hom  = T_ca_to_g @ t_feat_in_ca_hom
            t_feat_in_g      = t_feat_in_g_hom[:3].reshape(3,1)

            for projection_idx in range(anchor_idx, self.number_of_keyframes):
                
                # Get the estimated camera pose with respect to global reference frame
                T_cnl_to_g = self.estimated_poses[projection_idx]    # Pose of the left cam at time projection_idx
                
                # Compute the pose of the landmark in left camera frame
                t_feat_in_cnl_hom =  np.linalg.inv(T_cnl_to_g) @ t_feat_in_g_hom
                t_feat_in_cnl     = t_feat_in_cnl_hom[:3].reshape(3,1)

                # Compute the pose of the landmark in right camera frame
                t_feat_in_cnr_hom =  self.cam.T_l_r @ t_feat_in_cnl_hom
                t_feat_in_cnr     = t_feat_in_cnr_hom[:3].reshape(3,1)

                # Compute the column index
                J_col_idx = 6 * projection_idx

                # Compute relative poses
                T_ca_to_cnl = np.linalg.inv(T_cnl_to_g) @ T_ca_to_g
                T_ca_to_cnr = self.cam.T_l_r @ T_ca_to_cnl

                ####################### COMPUTE THE PROJECTION FOR THE LEFT CAMERA FIRST #######################
                
                # Compute the residual
                observation                 = observations[projection_idx][landmark_idx][False]
                estimation, estimated_alpha = self.cam.project(T_cam_in_global=T_cnl_to_g, t_feat_in_global=t_feat_in_g, right=False)
                residual                    = observation.reshape(3) - estimation.reshape(3)

                # If the measurement is not in the image frame, ignore it
                if observation[0]<0 or observation[1]<0 or observation[0]>self.cam.width or observation[1]>self.cam.height or estimated_alpha>self.max_alpha or estimated_alpha<self.min_alpha:
                    w = 0.0
                else:
                    w = 1.0

                w = compute_cauchy_weigth(residual[:2])
                # Compute the Jacobian with respect to estimated pose
                del_pnl_del_tnl = self.cam.Kl @ self.jacobian_of_projection(vec=t_feat_in_cnl)
                del_tnl_del_xin = self.del_tnl_del_xin(t_feat_cnl=t_feat_in_cnl) 
                del_pnl_del_xin = del_pnl_del_tnl @ del_tnl_del_xin

                if projection_idx>anchor_idx:
                    del_meas_del_pose = del_pnl_del_xin[:2, :].reshape(2,6)
                    H_TT[J_col_idx:J_col_idx+6, J_col_idx:J_col_idx+6]  += w * w * del_meas_del_pose.T @ del_meas_del_pose
                    g_TT[J_col_idx:J_col_idx+6, 0]                      += w * w * del_meas_del_pose.T @ residual[:2].reshape(-1)

                    # Compute the Jacobain with respect to estimated inverse depth
                    del_tnl_del_alpha = - T_ca_to_cnl[:3,:3] @ t_feat_in_ca / alpha
                    del_pnl_del_alpha = del_pnl_del_tnl @ del_tnl_del_alpha

                    H_aa[landmark_idx,0] += w * w * del_pnl_del_alpha[:2,:].T @ del_pnl_del_alpha[:2,:]
                    g_aa[landmark_idx,0] += w * w * del_pnl_del_alpha[:2,:].T @ residual[:2].reshape(-1)

                    # Finally, fill the B matrix
                    B[J_col_idx:J_col_idx+6,landmark_idx] += (w * w * del_meas_del_pose.T @ del_pnl_del_alpha[:2,:]).squeeze()

                    

                ####################### COMPUTE THE PROJECTION FOR THE RIGHT CAMERA NOW #######################
                # Compute the residual
                observation                 = observations[projection_idx][landmark_idx][True]
                estimation, estimated_alpha = self.cam.project(T_cam_in_global=T_cnl_to_g, t_feat_in_global=t_feat_in_g, right=True)
                residual                    = observation.reshape(3) - estimation.reshape(3)

                # If the measurement is not in the image frame, ignore it
                if observation[0]<0 or observation[1]<0 or observation[0]>self.cam.width or observation[1]>self.cam.height or estimated_alpha>self.max_alpha or estimated_alpha<self.min_alpha:
                    w = 0.0
                else:
                    w = 1.0
                w = compute_cauchy_weigth(residual[:2])
                # Compute the Jacobian with respect to estimated pose
                del_pnr_del_tnr = self.cam.Kr @ self.jacobian_of_projection(vec=t_feat_in_cnr)
                del_tnr_del_xin = self.cam.T_l_r[:3, :3] @ del_tnl_del_xin
                del_pnr_del_xin = del_pnr_del_tnr @ del_tnr_del_xin

                del_meas_del_pose = del_pnr_del_xin[:2, :].reshape(2,6)
                

                if projection_idx>anchor_idx:
                    H_TT[J_col_idx:J_col_idx+6, J_col_idx:J_col_idx+6] += w * w * del_meas_del_pose.T @ del_meas_del_pose
                    g_TT[J_col_idx:J_col_idx+6, 0] += w * w * del_meas_del_pose.T @ residual[:2].reshape(-1)

                    # if landmark_idx == 0 and projection_idx == 1:
                    #     print(f"g_TT - py : \n{del_meas_del_pose.T @ residual[:2].reshape(-1)}")
                    #     print(f"residual - py : \n{residual}")

                # Compute the Jacobain with respect to estimated inverse depth
                del_tnr_del_alpha = - T_ca_to_cnr[:3,:3] @ t_feat_in_ca / alpha
                del_pnr_del_alpha = del_pnr_del_tnr @ del_tnr_del_alpha

                H_aa[landmark_idx,0] += w * w * del_pnr_del_alpha[:2,:].T @ del_pnr_del_alpha[:2,:]
                g_aa[landmark_idx,0] += w * w * del_pnr_del_alpha[:2,:].T @ residual[:2].reshape(-1)

                if projection_idx>anchor_idx:
                    # Finally, fill the B matrix
                    B[J_col_idx:J_col_idx+6,landmark_idx] += (w * w * del_meas_del_pose.T @ del_pnr_del_alpha[:2,:]).squeeze()

        return H_TT, g_TT, H_aa, g_aa, B
    

    def getJacobiansAndResidual(self, observations:dict):
        # State Dimension
        pose_states_dim  = self.number_of_keyframes * 6
        depth_states_dim = len(self.estimated_inverse_depths)

        # For a single landmark, we have more than a single observation
        size_of_single_observation = self.number_of_keyframes * 4

        # Size of total observations
        observation_dimension = self.number_of_landmarks * size_of_single_observation

        # Initialize the Jacobian and residual matrices
        J_T = np.zeros((observation_dimension, pose_states_dim))
        J_a = np.zeros((observation_dimension, depth_states_dim))
        r   = np.zeros((observation_dimension,1))
        
        J_row_idx = 0
        for landmark_idx in range(self.number_of_landmarks):
                    
            anchor_idx = map_value_to_index(v=landmark_idx, x=self.number_of_landmarks, n=self.number_of_keyframes)

            # Pose of the anchor frame
            T_ca_to_g = self.estimated_poses[anchor_idx]   

            # Homogenous pixel coordinates of the landmark at the anchor frame
            pa_hom = observations[anchor_idx][landmark_idx][False]
            alpha  = self.estimated_inverse_depths[landmark_idx]

            # Compute the location of the landmark in the global frame
            t_feat_in_ca     = self.cam.Kl_inv @ pa_hom.reshape(3,1) / alpha
            t_feat_in_ca_hom = np.append(t_feat_in_ca, 1).reshape(4, 1)

            t_feat_in_g_hom  = T_ca_to_g @ t_feat_in_ca_hom
            t_feat_in_g      = t_feat_in_g_hom[:3].reshape(3,1)

            for projection_idx in range(anchor_idx+1, self.number_of_keyframes):
                
                # Get the estimated camera pose with respect to global reference frame
                T_cnl_to_g = self.estimated_poses[projection_idx]    # Pose of the left cam at time projection_idx
                
                # Compute the pose of the landmark in left camera frame
                t_feat_in_cnl_hom =  np.linalg.inv(T_cnl_to_g) @ t_feat_in_g_hom
                t_feat_in_cnl     = t_feat_in_cnl_hom[:3].reshape(3,1)

                # Compute the pose of the landmark in right camera frame
                t_feat_in_cnr_hom =  self.cam.T_l_r @ t_feat_in_cnl_hom
                t_feat_in_cnr     = t_feat_in_cnr_hom[:3].reshape(3,1)

                # Compute the column index
                J_col_idx = 6 * projection_idx

                # Compute relative poses
                T_ca_to_cnl = np.linalg.inv(T_cnl_to_g) @ T_ca_to_g
                T_ca_to_cnr = self.cam.T_l_r @ T_ca_to_cnl

                ####################### COMPUTE THE PROJECTION FOR THE LEFT CAMERA FIRST #######################
                # Compute the residual
                observation                 = observations[projection_idx][landmark_idx][False]
                estimation, estimated_alpha = self.cam.project(T_cam_in_global=T_cnl_to_g, t_feat_in_global=t_feat_in_g, right=False)
                residual                    = observation.reshape(3) - estimation.reshape(3)

                # If the measurement is not in the image frame, ignore it
                if observation[0]<0 or observation[1]<0 or observation[0]>self.cam.width or observation[1]>self.cam.height or estimated_alpha>self.max_alpha or estimated_alpha<self.min_alpha:
                    w = 0.0
                else:
                    w = 1.0

                # Compute the Jacobian with respect to estimated pose
                del_pnl_del_tnl = self.cam.Kl @ self.jacobian_of_projection(vec=t_feat_in_cnl)
                del_tnl_del_xin = self.del_tnl_del_xin(t_feat_cnl=t_feat_in_cnl) 
                del_pnl_del_xin = del_pnl_del_tnl @ del_tnl_del_xin

                # Compute the Jacobain with respect to estimated inverse depth
                del_tnl_del_alpha = - T_ca_to_cnl[:3,:3] @ t_feat_in_ca / alpha
                del_pnl_del_alpha = del_pnl_del_tnl @ del_tnl_del_alpha

                # Fill the residual and Jacobian matrices
                J_T[J_row_idx:J_row_idx+2, J_col_idx:J_col_idx+6] = w * del_pnl_del_xin[:2, :] 
                J_a[J_row_idx:J_row_idx+2, landmark_idx]          = (w * del_pnl_del_alpha[:2, :]).squeeze() 
                r[J_row_idx:J_row_idx+2,0]                        = w * residual[:2]
                
                J_row_idx += 2
                ####################### COMPUTE THE PROJECTION FOR THE RIGHT CAMERA NOW #######################
                # Compute the residual
                observation                 = observations[projection_idx][landmark_idx][True]
                estimation, estimated_alpha = self.cam.project(T_cam_in_global=T_cnl_to_g, t_feat_in_global=t_feat_in_g, right=True)
                residual                    = observation.reshape(3) - estimation.reshape(3)

                # If the measurement is not in the image frame, ignore it
                if observation[0]<0 or observation[1]<0 or observation[0]>self.cam.width or observation[1]>self.cam.height or estimated_alpha>self.max_alpha or estimated_alpha<self.min_alpha:
                    w = 0.0
                else:
                    w = 1.0

                # Compute the Jacobian with respect to estimated pose
                del_pnr_del_tnr = self.cam.Kr @ self.jacobian_of_projection(vec=t_feat_in_cnr)
                del_tnr_del_xin = self.cam.T_l_r[:3, :3] @ del_tnl_del_xin 
                del_pnr_del_xin = del_pnr_del_tnr @ del_tnr_del_xin

                # Compute the Jacobain with respect to estimated inverse depth
                del_tnr_del_alpha = - T_ca_to_cnr[:3,:3] @ t_feat_in_ca / alpha
                del_pnr_del_alpha = del_pnr_del_tnr @ del_tnr_del_alpha

                # Fill the residual and Jacobian matrices
                J_T[J_row_idx:J_row_idx+2, J_col_idx:J_col_idx+6] = w * del_pnr_del_xin[:2, :] 
                J_a[J_row_idx:J_row_idx+2, landmark_idx]          = (w * del_pnr_del_alpha[:2, :]).squeeze() 
                r[J_row_idx:J_row_idx+2,0]                        = w * residual[:2]
                
                J_row_idx += 2

        return J_T, J_a, r
        

    def step(self, observations:dict):
        A, g_TT, H_aa, g_aa, BB = self.getHessians(observations=observations)

        H_aa = H_aa.reshape(-1) + 0.1
        C_inv = np.diag(1.0 / H_aa)

        B_C_inv = BB @ C_inv 
        H_T_new = A - B_C_inv @ BB.T 
        g_T_new = g_TT.reshape(-1,1) - (B_C_inv @ g_aa).reshape(-1,1)
        
    
        H_T_new += np.eye(H_T_new.shape[0]) * 1.0 # Required for stability
        delta_pose  = np.linalg.solve(H_T_new, g_T_new)
        delta_alpha = C_inv @ (g_aa.reshape(-1,1)  - BB.T @ delta_pose.reshape(-1,1)) 

        # Step size is also an important consept
        delta_pose  = self.step_size * delta_pose
        delta_alpha = self.step_size * delta_alpha

        self.update_estimated_poses(delta_state=delta_pose)
        self.update_depths(delta_alpha=delta_alpha)
        


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
        """
        size_of_single_observation = self.number_of_keyframes * 4


        # Size of total observations
        observation_dimension = self.number_of_landmarks * size_of_single_observation

        # Initialize the Jacobian and residual matrices
        J = np.zeros((observation_dimension, state_dim))
        r = np.zeros((observation_dimension,1))

        H_eff = np.zeros((6*self.number_of_keyframes, 6*self.number_of_keyframes))
        g_eff = np.zeros((6*self.number_of_keyframes,1))

        # Get the pixel coordinates with respect to anchor frame
        J_row_idx = 0
        for landmark_idx in range(self.number_of_landmarks):
                    
            anchor_idx = map_value_to_index(v=landmark_idx, x=self.number_of_landmarks, n=self.number_of_keyframes)

            # Pose of the anchor frame
            T_ca_to_g = self.estimated_poses[anchor_idx]   

            # Homogenous pixel coordinates of the landmark at the anchor frame
            pa_hom = observations[anchor_idx][landmark_idx][False]
            alpha  = actual_depths[anchor_idx][landmark_idx][False]  # Inverse depth

            # Compute the location of the landmark in the global frame
            t_feat_in_ca     = self.cam.Kl_inv @ pa_hom.reshape(3,1) / alpha
            t_feat_in_ca_hom = np.append(t_feat_in_ca, 1).reshape(4, 1)

            t_feat_in_g_hom  = T_ca_to_g @ t_feat_in_ca_hom
            t_feat_in_g      = t_feat_in_g_hom[:3].reshape(3,1)

            for projection_idx in range(anchor_idx+1, self.number_of_keyframes):
                
                # Get the estimated camera pose with respect to global reference frame
                T_cnl_to_g = self.estimated_poses[projection_idx]    # Pose of the left cam at time projection_idx
                
                # Compute the pose of the landmark in left camera frame
                t_feat_in_cnl_hom =  np.linalg.inv(T_cnl_to_g) @ t_feat_in_g_hom
                t_feat_in_cnl     = t_feat_in_cnl_hom[:3].reshape(3,1)

                # Compute the pose of the landmark in RİGHT camera frame
                t_feat_in_cnr_hom =  self.cam.T_l_r @ t_feat_in_cnl_hom
                t_feat_in_cnr     = t_feat_in_cnr_hom[:3].reshape(3,1)

                # Compute the column index
                J_col_idx = 6 * projection_idx

                ####################### COMPUTE THE PROJECTION FOR THE LEFT CAMERA FIRST #######################
    
                # Compute the residual
                observation   = observations[projection_idx][landmark_idx][False]
                estimation, estimated_alpha = self.cam.project(T_cam_in_global=T_cnl_to_g, t_feat_in_global=t_feat_in_g, right=False)
                residual      = observation.reshape(3) - estimation.reshape(3)

                # If the measurement is not in the image frame, ignore it
                if observation[0]<0 or observation[1]<0 or observation[0]>self.cam.width or observation[1]>self.cam.height or estimated_alpha>self.max_alpha or estimated_alpha<self.min_alpha:
                    w = 0.0
                else:
                    w = 1.0

                # Compute the Jacobian
                del_pnl_del_tnl = self.cam.Kl @ self.jacobian_of_projection(vec=t_feat_in_cnl)
                del_tnl_del_xin = self.del_tnl_del_xin(t_feat_cnl=t_feat_in_cnl) 
                del_pnl_del_xin = del_pnl_del_tnl @ del_tnl_del_xin

                # Fill the residual and Jacobian matrices
                r[J_row_idx:J_row_idx+2,0] = residual[:2]
                J[J_row_idx:J_row_idx+2, J_col_idx:J_col_idx+6] = del_pnl_del_xin[:2, :] 

                J_row_idx += 2

                del_meas_del_pose = del_pnl_del_xin[:2, :].reshape(2,6)
                H_eff[J_col_idx:J_col_idx+6, J_col_idx:J_col_idx+6] += w * w * del_meas_del_pose.T @ del_meas_del_pose
                g_eff[J_col_idx:J_col_idx+6, 0] += w * w * del_meas_del_pose.T @ residual[:2].reshape(-1)

                ####################### COMPUTE THE PROJECTION FOR THE RIGHT CAMERA NOW #######################
                # Compute the residual
                observation   = observations[projection_idx][landmark_idx][True]
                estimation, estimated_alpha = self.cam.project(T_cam_in_global=T_cnl_to_g, t_feat_in_global=t_feat_in_g, right=True)
                residual      = observation.reshape(3) - estimation.reshape(3)

                # If the measurement is not in the image frame, ignore it
                if observation[0]<0 or observation[1]<0 or observation[0]>self.cam.width or observation[1]>self.cam.height or estimated_alpha>self.max_alpha or estimated_alpha<self.min_alpha:
                    w = 0.0
                else:
                    w = 1.0

                # Compute the Jacobian
                del_pnr_del_tnl = self.cam.Kr @ self.jacobian_of_projection(vec=t_feat_in_cnr)
                del_tnr_del_xin = self.cam.T_l_r[:3, :3] @ del_tnl_del_xin
                del_pnr_del_xin = del_pnr_del_tnl @ del_tnr_del_xin
            
                # Fill the residual and Jacobian matrices
                r[J_row_idx:J_row_idx+2,0] = residual[:2]
                J[J_row_idx:J_row_idx+2, J_col_idx:J_col_idx+6] = del_pnr_del_xin[:2, :] 

                J_row_idx += 2

                del_meas_del_pose = del_pnr_del_xin[:2, :].reshape(2,6)
                H_eff[J_col_idx:J_col_idx+6, J_col_idx:J_col_idx+6] += w * w * del_meas_del_pose.T @ del_meas_del_pose
                g_eff[J_col_idx:J_col_idx+6, 0] += w * w * del_meas_del_pose.T @ residual[:2].reshape(-1)

        # zero_rows = np.sum(np.all(np.abs(J) < 10.0, axis=1))
        # print(f"Zero Jacobian rows: {zero_rows}/{J.shape[0]}")

        # Compute the incremental update to the state
        H = J.T @ J
        g = J.T @ r

        H_eff += np.eye(H_eff.shape[0]) * 1.0 # Regularization term for numeric stability
        delta_state = np.linalg.solve(H_eff, g_eff)
        delta_state = self.step_size * delta_state

        # Finally, update the estimated incremental states
        self.update_estimated_poses(delta_state=delta_state)

        visualize_jacobian_and_residual_to_cv(J,r)
        visualize_hessian_and_g(H=H_eff,g=g_eff)

    def del_tnl_del_xin(self, t_feat_cnl:np.array):
        del_tnl_del_xin = np.zeros((3,6))
        del_tnl_del_xin[:3, :3] = self.LU.skew(t_feat_cnl)
        del_tnl_del_xin[:3, 3:] = -np.eye(3)
        return del_tnl_del_xin


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
        # size_of_single_observation = self.number_of_keyframes * 2

        # Size of total observations
        observation_dimension = self.number_of_landmarks * size_of_single_observation

        # State is only composed of depths for this case
        state_dim = len(self.estimated_inverse_depths)       

        # Initialize the Jacobian and residual matrices
        J = np.zeros((observation_dimension, state_dim))
        r = np.zeros((observation_dimension,1))



        J_row_idx = 0
        for landmark_idx in range(self.number_of_landmarks):
            
            anchor_idx = map_value_to_index(v=landmark_idx, x=self.number_of_landmarks, n=self.number_of_keyframes)

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
            t_feat_in_g      = t_feat_in_g_hom[:3,0].reshape(3,1)
            
            # Iterate through all the observations
            for right in [False, True]:
                start_idx = 0 if right else 1
                for projection_idx in range(start_idx, self.number_of_keyframes+1):
                    # Get the Jacobian with respect to inverse depth
                    del_pn_del_alpha = self.del_pn_del_alpha(pa_hom=pa_hom, alpha=alpha, poses=actual_poses, anchor_idx=anchor_idx, projection_idx=projection_idx, right=right)
                    
                    # Get the camera pose with respect to global reference frame
                    T_cn_to_g = actual_poses[projection_idx]    # Pose of the left cam at time projection_idx
                    
                    # Compute the residual
                    observation   = observations[projection_idx][landmark_idx][right]
                    estimation, _ = self.cam.project(T_cam_in_global=T_cn_to_g, t_feat_in_global=t_feat_in_g, right=right)
                    residual      = observation.reshape(3) - estimation.reshape(3)
                    
                    # Compute the column index
                    J_col_idx = landmark_idx

                    # # If the measurement is not in the image frame, ignore it
                    # if observation[0]<0 or observation[1]<0 or observation[0]>self.cam.width or observation[1]>self.cam.height or actual_alpha>self.max_alpha or actual_alpha<self.min_alpha:
                    #     w = 0.0
                    # else:
                    #     w = 1.0

                    # if _<0.0:
                    #     w = 0.0

                    # residual = residual * w

                    # sqrt_w = 1.0

                    # cauchy_c = 10.0  # Robustness parameter — tune this
                    # norm = np.linalg.norm(residual)
                    # weight = 1.0 / (1.0 + (norm / cauchy_c) ** 2)
                    # sqrt_w = np.sqrt(weight)

                    w      = 1.0
                    sqrt_w = 1.0

                    # Fill the residual and Jacobian matrices
                    r[J_row_idx:J_row_idx+2,0] = w * sqrt_w * residual[:2]
                    J[J_row_idx:J_row_idx+2, J_col_idx:J_col_idx+1] = w * sqrt_w * del_pn_del_alpha[:2, :] 

                    J_row_idx += 2
        
        # Compute the incremental update to the state
        H = J.T @ J
        g = J.T @ r

        H += np.eye(H.shape[0]) * 1e-3 # Required for stability
        delta_alpha = np.linalg.solve(H, g)
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

    def update_estimated_poses(self, delta_state):
        # print(f"Norm of update : {np.linalg.norm(delta_state)}")
        for state_idx in range(self.number_of_keyframes):
            row_idx = state_idx * 6
            
            # Get the incremental update 
            delta = delta_state[row_idx:row_idx+6]

            # Get the current state
            T_c_g = self.estimated_poses[state_idx]

            # Update the state
            T_c_g = T_c_g @ self.LU.Exp_SE3(delta)

            # Rewrite to the array
            self.estimated_poses[state_idx] = T_c_g

    def update_depths(self, delta_alpha):
        for idx in range(self.number_of_landmarks):
            self.estimated_inverse_depths[idx] += delta_alpha[idx]
            val = float(self.estimated_inverse_depths[idx])
            clipped_val = max(self.min_alpha, min(val, self.max_alpha))
            self.estimated_inverse_depths[idx] = clipped_val


    def del_pn_del_alpha(self, pa_hom:np.array, alpha:float, poses:list, anchor_idx:int, projection_idx:int, right=False):
        
        # Get the global poses
        T_ca_to_g = poses[anchor_idx]        # Pose of the anchor frame
        T_cn_to_g = poses[projection_idx]    # Pose of the left cam at time projection_idx

        if right:
            # Adjust for left-to-right baseline
            T_cn_to_g = T_cn_to_g @ self.cam.T_r_l   

        # Compute the T_ca_to_cn 
        T_ca_to_cn = np.linalg.inv(T_cn_to_g) @ T_ca_to_g
        R_ca_to_cn = T_ca_to_cn[:3,:3]
        t_ca_in_cn = T_ca_to_cn[:3,3]

        # Compute the t_feat_in_ca
        t_feat_in_ca     = self.cam.Kl_inv @ pa_hom.reshape(3,1) / alpha

        # Compute the t_feat_in_cn
        t_feat_in_cn = (R_ca_to_cn @ t_feat_in_ca).reshape(3,1) + t_ca_in_cn.reshape(3,1)

        # Get intrinsic matrix
        K = self.cam.Kr if right else self.cam.Kl

        # Compute partial derivatives
        del_pcn_del_bcn   = K
        del_bcn_del_tcn   = self.jacobian_of_projection(vec=t_feat_in_cn)
        del_tcn_del_tca   = R_ca_to_cn
        del_tca_del_alpha = - t_feat_in_ca / alpha

        # Finally, compute the Jacobian with respect to alpha
        del_pn_del_alpha  = del_pcn_del_bcn @ del_bcn_del_tcn @ del_tcn_del_tca @ del_tca_del_alpha

        return del_pn_del_alpha
    

    def jacobian_of_projection(self, vec:np.array):
        vec = vec.reshape(3)
        result = np.array([
            1.0/vec[2], 0.0, -(vec[0]/(vec[2]*vec[2])),
            0.0, 1.0/vec[2], -(vec[1]/(vec[2]*vec[2])),
            0.0,      0.0,                 0.0
        ]).reshape(3,3)
        return result