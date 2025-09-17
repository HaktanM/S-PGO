import numpy as np

class StereoSetup():
    """
    This class stores the camera intrinsics and extrinsics
    """
    def __init__(self):

        self.width  = 640
        self.height = 512
        # Intrinsics for left and right cameras (3x3 matrices)
        self.Kl = np.array([[320.0, 0.0, 320.0],
                            [0.0, 320.0, 240.0],
                            [0.0, 0.0, 1.0]]).reshape(3,3)
        
        # self.Kr = self.Kl.copy()
        self.Kr = np.array([[320.0, 0.0, 320.0],
                            [0.0, 320.0, 240.0],
                            [0.0, 0.0, 1.0]]).reshape(3,3)
        
        self.Kl_inv = np.linalg.inv(self.Kl)
        self.Kr_inv = np.linalg.inv(self.Kr)
        
        # Pose of left camera in right camera frame
        # self.T_l_r = pp.randn_SE3(1).matrix().cpu().numpy().reshape(4,4)
        self.T_l_r = np.array([
            1.0, 0.0, 0.0, -0.2,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]).reshape(4,4)

        self.T_r_l = np.linalg.inv(self.T_l_r) 
    

    def project(self, T_cam_in_global:np.array, t_feat_in_global:np.array, right=False):
        """
        Project a 3D point in the global frame to 2D pixel coordinates in the image plane.
        
        T_cam_in_global: (4,4) Pose of the camera frame in global coordinate system
        t_feat_global  : (3,) 3D point in the global frame (x, y, z)
        Returns: 
            p: (3,) 2D homogeneous pixel coordinate in image plane
            alpha: inverse depth (scalar)
        """


        # Get the camera pose in global frame
        T_c_g = T_cam_in_global.copy()
        if right:
            # Adjust for left-to-right baseline
            T_c_g = T_c_g @ self.T_r_l                  

        # Convert the 3D point in the global frame to homogeneous coordinates
        t_feat_in_global_hom = np.append(t_feat_in_global, 1.0).reshape(4, 1)  # (4,1)

        # Convert the point from global frame to corresponding camera frame
        t_feat_in_c_hom      = np.linalg.inv(T_c_g) @ t_feat_in_global_hom 

        # Extract the 3D point in the camera frame (homogeneous coordinates -> 3D)
        t_feat_in_c = t_feat_in_c_hom[:3, 0].reshape(3, 1)  # (3,1) 3D point in the camera frame

        # Get the left-or-right intrinsics
        K = self.Kr if right else self.Kl

        # Project the point onto the image plane using the intrinsic matrix
        projection = K @ t_feat_in_c  # (3,1), projected onto image plan

        # Normalize by the depth (z-component) to get homogeneous pixel coordinates
        alpha = 1.0 / projection[2, 0]  # Inverse depth (z-coordinate)
        p = alpha * projection  # (3,1) homogeneous pixel coordinate (x, y, 1)
        
        return p, alpha
