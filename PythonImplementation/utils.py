import pypose as pp
import numpy as np
import pypose as pp
import torch
from scipy.spatial.transform import Rotation as ori

class SamplePoses():
    """
    This class is responsible for creating n camera frames, all looks toward the global origin
    """
    def __init__(self, n=10):
        self.n = n
        
        self.poses = []

        # self.create_samples_deterministic()
        self.create_samples(n=self.n)

    def create_samples(self, n=10):
        theta_rw = 0.0
        phi_rw = 0.0
        for _ in range(n):
            # Sample a random position on a sphere of radius r
            r     = np.random.uniform(2.0 + _*0.5, 2.5 + _*0.5)  # radius of the camera orbit
            # theta = np.random.uniform(0, 2 * np.pi)
            # phi   = np.random.uniform(0, np.pi)

            # x = r * np.random.uniform(0.9, 1.1) # np.sin(phi) * np.cos(theta)
            # y = r * np.random.uniform(0.9, 1.1) # np.sin(phi) * np.sin(theta)
            # z = r * np.random.uniform(0.9, 1.1) # np.cos(phi)

            # x = r
            # y = r
            # z = r

            # x = r * np.sin(phi) * np.cos(theta)
            # y = r * np.sin(phi) * np.sin(theta)
            # z = r * np.cos(phi)

            theta_rw += np.abs ( np.random.uniform(0.0, 45.0) * np.pi / 180.0 )
            phi_rw   += np.abs ( np.random.uniform(0.0, 50.0) * np.pi / 180.0 )

            x = r * np.sin(phi_rw) * np.cos(theta_rw) * np.random.uniform(0.9, 1.1)
            y = r * np.sin(phi_rw) * np.sin(theta_rw) * np.random.uniform(0.9, 1.1)
            z = r * np.cos(phi_rw) * np.random.uniform(0.9, 1.1)

            position = torch.tensor([[x, y, z]], dtype=torch.float32)

            # Look at origin (0, 0, 0), compute forward vector
            forward = - position / torch.norm(position)
            up      =   torch.tensor([[np.random.uniform(0, 0.2), np.random.uniform(0, 0.2), 1.0]], dtype=torch.float32)
            up      =   up / torch.norm(up)

            # Right vector via cross product
            right = torch.cross(up, forward, dim=1)
            right = right / torch.norm(right)

            # Recompute up to make it orthogonal
            up = torch.cross(forward, right, dim=1)

            # Rotation matrix: [right; up; forward] as rows
            R = torch.cat([right, up, forward], dim=0).T  # 3x3
            quaternion = ori.from_matrix(R.cpu().numpy()).as_quat()

            # Convert quaternion to a tensor
            quaternion_tensor = torch.tensor(quaternion, dtype=torch.float32)
            
            # Add a bit of noise to the orientation (optional)
            noise_angle = 10.0 * np.pi / 180 
            noise_axis = torch.randn(3)
            noise_axis = noise_axis / torch.norm(noise_axis)
            angle_axis = noise_angle * noise_axis
            noise_rot = pp.so3(angle_axis.unsqueeze(0)).Exp()
            quaternion_tensor = noise_rot * quaternion_tensor

            # Concatenate quaternion and position to form a [7] vector
            pose_vector = torch.cat([position, quaternion_tensor], dim=1)

            pose = pp.SE3(pose_vector).matrix().cpu().numpy().reshape(4,4)  # (1,4,4)
            self.poses.append(pose)


    def create_samples_deterministic(self):
        """
        Generate deterministic camera poses evenly spaced around a sphere,
        all looking at the origin (0, 0, 0).
        """

        self.poses = []  # Clear previous poses

        r_theta_phi = [
            (2.5, 0.0, 0.0),                    # Top of sphere (forward aligned with up)
            (2.5, 0.0, np.pi / 2),              # Equator
            (3.5, 0.0, -np.pi / 2)         # General position
        ]

        for r, theta, phi in r_theta_phi:
            # Convert spherical to Cartesian coordinates
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            position = torch.tensor([[x, y, z]], dtype=torch.float32)

            # Forward vector (look at origin)
            forward = -position / torch.norm(position)

            # Default up vector
            world_up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

            # Check for alignment between forward and world_up
            dot = torch.abs(torch.sum(forward * world_up))
            if dot > 0.99:
                world_up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)  # Use Y-up if too aligned

            # Compute right and up vectors
            right = torch.cross(world_up, forward, dim=1)
            right = right / torch.norm(right + 1e-8)  # Avoid div by 0
            up = torch.cross(forward, right, dim=1)

            # Assemble rotation matrix (right, up, forward as columns)
            R = torch.cat([right, up, forward], dim=0).T  # 3x3

            # Assemble 4x4 pose matrix
            T = torch.eye(4)
            T[:3, :3] = R
            T[:3, 3] = position.squeeze()

            self.poses.append(T.numpy())


########### Landmark Sampling Utils ###########
def sample_point_on_sphere(r):
    theta = np.random.uniform(0, 2 * np.pi)
    u = np.random.uniform(-1, 1)  # u = cos(φ)
    phi = np.arccos(u)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return np.array([x, y, z])

def get_point_samples(m=10):
    points = []
    for _ in range(m):
        r = np.random.uniform(0, 2.5)
        point = sample_point_on_sphere(r=r)
        points.append(point)
    return points
###############################################

