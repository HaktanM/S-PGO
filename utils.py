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
        self.create_samples(n=self.n)

    def create_samples(self, n=10):
        for _ in range(n):
            # Sample a random position on a sphere of radius r
            r     = np.random.uniform(1.75 + _*0.25, 2.0 + _*0.25)  # radius of the camera orbit
            theta = np.random.uniform(0, 2 * np.pi)
            phi   = np.random.uniform(0, np.pi)

            # x = r * np.random.uniform(0.9, 1.1) # np.sin(phi) * np.cos(theta)
            # y = r * np.random.uniform(0.9, 1.1) # np.sin(phi) * np.sin(theta)
            # z = r * np.random.uniform(0.9, 1.1) # np.cos(phi)

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

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

