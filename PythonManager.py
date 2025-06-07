from build import Solver
import numpy as np

from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import map_value_to_index


class Manager():
    def __init__(self):

        # Number of keyframes
        self.n = 10

        # NUmber of landmarks per frame
        self.m = 40

        # Number of total landmarks 
        self.M = self.n * self.m

        # Initialize the simulator
        self.simulator = Simulator(n=self.n+1, m=self.M)

        # Initilize the CUDA solver
        self.solver = Solver.CudaSolver(self.n, self.m)

        # Load calibration to solver
        self.loadCalibration()

        # Load Observations to solver
        self.loadObservations()

    def loadObservations(self):
        for landmark_idx in range(self.M): 
            anchor_idx = map_value_to_index(v=landmark_idx, x=self.M, n=self.n)
            for projection_idx in range(anchor_idx, self.n+1):
                if self.simulator.validty[projection_idx][landmark_idx][False] and self.simulator.validty[projection_idx][landmark_idx][True]:
                    left_obs_py  = self.simulator.observations[projection_idx][landmark_idx][False].reshape(3).astype(np.float32)
                    right_obs_py = self.simulator.observations[projection_idx][landmark_idx][True].reshape(3).astype(np.float32)
                    self.solver.writeObservations(anchor_idx, projection_idx, landmark_idx, left_obs_py, right_obs_py)
                else:
                    print("Non Valid Observation has been detected")

    def loadCalibration(self):
        intrinsics = np.array([
            self.simulator.cam.Kl[0,0], self.simulator.cam.Kl[1,1], self.simulator.cam.Kl[0,2], self.simulator.cam.Kl[1,2],
            self.simulator.cam.Kr[0,0], self.simulator.cam.Kr[1,1], self.simulator.cam.Kr[0,2], self.simulator.cam.Kr[1,2]
        ]).reshape(8).astype(np.float32)

        T_r_to_l = self.simulator.cam.T_r_l.reshape(16).astype(np.float32)

        self.solver.loadCalibration(intrinsics, T_r_to_l)
if __name__ == "__main__":
    manager = Manager()
    left_obs, right_obs = manager.solver.getObservation(0,0)
    print(left_obs)
    print(right_obs)

    intrinsics, T_r_to_l = manager.solver.getCalibration()
    print(intrinsics)
    print(T_r_to_l)

    manager.solver.step(1)