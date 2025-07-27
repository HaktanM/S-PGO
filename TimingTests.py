from build import Solver
from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import Optimizer
from PythonUtils.SceneRenderer import Renderer
from PythonUtils.LieUtils import LieUtils
from PythonUtils.Optimizer import map_value_to_index
from PythonUtils.visualization_utils import visualize_hessian_and_g
from PythonUtils.visualization_utils import visualize_depth_estimation
import time
import threading
import numpy as np
from tqdm import tqdm  # Import tqdm

from CUDAmanager import Manager


if __name__ == "__main__":
    
    # How many run to test
    number_of_trials = 100

    number_of_poses = [3,  6,  9,  12]
    number_of_feats = [24, 48, 72, 96]

    elapsed_times = np.zeros((len(number_of_poses)*len(number_of_poses), number_of_trials+3))

    f = open('TimingTests.txt', 'w')
    f.close()
    
    row_idx = 0
    for n in number_of_poses:
        for m in number_of_feats:
            manager = Manager(n=n, m=m)
            f = open('TimingTests.txt', 'a')
            f.write(f"{n} {m} {manager.meas_size} ")
            for _ in range(number_of_trials):
                manager = Manager(n=n, m=m)
                manager.solver.setStepSize(0.1)
                t_start = time.monotonic_ns()
                manager.solver.step(10)
                t_stop = time.monotonic_ns()
                elapsed_time = (t_stop - t_start)

                elapsed_times[row_idx,0] = n
                elapsed_times[row_idx,1] = m
                elapsed_times[row_idx, 2+_] = elapsed_time

                
                if _ < number_of_trials - 1:
                    f.write(f"{int(elapsed_time)} ")
                else:
                    f.write(f"{int(elapsed_time)}")
            f.write("\n")
            f.close()
            row_idx += 1 