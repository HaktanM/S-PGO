from build_gp import Solver
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
import threading

if __name__ == "__main__":
    
    manager = Manager(n=10, m=96)
    manager.solver.setStepSize(1.5)

    worker = threading.Thread(
        target=manager.optimization_loop,
        daemon=True            # ensures the thread won’t block process exit
    )
    worker.start()
    time.sleep(1)

    manager.visualizer.start_rendering()

    worker.join()

        

    
