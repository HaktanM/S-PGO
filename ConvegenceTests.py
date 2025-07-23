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

    # How many iterations to go in the optimization
    number_of_steps  = 100

    # Matrix of errors
    errors_matrix = np.zeros((number_of_trials, number_of_steps))
    elapsed_times = np.zeros((number_of_trials, number_of_steps))

    diverged = 0
    # Testing Loop
    for run_id in range(number_of_trials):
        manager = Manager(n=10, m=96)
        manager.solver.setStepSize(0.5)

        # Optimization Loop
        for iter in tqdm(range(number_of_steps), desc=f"Run ID:{run_id}"):
            # # Set the random step size
            # step_size = np.random.rand(1) #  * 0.5
            # manager.solver.setStepSize(step_size)
            
            # Optimization Step
            t_start = time.monotonic_ns()
            manager.solver.step(1)
            t_stop = time.monotonic_ns()

            elapsed_time = (t_stop-t_start)
            elapsed_times[run_id, iter]

            # Get the error
            errors     = manager.compute_estimation_errors()
            error_mean = np.array(errors).sum()
            errors_matrix[run_id, iter] = error_mean

            # if iter == int(number_of_steps/2):
            #     manager.solver.setStepSize(1.0)

        if errors_matrix[run_id,-1]>0.5:
            diverged += 1
        print(f"Divergence Rate : {diverged} / {run_id+1}")

    # Once the test is competed, save the errors
    np.savetxt("GI_ConvergenceTest.txt", errors_matrix)
    np.savetxt("GI_ConvergenceTest_timing.txt", elapsed_times)




    
