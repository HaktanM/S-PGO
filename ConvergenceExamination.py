import numpy as np
import matplotlib.pyplot as plt



# Load the test results
errors = np.loadtxt("GI_ConvergenceTest.txt")
timing = np.loadtxt("GI_ConvergenceTest_timing.txt")



fig, ax = plt.subplots()


for run_id in range(errors.shape[0]):
    ax.plot(errors[run_id,:], linewidth=2, alpha=0.1, color="gray")

plt.show()