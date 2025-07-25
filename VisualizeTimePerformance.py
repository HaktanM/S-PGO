import numpy as np
import matplotlib.pyplot as plt

gpu_times = np.loadtxt("TimingTests.txt")
cpu_times = np.loadtxt("TimingTestsPython.txt")

gpu_number_of_frames  = gpu_times[:,0]
gpu_number_of_factors = gpu_times[:,2]
gpu_elapsed_times_all = gpu_times[:,3:]
gpu_elapsed_times = np.zeros((gpu_elapsed_times_all.shape[0]))
for _ in range(gpu_elapsed_times_all.shape[0]):
    gpu_elapsed_times[_] = np.array(gpu_elapsed_times_all[_,:]).mean()
gpu_elapsed_times = gpu_elapsed_times / 10 * (1e-6)

cpu_number_of_frames  = cpu_times[:,0]
cpu_number_of_factors = cpu_times[:,2]
cpu_elapsed_times_all = cpu_times[:,3:]
cpu_elapsed_times = np.zeros((cpu_elapsed_times_all.shape[0]))
for _ in range(cpu_elapsed_times_all.shape[0]):
    cpu_elapsed_times[_] = np.array(cpu_elapsed_times_all[_,:]).mean()
cpu_elapsed_times = cpu_elapsed_times * (1e-6)


# Create a figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# ---- Subplot 1: Number of Factors vs Elapsed Time ----
ax1.scatter(
    gpu_number_of_factors, gpu_elapsed_times,
    s=50, alpha=0.5, c="gray", label="GPU"
)
# ax1.scatter(
#     cpu_number_of_factors, cpu_elapsed_times,
#     s=50, alpha=0.5, c="blue", label="CPU"
# )
ax1.set_xlabel("Number of Factors")
ax1.set_ylabel("Elapsed Time")
ax1.set_title("Factors vs Elapsed Time")
ax1.legend()
ax1.grid(True)

# ---- Subplot 2: Number of Frames vs Elapsed Time ----
ax2.scatter(
    gpu_number_of_frames, gpu_elapsed_times,
    s=50, alpha=0.5, c="gray", label="GPU"
)
# ax2.scatter(
#     cpu_number_of_frames, cpu_elapsed_times,
#     s=50, alpha=0.5, c="blue", label="CPU"
# )
ax2.set_xlabel("Number of Frames")
ax2.set_title("Frames vs Elapsed Time")
ax2.legend()
ax2.grid(True)

# Layout and show
plt.tight_layout()
plt.show()