import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load data
errors = np.loadtxt("GI_ConvergenceTest.txt")
timing = np.loadtxt("GI_ConvergenceTest_timing.txt")

# Compute mean trajectory
mean_error = np.mean(errors, axis=0)

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Plot all runs (light gray) with reduced alpha for a more subtle look
for run_id in range(errors.shape[0]):
    ax.plot(errors[run_id, :], linewidth=1.5, alpha=0.1, color="red")

# Plot the mean trajectory with a bold and distinct style
ax.plot(mean_error, linewidth=3, color="tab:blue", linestyle='--', label="Mean Trajectory")

# LaTeX-style x-axis labels
def latex_formatter(x, pos):
    return f'${int(x)}$'  # Converts ticks to integers with LaTeX formatting

# Apply formatter to x-axis
ax.xaxis.set_major_formatter(FuncFormatter(latex_formatter))

# Styling for axes and labels
ax.set_xlabel("Iteration", fontsize=14, family='serif', labelpad=15)
ax.set_ylabel("Error", fontsize=14, family='serif', labelpad=15)
ax.set_title("Convergence Analysis", fontsize=16, family='serif', weight='bold', pad=20)

# Increase grid visibility
ax.grid(True, linestyle='-', color='gray', alpha=0.3, linewidth=0.8)

# Minor gridlines for better visual precision
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', color='gray', alpha=0.2, linewidth=0.6)

# Custom ticks for better precision on x-axis
ax.tick_params(axis='both', labelsize=14)

# Legend with better positioning
ax.legend(fontsize=12, loc="upper right", fancybox=True, framealpha=0.7)

# Add a border to the plot for a clean look
fig.patch.set_edgecolor('black')
fig.patch.set_linewidth(1)

# Layout adjustments for a clean fit
plt.tight_layout()
plt.show()
