from build_gp import Solver
from PythonUtils.Simulator import Simulator
from PythonUtils.Optimizer import Optimizer
from PythonUtils.SceneRenderer import Renderer
from PythonUtils.LieUtils import LieUtils
from PythonUtils.Optimizer import map_value_to_index
from PythonUtils.visualization_utils import visualize_hessian_and_g
import time
import threading
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from CUDAmanager import Manager

if __name__ == "__main__":


    manager = Manager()

    # Get the optimization parameters from Python

    H_TT, g_TT, H_aa, g_aa, BB = manager.PyOptimizer.getHessians(observations=manager.simulator.observations)
    J_T, J_a, r = manager.PyOptimizer.getJacobiansAndResidual(observations=manager.simulator.observations)
    J = np.log1p(np.log1p(np.log1p(np.abs(J_T[:manager.meas_size,:]))))

    sns.set_theme(style="white")
    plt.figure(figsize=(8,6))

    # Create heatmap
    ax = sns.heatmap(
        J,
        cmap=sns.color_palette("Spectral", as_cmap=True),
        center=0,          # center at 0 for Jacobians
        cbar_kws={"shrink": 0.8, "label": "Value"},
        cbar=True,        # hide colorbar
        linewidths=0.0,
        square=False,
        annot=False,       # ensures no cell values
        xticklabels=False, # hide x-axis numbers
        yticklabels=False  # hide y-axis numbers
    )

    ax.set_title("Logarithmic Heatmap of Jacobian", fontsize=14, family='serif', weight='bold', pad=20)
    ax.set_xlabel("Axis of Estimated Variables", fontsize=14, family='serif')
    ax.set_ylabel("Axis of Observations", fontsize=14, family='serif')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig("Jacobian_GP.pdf")




    plt.figure(figsize=(8,6))
    J = np.log1p(np.abs(H_TT))
    sns.set_theme(style="white")
    plt.figure(figsize=(8,6))


    # Create heatmap
    ax = sns.heatmap(
        J,
        cmap=sns.color_palette("Spectral", as_cmap=True),
        center=0,          # center at 0 for Jacobians
        cbar_kws={"shrink": 0.8, "label": "Value"},
        cbar=True,        # hide colorbar
        linewidths=0.0,
        square=False,
        annot=False,       # ensures no cell values
        xticklabels=False, # hide x-axis numbers
        yticklabels=False  # hide y-axis numbers
    )

    ax.set_title("Logarithmic Heatmap of Hessian", fontsize=14, family='serif', weight='bold', pad=20)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig("Hessian_GP.pdf")