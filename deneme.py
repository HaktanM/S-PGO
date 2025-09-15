import matplotlib.pyplot as plt
import numpy as np

# Example: replace with your Jacobian
# J = manager.optimizer.getJacobians(...)[0]
J = np.random.randn(200, 200)  # demo: large Jacobian

plt.figure(figsize=(10, 8))

# Smooth heatmap without borders
im = plt.imshow(J, cmap="RdBu_r", aspect="auto", interpolation="nearest")

# Add colorbar with a clean style
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Value", fontsize=14)

# Remove ticks for a clean look
plt.xticks([])
plt.yticks([])

plt.title("Jacobian Heatmap", fontsize=18, pad=15)
plt.tight_layout()
plt.show()
