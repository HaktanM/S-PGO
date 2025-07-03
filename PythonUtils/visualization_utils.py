import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import cv2
import numpy as np

def visualize_jacobian_and_residual_to_cv(J, residual):
    """
    Visualize Jacobian magnitude and residual vector on the same figure and return as an OpenCV BGR image.

    Parameters:
        J (np.ndarray): Jacobian matrix.
        residual (np.ndarray): Residual vector, shape (rows,) or (rows, 1).

    Returns:
        img_bgr (np.ndarray): Combined visualization as OpenCV BGR image.
    """
    magnitude = np.log1p(np.abs(J))
    residual = residual.flatten()

    # Create figure and canvas
    fig = plt.figure(figsize=(10, 8), dpi=100)
    canvas = FigureCanvas(fig)

    # Heatmap subplot
    ax1 = fig.add_subplot(2, 1, 1)
    cax = ax1.imshow(magnitude, cmap='viridis', aspect='auto')
    fig.colorbar(cax, ax=ax1, label='Jacobian Magnitude')
    ax1.set_title('Jacobian Magnitude Heatmap')
    ax1.set_xlabel('Columns (Parameters)')
    ax1.set_ylabel('Rows (Observations)')

    # Residual subplot
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(residual, marker='o', linestyle='-', color='crimson')
    ax2.set_title('Residual Vector')
    ax2.set_xlabel('Observation Index')
    ax2.set_ylabel('Residual Magnitude')
    ax2.grid(True)

    fig.tight_layout()

    # Convert to image
    canvas.draw()
    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imshow("Window", img_bgr)
    cv2.waitKey(1)
    plt.close(fig)
    return img_bgr


def visualize_jacobian(J):
    """
    Visualize the magnitude of a Jacobian matrix using a heatmap.
    
    Parameters:
        J (np.ndarray): The Jacobian matrix.
    """
    # Compute magnitude
    magnitude = np.log1p(np.abs(J))

    # Create figure and use Agg canvas
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    canvas = FigureCanvas(fig)  
    cax = ax.imshow(magnitude, cmap='viridis', aspect='auto')
    fig.colorbar(cax, label='Magnitude')
    ax.set_title('Jacobian Magnitude Heatmap')
    ax.set_xlabel('Columns (Parameters)')
    ax.set_ylabel('Rows (Observations)')
    fig.tight_layout()

    # Draw the canvas and convert to image
    canvas.draw()
    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))  # RGBA


    plt.close(fig)  # Close the figure to free memory

    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Window", img_bgr)
    cv2.waitKey(1)


def visualize_hessian_and_g(H, g, window_name="Hessian"):
    """
    Visualize Jacobian magnitude and residual vector on the same figure and return as an OpenCV BGR image.

    Parameters:
        J (np.ndarray): Jacobian matrix.
        residual (np.ndarray): Residual vector, shape (rows,) or (rows, 1).

    Returns:
        img_bgr (np.ndarray): Combined visualization as OpenCV BGR image.
    """
    magnitude = np.log1p(np.abs(H))
    g = g.flatten()

    # Create figure and canvas
    fig = plt.figure(figsize=(10, 8), dpi=100)
    canvas = FigureCanvas(fig)

    # Heatmap subplot
    ax1 = fig.add_subplot(2, 1, 1)
    cax = ax1.imshow(magnitude, cmap='viridis', aspect='auto')
    fig.colorbar(cax, ax=ax1, label='Jacobian Magnitude')
    ax1.set_title('Hessian Magnitude Heatmap')
    ax1.set_xlabel('Columns (Parameters)')
    ax1.set_ylabel('Rows (Observations)')

    # Residual subplot
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(g, marker='o', linestyle='-', color='crimson')
    ax2.set_title('g Vector')
    ax2.set_xlabel('Observation Index')
    ax2.set_ylabel('g Magnitude')
    ax2.grid(True)

    fig.tight_layout()

    # Convert to image
    canvas.draw()
    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imshow(window_name, img_bgr)
    cv2.waitKey()

    plt.close(fig)

    return img_bgr


def visualize_depth_estimation(actual_depths, estimated_depths):
    x = np.arange(len(actual_depths))

    # Create figure and Agg canvas
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    canvas = FigureCanvas(fig)

    # Scatter plot
    ax.scatter(x, actual_depths, label='Actual Depth', color='blue', marker='o')
    ax.scatter(x, estimated_depths, label='Estimated Depth', color='red', marker='x')
    ax.set_title("Scatter Plot: Actual vs Estimated Depth")
    ax.set_xlabel("Index")
    ax.set_ylabel("Depth Value")
    ax.set_ylim([-1,25])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # Draw and convert to RGBA image
    canvas.draw()
    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
    plt.close(fig)  # Free memory

    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Show with OpenCV
    cv2.imshow("Depth Estimation Scatter Plot", img_bgr)
    cv2.waitKey(1)  # Adjust delay or use 0 to wait for key press
