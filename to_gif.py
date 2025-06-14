import cv2
import imageio.v2 as imageio
import glob
import os

def generate_gif_from_frames_opencv(frame_folder='render_output', output_gif='output.gif', fps=10):
    frame_paths = sorted(glob.glob(os.path.join(frame_folder, '*.png')))

    if not frame_paths:
        print("No frames found to create GIF.")
        return

    images = []
    for frame_path in frame_paths:
        img_bgr = cv2.imread(frame_path)               # Read using OpenCV (BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB for GIF
        images.append(img_rgb)

    imageio.mimsave(output_gif, images, duration=1/fps)
    print(f"GIF saved as '{output_gif}' with {len(images)} frames at {fps} FPS.")

generate_gif_from_frames_opencv('render_output', 'optimization_steps.gif', fps=5)