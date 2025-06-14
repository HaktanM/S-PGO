import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import cv2

def draw_cube(center=(0.0, 0.0, 0.0)):
    glBegin(GL_QUADS)

    cx, cy, cz = center

    # Half the side length (size 2 → half = 1)
    hs = 1.0

    # Define cube vertices relative to center
    vertices = [
        [cx - hs, cy - hs, cz - hs],
        [cx + hs, cy - hs, cz - hs],
        [cx + hs, cy + hs, cz - hs],
        [cx - hs, cy + hs, cz - hs],

        [cx - hs, cy - hs, cz + hs],
        [cx + hs, cy - hs, cz + hs],
        [cx + hs, cy + hs, cz + hs],
        [cx - hs, cy + hs, cz + hs],

        [cx - hs, cy - hs, cz - hs],
        [cx - hs, cy - hs, cz + hs],
        [cx - hs, cy + hs, cz + hs],
        [cx - hs, cy + hs, cz - hs],

        [cx + hs, cy - hs, cz - hs],
        [cx + hs, cy - hs, cz + hs],
        [cx + hs, cy + hs, cz + hs],
        [cx + hs, cy + hs, cz - hs],

        [cx - hs, cy - hs, cz - hs],
        [cx - hs, cy - hs, cz + hs],
        [cx + hs, cy - hs, cz + hs],
        [cx + hs, cy - hs, cz - hs],

        [cx - hs, cy + hs, cz - hs],
        [cx - hs, cy + hs, cz + hs],
        [cx + hs, cy + hs, cz + hs],
        [cx + hs, cy + hs, cz - hs]
    ]

    colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0]   # Cyan
    ]

    for i in range(6):
        glColor3f(*colors[i])
        glVertex3f(*vertices[i * 4 + 0])
        glVertex3f(*vertices[i * 4 + 1])
        glVertex3f(*vertices[i * 4 + 2])
        glVertex3f(*vertices[i * 4 + 3])

    glEnd()




def draw_camera_frame(transform_matrix, fov_y_deg=45.0, aspect_ratio=1.0, near=0.1, far=0.3):
    """
    Draws a clean camera frustum with a filled near plane (white square).

    :param transform_matrix: 4x4 numpy array (camera-to-world transform)
    :param fov_y_deg: vertical field of view in degrees
    :param aspect_ratio: width / height of the view
    :param near: near plane distance
    :param far: far plane distance
    """

    R = transform_matrix[:3, :3]
    T = transform_matrix[:3, 3]

    # 🔵 Draw local axes
    glLineWidth(2.0)
    glBegin(GL_LINES)
    axis_len = 0.1
    axes = [
        (np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]),  # X (red)
        (np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]),  # Y (green)
        (np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])   # Z (blue)
    ]
    for direction, color in axes:
        glColor3f(*color)
        end = T + R @ direction * axis_len
        glVertex3f(*T)
        glVertex3f(*end)
    glEnd()


def draw_origin():
    # 🔵 Draw global axes
    glLineWidth(2.0)
    glBegin(GL_LINES)
    axis_len = 0.1
    axes = [
        (np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]),  # X (red)
        (np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]),  # Y (green)
        (np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])   # Z (blue)
    ]

    origin = np.array([0.0,0.0,0.0])
    for direction, color in axes:
        glColor3f(*color)
        end = direction * axis_len
        glVertex3f(*origin)
        glVertex3f(*end)
    glEnd()

def draw_camera_frustum(transform_matrix, fov_y_deg=45.0, aspect_ratio=1.0, near=0.1, far=0.2, color=[0.6, 0.6, 0.8]):
    """
    Draws a clean camera frustum with a filled near plane (white square).

    :param transform_matrix: 4x4 numpy array (camera-to-world transform)
    :param fov_y_deg: vertical field of view in degrees
    :param aspect_ratio: width / height of the view
    :param near: near plane distance
    :param far: far plane distance
    """

    R = transform_matrix[:3, :3]
    T = transform_matrix[:3, 3]

    def get_frustum_plane(fov_y, aspect, dist):
        h = np.tan(np.radians(fov_y) / 2.0) * dist
        w = h * aspect
        return [
            np.array([-w, -h, dist]),  # bottom left
            np.array([ w, -h, dist]),  # bottom right
            np.array([ w,  h, dist]),  # top right
            np.array([-w,  h, dist])   # top left
        ]

    near_corners = get_frustum_plane(fov_y_deg, aspect_ratio, near)
    far_corners = get_frustum_plane(fov_y_deg, aspect_ratio, far)

    near_world = [R @ p + T for p in near_corners]
    far_world = [R @ p + T for p in far_corners]

    # 🟦 Draw filled near plane
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_QUADS)
    for pt in near_world:
        glVertex3f(*pt)
    glEnd()


    # 🟨 Draw frustum lines (edges + far plane)
    glLineWidth(1.5)
    glBegin(GL_LINES)
    glColor3f(color[0], color[1], color[2])

    # Far plane box
    for i in range(4):
        glVertex3f(*far_world[i])
        glVertex3f(*far_world[(i + 1) % 4])

    # Lines connecting near and far corners
    for i in range(4):
        glVertex3f(*near_world[i])
        glVertex3f(*far_world[i])
    glEnd()

    # 🔵 Draw local axes
    glLineWidth(2.0)
    glBegin(GL_LINES)
    axis_len = 0.1
    axes = [
        (np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]),  # X (red)
        (np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]),  # Y (green)
        (np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])   # Z (blue)
    ]
    for direction, color in axes:
        glColor3f(*color)
        end = T + R @ direction * axis_len
        glVertex3f(*T)
        glVertex3f(*end)
    glEnd()


def add_landmark(landmark):
    glPointSize(5.0)  # Size of the landmark points
    glColor3f(1.0, 0.0, 0.0)  # Red color for landmarks

    glBegin(GL_POINTS)
    glVertex3f(landmark[0], landmark[1], landmark[2])
    glEnd()

class Renderer:
    def __init__(self):

        # Camera position (modify these for your scene)
        self.camera_pos = np.array([0.0, 0.0, 5.0])  # Camera position in world space
        self.target_pos = np.array([0.0, 0.0, 0.0])  # Camera position in world space
        self.last_x, self.last_y = 0, 0  # Last mouse position for tracking movement
        self.is_left_button_pressed = False
        self.is_right_button_pressed = False

        # Camera frames to be visualized
        self.cam_frames = []

        # Camera frames to be visualized
        self.estimated_cam_frames = []

        # Landmarks to be visualized
        self.landmarks  = []

        # This is used when saving the scene
        self.frame_id = 0

    def zoom_in(self):
        looking_direction = self.camera_pos - self.target_pos
        looking_direction = looking_direction * 0.9
        self.camera_pos   = self.target_pos + looking_direction
        glutPostRedisplay()  # Render the scene again

    def zoom_out(self):
        looking_direction = self.camera_pos - self.target_pos
        looking_direction = looking_direction * 1.1
        self.camera_pos   = self.target_pos + looking_direction
        glutPostRedisplay()  # Render the scene again

    # Function to handle mouse button presses (for left/right mouse buttons)
    def mouse_button_callback(self, button, state, x, y):

         # Mouse wheel scroll up
        if button == 3:
            self.zoom_in()
            return

        # Mouse wheel scroll down
        if button == 4:
            self.zoom_out()
            return

        # Left button pressed/released
        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self.is_left_button_pressed = True
            else:
                self.is_left_button_pressed = False

        # Right button pressed/released
        if button == GLUT_RIGHT_BUTTON:
            if state == GLUT_DOWN:
                self.is_right_button_pressed = True
            else:
                self.is_right_button_pressed = False

        # Update the last mouse position
        self.last_x, self.last_y = x, y

    # Function to handle mouse motion (for camera position/rotation)
    def mouse_motion_callback(self, x, y):
        # Calculate the mouse movement delta
        delta_x = x - self.last_x
        delta_y = y - self.last_y

        if self.is_left_button_pressed:
            # Left button pressed, change the camera position
            sensitivity = 0.01
            
            # Compute current vector from target to camera
            direction = self.camera_pos - self.target_pos
            distance = np.linalg.norm(direction)

            # Convert to spherical coordinates
            theta = np.arctan2(direction[2], direction[0])  # horizontal angle
            phi = np.arccos(direction[1] / distance)        # vertical angle

            # Apply mouse delta to angles
            theta -= delta_x * sensitivity
            phi -= delta_y * sensitivity
            phi = np.clip(phi, 0.01, np.pi - 0.01)  # Avoid flipping

            # Convert back to cartesian coordinates
            new_direction = np.array([
                distance * np.sin(phi) * np.cos(theta),
                distance * np.cos(phi),
                distance * np.sin(phi) * np.sin(theta)
            ])

            # Update the camera position
            self.camera_pos = self.target_pos + new_direction


        if self.is_right_button_pressed:
            # Right button pressed, change the target position
            sensitivity = 0.01
            self.target_pos [0] += delta_x * sensitivity
            self.target_pos [1] += delta_y * sensitivity

            self.camera_pos [0] += delta_x * sensitivity
            self.camera_pos [1] += delta_y * sensitivity

        # Update the last mouse position
        self.last_x, self.last_y = x, y

        # Redraw the scene
        glutPostRedisplay()

    def key_pressed(self, key, x, y):
        # Find the looking direction
        looking_direction = self.camera_pos - self.target_pos

        forward = looking_direction / np.linalg.norm(looking_direction)  # Normalize

        # Step 2: Assume world up vector (e.g., y-axis is up)
        world_up = np.array([0.0, 1.0, 0.0])

        # Step 3: Gram-Schmidt - get the right vector
        right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)

        # Step 4: Use right and forward to get the orthogonal up vector
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        sensitivity = 0.2
        if key.decode() == "w":
            motion = - forward * sensitivity
            self.camera_pos = self.camera_pos + motion
            self.target_pos = self.target_pos + motion
            glutPostRedisplay()  # Render the scene again

        elif key.decode() == "s":
            motion = forward * sensitivity
            self.camera_pos = self.camera_pos + motion
            self.target_pos = self.target_pos + motion
            glutPostRedisplay()  # Render the scene again

        elif key.decode() == "d":
            motion = right * sensitivity
            self.camera_pos = self.camera_pos + motion
            self.target_pos = self.target_pos + motion
            glutPostRedisplay()  # Render the scene again

        elif key.decode() == "a":
            motion = - right * sensitivity
            self.camera_pos = self.camera_pos + motion
            self.target_pos = self.target_pos + motion
            glutPostRedisplay()  # Render the scene again
            
        
        

    # Set up the projection matrix and camera position
    def setup_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Adjust the field of view (FOV) dynamically (if needed)
        gluPerspective(45.0, 1.0, 0.1, 100.0)  # Field of view, aspect ratio, near and far planes

        glMatrixMode(GL_MODELVIEW)

    # Main drawing function
    def draw_scene(self):
        glClearColor(1.0, 0.95, 0.95, 0.0) # Set the background color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set up the projection matrix
        self.setup_projection()

        # Set up the camera (this sets the view matrix)
        gluLookAt(  self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
                    self.target_pos[0], self.target_pos[1], self.target_pos[2],
                    0.0, 1.0, 0.0)
    
        # Draw the origin
        draw_origin()

        # Call the function to draw the camera frame
        for item in self.cam_frames:
            draw_camera_frustum(item, color=[0.5, 0.8, 0.5])

        # Call the function to draw the estimated camera frame
        for item in self.estimated_cam_frames:
            draw_camera_frustum(item, color=[0.8, 0.5, 0.5])
 
        # Also insert our landmarks
        for item in self.landmarks:
            add_landmark(item)

        # self.save_screenshot()

        glutSwapBuffers()

    # Function to handle window resizing
    def reshape(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    # Whenever you wish to render again, you call this function
    def re_render(self):
        glutPostRedisplay()


    def timer_callback(self, value):
        # Re-register the timer for the next callback
        glutTimerFunc(self.timer_interval, self.timer_callback, 0)
        glutPostRedisplay() # Render the scene again

    def start_rendering(self):
        # Main function to initialize and run the OpenGL window
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(800, 600)
        glutCreateWindow("Scene Renderer")

        glEnable(GL_DEPTH_TEST)                             # Enable depth testing for 3D rendering  # Initialize OpenGL settings
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glutDisplayFunc(self.draw_scene)  # Register display function

        glutMouseFunc(self.mouse_button_callback)   # Register mouse button callback
        glutMotionFunc(self.mouse_motion_callback)  # Register mouse motion callback
        glutKeyboardFunc(self.key_pressed)          # Register to keybord

        # Start the timer loop
        self.timer_interval = 500  # in milliseconds
        glutTimerFunc(self.timer_interval, self.timer_callback, 0)

        glutMainLoop()  # Start the main loop

    def save_screenshot(self):
        width = glutGet(GLUT_WINDOW_WIDTH)
        height = glutGet(GLUT_WINDOW_HEIGHT)

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)

        # Flip vertically (OpenGL origin is bottom-left; OpenCV expects top-left)
        image = np.flipud(image)

        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        os.makedirs("render_output", exist_ok=True)
        filename = f"render_output/{self.frame_id:04d}.png"
        cv2.imwrite(filename, image)

        self.frame_id += 1

if __name__ == "__main__":
    renderer = Renderer()
    renderer.start_rendering()
