from PythonManager import Manager
import numpy as np
import time
import threading

if __name__ == "__main__":
    manager = Manager()

    # Create and start the background thread
    worker = threading.Thread(
        target=manager.pose_only_optimization_loop,
        daemon=True            # ensures the thread won’t block process exit
    )
    worker.start()

    manager.visualizer.start_rendering()

    worker.join()