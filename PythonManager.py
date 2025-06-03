from build import Manager
import numpy as np

if __name__ == "__main__":
    manager = Manager.PyManager()


    # Prepare new observation arrays
    src = np.array([0,0,1,1], dtype=np.int32)    # source frame index
    tgt = np.array([1,2,2,2], dtype=np.int32)    # target frame index
    lmk = np.array([0,1,2,3], dtype=np.int32)    # landmark index (same as just created)

    # Append the observation
    manager.addObservation(src, tgt, lmk)


    # Prepare new observation arrays
    src = np.array([0,0,1,1,2,2], dtype=np.int32)    # source frame index
    tgt = np.array([3,3,3,3,3,3], dtype=np.int32)    # target frame index
    lmk = np.array([0,1,2,3,4,5], dtype=np.int32)    # landmark index (same as just created)

    # Append the observation
    manager.addObservation(src, tgt, lmk)

    # Optional: print the results
    manager.printObservations()