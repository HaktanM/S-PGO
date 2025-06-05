# distutils: language = c++

from libc.stdlib cimport realloc
from cython cimport int
from libc.string cimport memcpy



import numpy as np
cimport numpy as np

cimport Solver  

from Solver cimport Solver  # ✅ cimport the class directly

cdef class CudaSolver:
    cdef Solver* thisptr

    def __cinit__(self, int N, int M):
        self.thisptr = new Solver(N, M)

    def __dealloc__(self):
        del self.thisptr

    def writeObservations(self, int anchor_frame_ID, int target_frame_ID, int global_feat_ID, float[:] left_obs_py, float[:] right_obs_py):
        # Convert python objects to C objects
        cdef float[:] left_obs  = left_obs_py
        cdef float[:] right_obs = right_obs_py
        self.thisptr.writeObservations(anchor_frame_ID, target_frame_ID, global_feat_ID, &left_obs[0], &right_obs[0])


    def getIncrementalPose(self, int keyFrameID):
        # Create NumPy array to hold 16 floats (4x4)
        T_curr_to_next = np.empty(16, dtype=np.float32)

        # Get memoryview
        cdef float[::1] T_view = T_curr_to_next

        # Pass pointer to C++
        self.thisptr.getIncrementalPose(keyFrameID, &T_view[0])

        # Reshape into (4, 4)
        return T_curr_to_next.reshape((4, 4))

    def getObservation(self, int keyFrameID,  int global_feat_ID):
        # Create NumPy array to hold 16 floats (4x4)
        left_obs_py  = np.empty(3, dtype=np.float32)
        right_obs_py = np.empty(3, dtype=np.float32)

        # Get memoryview
        cdef float[::1] left_obs_view  = left_obs_py
        cdef float[::1] right_obs_view = right_obs_py

        # Pass pointer to C++
        self.thisptr.getObservation(keyFrameID, global_feat_ID, &left_obs_view[0], &right_obs_view[0])

        # Reshape into (4, 4)
        return ( left_obs_py.reshape((3, 1)), right_obs_py.reshape((3, 1)) )