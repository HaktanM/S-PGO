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

    def step(self, int iterations):
        self.thisptr.step(iterations)

    def loadCalibration(self, float[:] intrinsics_py, float[:] T_r_to_l_py):
        # Convert python objects to C objects
        cdef float[:] intrinsics = intrinsics_py
        cdef float[:] T_r_to_l = T_r_to_l_py
        self.thisptr.loadCalibration(&intrinsics[0], &T_r_to_l[0])

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

    def writeIncrementalPose(self, int keyFrameID, float[:] T_curr_to_next_py):
        # Convert python objects to C objects
        cdef float[:] T_curr_to_next  = T_curr_to_next_py
        self.thisptr.writeIncrementalPose(keyFrameID, &T_curr_to_next[0])

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


    def getCalibration(self):
        # Create NumPy array to hold 16 floats (4x4)
        intrinsics_py = np.empty(8, dtype=np.float32)
        T_r_to_l_py = np.empty(16, dtype=np.float32)

        # Get memoryview
        cdef float[::1] intrinsics_view = intrinsics_py
        cdef float[::1] T_r_to_l_view = T_r_to_l_py

        # Pass pointer to C++
        self.thisptr.getCalibration(&intrinsics_view[0], &T_r_to_l_view[0])

        # Reshape into (4, 4)
        return ( intrinsics_py.reshape((4, 2)), T_r_to_l_py.reshape((4, 4)) )

    def loadInverseDepths(self, float[:] inverse_depths_py):
        # Convert python objects to C objects
        cdef float[:] inverse_depths_view = inverse_depths_py
        self.thisptr.loadInverseDepths(&inverse_depths_view[0])

    def getInverseDepths(self):       
        inverse_depths  = np.empty( self.thisptr._number_of_keyframes * self.thisptr._number_of_observations_per_frame, dtype=np.float32)
        
        # Get memoryview
        cdef float[::1] inverse_depths_view  = inverse_depths

        # Pass pointer to C++
        self.thisptr.getInverseDepths(&inverse_depths_view[0])

        return inverse_depths

    def setStepSize(self, step_size):
        self.thisptr._step_size = step_size