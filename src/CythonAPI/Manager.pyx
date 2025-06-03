# distutils: language = c++

from libc.stdlib cimport realloc
from cython cimport int
from libc.string cimport memcpy

cimport Manager  

from Manager cimport Manager  # ✅ cimport the class directly

cdef class PyManager:
    cdef Manager* thisptr

    def __cinit__(self):
        self.thisptr = new Manager()

    def __dealloc__(self):
        del self.thisptr

    def addObservation(self, int[:] source_frame_indexes, int[:] target_frame_indexes, int[:] landmark_ids):
        
        # Get the number of new measurements
        cdef int new_measurement_size = source_frame_indexes.shape[0]

        # Convert python objects to C objects
        cdef int[:] src = source_frame_indexes
        cdef int[:] tgt = target_frame_indexes
        cdef int[:] lmk = landmark_ids

        
        # Reallocate memory for our new observations
        self.thisptr.addObservation(&src[0], &tgt[0], &lmk[0], new_measurement_size)

    def printObservations(self):
        self.thisptr.printObservations()