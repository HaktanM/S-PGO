# distutils: language = c++

cimport Manager  

from Manager cimport Manager  # ✅ cimport the class directly

cdef class PyManager:
    cdef Manager* thisptr

    def __cinit__(self):
        self.thisptr = new Manager()

    def __dealloc__(self):
        del self.thisptr

    def createLandmark(self, int anchor_frame_idx, float inv_depth):
        self.thisptr.createLandmark(anchor_frame_idx, inv_depth)