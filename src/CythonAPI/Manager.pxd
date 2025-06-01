# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string

cdef extern from "Manager.hpp":
    cdef cppclass Manager:
        Manager()
        void createLandmark(int anchor_frame_idx, float inv_depth)