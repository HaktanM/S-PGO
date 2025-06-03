# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string

cdef extern from "Manager.hpp":
    cdef cppclass Manager:
        Manager()
        void addObservation(int *src_idx, int *tgt_idx, int *lmk_idx, int new_measurement_size)
        void printObservations();
        