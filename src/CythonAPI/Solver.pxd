# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string

cdef extern from "Solver.hpp":
    cdef cppclass Solver:
        Solver(int N, int M)
        void loadCalibration(float *intrinsics, float *T_r_to_l)
        void writeObservations(int anchor_frame_ID, int target_frame_ID, int global_feat_ID, float *left_obs, float *right_obs)
        void step(int iterations)
        
        void getIncrementalPose(int keyFrameID, float *T_curr_to_next)
        void writeIncrementalPose(int keyFrameID, float *T_curr_to_next)
        void getObservation(int frame_ID, int global_feat_ID, float *left_obs, float *right_obs)
        void getCalibration(float *intrinsics, float *T_r_to_l)
        void loadInverseDepths(float *inverse_depths)
        void getInverseDepths(float *inverse_depths)

        int _number_of_keyframes
        int _number_of_observations_per_frame
        