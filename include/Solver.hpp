#ifndef SOLVER_H
#define SOLVER_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

// Store the data in a torch array
#include <torch/torch.h>

// Cuda libraries are required to allocate memory in device
#include <cuda.h>
#include <cuda_runtime.h>

#include "Solver_cuda.h"


struct Solver{
public:
    Solver(int N, int M): _number_of_keyframes(N), _number_of_observations_per_frame(M), _max_meas_size(N*(N+1)*M) {

        // Check for CUDA availability and select device
        torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        std::cout << "Selected device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Set tensor options with float dtype and chosen device
        _options_float = torch::TensorOptions().dtype(torch::kFloat).device(device);
        _options_int   = torch::TensorOptions().dtype(torch::kInt).device(device);

        
        // Frame ID, Global Feature ID, Left or Right Frame, Homogenous Pixel Coordinates
        // 0 stands for left
        // 1 stands for right
        _observations = torch::zeros({N+1, N*M, 2, 3}, _options_float);
        
        // Incremental Pose ID and 4x4 rigid transformations
        _incremental_poses = torch::zeros({N, 4, 4}, _options_float);

        // Fill each [i] slice with identity matrix
        for (int i = 0; i < N; ++i) {
            _incremental_poses[i] = torch::eye(4, _options_float);
        }

        // Frame ID, Local Feature ID
        _inverse_depths    = torch::zeros({N*M}, _options_float);

        
        // Anchor frame of each feature is stored here
        _anchor_frame_id   = torch::zeros({_max_meas_size}, _options_int);
        _target_frame_id   = torch::zeros({_max_meas_size}, _options_int);
        _feat_glob_id      = torch::zeros({_max_meas_size}, _options_int);

        

        // intrinsics and extrinsics will be stored in torch arrays
        _intrinsics = torch::zeros({4,2}, _options_float);
        _T_r_to_l = torch::zeros({4,4}, _options_float);
    };

    void step(int iterations);
    void loadCalibration(float *intrinsics, float *T_r_to_l);
    void writeObservations(int anchor_frame_ID, int target_frame_ID, int global_feat_ID, float *left_obs, float *right_obs);

    void getIncrementalPose(int keyFrameID, float *T_curr_to_next);
    void getObservation(int frame_ID, int global_feat_ID, float *left_obs, float *right_obs);
    void getCalibration(float *intrinsics, float *T_r_to_l);

    // The order of decleration below is important !!! 
    long int _max_meas_size;
    int _number_of_keyframes, _number_of_observations_per_frame;
    torch::Tensor _observations, _incremental_poses, _inverse_depths, _anchor_frame_id, _target_frame_id, _feat_glob_id;
    int _counter{0};

private:
    torch::Tensor _intrinsics, _T_r_to_l;
    torch::TensorOptions _options_float, _options_int;
};

#endif