#ifndef MANAGER_H
#define MANAGER_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

// Cuda libraries are required to allocate memory in device
#include <cuda.h>
#include <cuda_runtime.h>

// We need to store the observations as arrays in Device !!!
struct Observations{
    Observations(size_t max_observation_size);

    // Create streams for asynchronous memory copies
    cudaStream_t _stream_src, _stream_tgt, _stream_lmk;
    
    int *_d_source_frame_indexes;   // Index of the frame where the patch is originally extracted
    int *_d_target_frame_indexes;   // Index of the frame where the current observation has been extracted 
    int *_d_landmark_ids;           // Index of the global observation
    size_t _size{0};                // Amount of observations till now

    void addObservation(int *src_idx, int *tgt_idx, int *lmk_idx, size_t new_measurement_size);
};



struct KeyFrame{
public:
    float *_coord_x;         // Pixel x coordinates in anchar frame
    float *_coord_y;         // Pixel y coordinates in anchar frame
    float *_inverse_depths;  // Estimated inverse depth of each frame
    
    // For stereo, we also store the image coordinates in the second_camera. 
    float *_coord_x_s;         // Pixel x coordinates in stereo frame
    float *_coord_y_s;         // Pixel y coordinates in stereo frame
};

struct Manager{
public:

    Manager() : _observations(_max_observations_size) {};
    int _landmark_counter{0};

    // Map between the landmark index and landmark
    std::unordered_map<int, KeyFrame> _keyframes;


    // First initialize the _max_observations_size, then declare the _observations
    size_t _max_observations_size{10000};
    Observations _observations;

    // When new frame arrives, we can add new observations to our pose graph
    void addObservation(int *src_idx, int *tgt_idx, int *lmk_idx, int new_measurement_size);
    void printObservations();

    
};

#endif