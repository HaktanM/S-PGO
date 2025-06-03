#include "Manager.hpp"


Observations::Observations(size_t max_observation_size){
    // Create cuda streams for asynchronous data processing
    cudaStreamCreate(&_stream_src);
    cudaStreamCreate(&_stream_tgt);
    cudaStreamCreate(&_stream_lmk);
    
    // Allocate memory for the observation arrays
    cudaError_t err;

    err = cudaMalloc(&_d_source_frame_indexes, max_observation_size * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed for _d_source_frame_indexes: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&_d_target_frame_indexes, max_observation_size * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed for _d_target_frame_indexes: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&_d_landmark_ids, max_observation_size * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed for _d_landmark_ids: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // No need to synchronize the streams immediately, they don't affect allocation checks
    std::cout << "Device Memory for observations has been allocated successfully." << std::endl;
}

void Observations::addObservation(int *src_idx, int *tgt_idx, int *lmk_idx, size_t new_measurement_size){

    // Copy the observations to the device asynchronously
    cudaMemcpyAsync(_d_source_frame_indexes + _size, src_idx, new_measurement_size * sizeof(int), cudaMemcpyHostToDevice, _stream_src);
    cudaMemcpyAsync(_d_target_frame_indexes + _size, tgt_idx, new_measurement_size * sizeof(int), cudaMemcpyHostToDevice, _stream_tgt);
    cudaMemcpyAsync(_d_landmark_ids + _size,         lmk_idx, new_measurement_size * sizeof(int), cudaMemcpyHostToDevice, _stream_lmk);

    // Wait for all streams to finish
    cudaStreamSynchronize(_stream_src);
    cudaStreamSynchronize(_stream_tgt);
    cudaStreamSynchronize(_stream_lmk);

    // Check for any error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error when adding new observations : " << cudaGetErrorString(err) << std::endl;
    }

    // Update the size
    _size += new_measurement_size;
}


void Manager::addObservation(int *src_idx, int *tgt_idx, int *lmk_idx, int new_measurement_size){
    _observations.addObservation(src_idx, tgt_idx, lmk_idx, static_cast<size_t>(new_measurement_size));
}

void Manager::printObservations() {

    // First, we need to carry the observations to CPU to print them.
    size_t size = static_cast<size_t>(_observations._size);

    // Allocate host memory using malloc
    int* h_source_frame_indexes;
    int* h_target_frame_indexes;
    int* h_landmark_ids;

    // Use cudaMallocHost for asynchronous memory copies
    cudaMallocHost(&h_source_frame_indexes, size * sizeof(int));
    cudaMallocHost(&h_target_frame_indexes, size * sizeof(int));
    cudaMallocHost(&h_landmark_ids,         size * sizeof(int));

    // Check for allocation failures
    if (!h_source_frame_indexes || !h_target_frame_indexes || !h_landmark_ids) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        cudaFreeHost(h_source_frame_indexes);
        cudaFreeHost(h_target_frame_indexes);
        cudaFreeHost(h_landmark_ids);
        return;
    }

    // Copy from device to host
    cudaMemcpyAsync(h_source_frame_indexes, _observations._d_source_frame_indexes, size * sizeof(int), cudaMemcpyDeviceToHost, _observations._stream_src);
    cudaMemcpyAsync(h_target_frame_indexes, _observations._d_target_frame_indexes, size * sizeof(int), cudaMemcpyDeviceToHost, _observations._stream_tgt);
    cudaMemcpyAsync(h_landmark_ids,         _observations._d_landmark_ids,         size * sizeof(int), cudaMemcpyDeviceToHost, _observations._stream_lmk);

    // Wait for all streams to finish
    cudaStreamSynchronize(_observations._stream_src);
    cudaStreamSynchronize(_observations._stream_tgt);
    cudaStreamSynchronize(_observations._stream_lmk);

    // Check for any error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error when printing observations : " << cudaGetErrorString(err) << std::endl;
    }

    for (size_t i = 0; i < size; ++i) {
        std::cout << "Observation " << i << ": "
                  << "Source Frame = " << h_source_frame_indexes[i]
                  << ", Target Frame = " << h_target_frame_indexes[i]
                  << ", Landmark ID = " << h_landmark_ids[i]
                  << std::endl;
    }

    // Free host memory
    cudaFreeHost(h_source_frame_indexes);
    cudaFreeHost(h_target_frame_indexes);
    cudaFreeHost(h_landmark_ids);
}