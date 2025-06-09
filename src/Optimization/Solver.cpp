#include "Solver.hpp"



void Solver::step(int iterations){
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    _num_of_pose_params = _incremental_poses.size(0)*6;
    _num_of_landmarks   = _inverse_depths.size(0);

    int J_T_col_size     = _num_of_pose_params;
    int J_alpha_col_size = _num_of_landmarks;
    int row_size         = _counter * 4;

    // Allocate memory on device to store the Jacobians and residual
    float *d_J_T;
    float *d_J_alpha;
    float *d_r;
    float *d_H_TT;
    float *d_g_T;

    cudaError_t err;
    err = cudaMalloc((void**)&d_J_T, row_size * J_T_col_size * sizeof(float));
    // Check for errors
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc((void**)&d_J_alpha, row_size * J_alpha_col_size * sizeof(float));
    // Check for errors
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc((void**)&d_r, row_size * sizeof(float));
    // Check for errors
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc((void**)&d_H_TT, J_T_col_size * J_T_col_size * sizeof(float));
    // Check for errors
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }


    err = cudaMalloc((void**)&d_g_T, J_T_col_size * sizeof(float));
    // Check for errors
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    for(int _=0; _<iterations; _++){
        updateState(
            _observations,
            _incremental_poses,
            _inverse_depths,
            _intrinsics,
            _T_r_to_l,
            _anchor_frame_id,
            _target_frame_id,
            _feat_glob_id,
            _counter,
            d_J_T,
            d_J_alpha,
            d_r,
            d_H_TT,
            d_g_T,
            J_T_col_size,
            J_alpha_col_size,
            row_size
        );
    }

    //////// Copy the output to CPU ////////
    // Delete the previous step
    free(_J_T);
    free(_J_alpha);
    free(_r);
    free(_H_TT);
    free(_g_T);

    // Allocate new memory
    _J_T        = (float *)malloc(row_size * J_T_col_size * sizeof(float));
    _J_alpha    = (float *)malloc(row_size * J_alpha_col_size * sizeof(float));
    _r          = (float *)malloc(row_size * sizeof(float));
    _H_TT       = (float *)malloc(J_T_col_size * J_T_col_size * sizeof(float));
    _g_T        = (float *)malloc(J_T_col_size * sizeof(float));

    // Copy from device to host BEFORE freeing GPU memory
    cudaMemcpy(_J_T, d_J_T, row_size * J_T_col_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(_J_alpha, d_J_alpha, row_size * J_alpha_col_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(_r, d_r, row_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(_H_TT, d_H_TT, J_T_col_size * J_T_col_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(_g_T, d_g_T, J_T_col_size * sizeof(float), cudaMemcpyDeviceToHost);

    //////// Free CUDA memory ////////
    cudaFree(d_J_T);
    cudaFree(d_J_alpha);
    cudaFree(d_r);
    cudaFree(d_H_TT);
}

void Solver::loadCalibration(float *intrinsics, float *T_r_to_l){
    // Load intrinsics and extrinsics into the device
    for(int row_idx = 0; row_idx<4; row_idx++){
        _intrinsics.index_put_({0,row_idx}, intrinsics[row_idx]);
        _intrinsics.index_put_({1,row_idx}, intrinsics[4 + row_idx]);
        for(int col_idx = 0; col_idx<4; col_idx++){
            _T_r_to_l.index_put_({row_idx, col_idx}, T_r_to_l[4*row_idx + col_idx]);
        }
    }
}

void Solver::writeObservations(int anchor_frame_ID, int target_frame_ID, int global_feat_ID, float *left_obs, float *right_obs){
    assert(anchor_frame_ID >= 0 && anchor_frame_ID < _observations.size(0));
    assert(target_frame_ID >= 0 && target_frame_ID < _observations.size(0));
    assert(global_feat_ID >= 0 && global_feat_ID < _observations.size(1));
    assert(_counter < static_cast<int>(_max_meas_size));
    
    // Write the pixel coordinates
    for (int row_idx = 0; row_idx < 3; ++row_idx) {
        _observations.index_put_({target_frame_ID, global_feat_ID, 0, row_idx}, left_obs[row_idx]);
        _observations.index_put_({target_frame_ID, global_feat_ID, 1, row_idx}, right_obs[row_idx]);
    }

    _anchor_frame_id.index_put_({_counter}, anchor_frame_ID);
    _target_frame_id.index_put_({_counter}, target_frame_ID);
    _feat_glob_id.index_put_({_counter}, global_feat_ID);
    ++_counter;
}


void Solver::getIncrementalPose(int keyFrameID, float *T_curr_to_next) {
    assert(keyFrameID >= 0 && keyFrameID < _incremental_poses.size(0));

    // Extract pose from CUDA tensor, copy to CPU
    torch::Tensor pose_cpu = _incremental_poses[keyFrameID].cpu().contiguous();

    // Copy the pose
    std::memcpy(T_curr_to_next, pose_cpu.data_ptr<float>(), sizeof(float) * 16);
}

void Solver::writeIncrementalPose(int keyFrameID, float *T_curr_to_next){
    _incremental_poses.index_put_({keyFrameID, 0, 0}, T_curr_to_next[0]);
    _incremental_poses.index_put_({keyFrameID, 0, 1}, T_curr_to_next[1]);
    _incremental_poses.index_put_({keyFrameID, 0, 2}, T_curr_to_next[2]);
    _incremental_poses.index_put_({keyFrameID, 0, 3}, T_curr_to_next[3]);

    _incremental_poses.index_put_({keyFrameID, 1, 0}, T_curr_to_next[4]);
    _incremental_poses.index_put_({keyFrameID, 1, 1}, T_curr_to_next[5]);
    _incremental_poses.index_put_({keyFrameID, 1, 2}, T_curr_to_next[6]);
    _incremental_poses.index_put_({keyFrameID, 1, 3}, T_curr_to_next[7]);

    _incremental_poses.index_put_({keyFrameID, 2, 0}, T_curr_to_next[8]);
    _incremental_poses.index_put_({keyFrameID, 2, 1}, T_curr_to_next[9]);
    _incremental_poses.index_put_({keyFrameID, 2, 2}, T_curr_to_next[10]);
    _incremental_poses.index_put_({keyFrameID, 2, 3}, T_curr_to_next[11]);
}

void Solver::getObservation(int frame_ID, int global_feat_ID, float *left_obs, float *right_obs){

    assert(frame_ID >= 0 && frame_ID < _observations.size(0));
    assert(global_feat_ID >= 0 && global_feat_ID < _observations.size(1));

    // Extract observation from CUDA tensor, copy to CPU
    torch::Tensor observation_left  = _observations[frame_ID][global_feat_ID][0].cpu().contiguous();
    torch::Tensor observation_right = _observations[frame_ID][global_feat_ID][1].cpu().contiguous();

    // Copy the observation
    std::memcpy(left_obs, observation_left.data_ptr<float>(), sizeof(float) * 3);
    std::memcpy(right_obs, observation_right.data_ptr<float>(), sizeof(float) * 3);
}


void Solver::getCalibration(float *intrinsics, float *T_r_to_l){
    // Copy calibration back to CPU
    torch::Tensor intrinsics_cpu = _intrinsics.cpu().contiguous();
    torch::Tensor T_r_to_l_cpu = _T_r_to_l.cpu().contiguous();

    // Copy the calibration
    std::memcpy(intrinsics, intrinsics_cpu.data_ptr<float>(), sizeof(float) * 8);
    std::memcpy(T_r_to_l, T_r_to_l_cpu.data_ptr<float>(), sizeof(float) * 16);
}


void Solver::getJacobiansAndResidual(float *J_T, float *J_alpha, float *r) {
    // Get sizes
    int num_residuals   = _counter * 4;
    int num_pose_params = _incremental_poses.size(0) * 6;
    int num_points      = _inverse_depths.size(0);

    // Copy values into the output pointers
    std::memcpy(J_T, _J_T, num_residuals * num_pose_params * sizeof(float));
    std::memcpy(J_alpha, _J_alpha, num_residuals * num_points * sizeof(float));
    std::memcpy(r, _r, num_residuals * sizeof(float));  // _r is [N, 1], but you copy as a flat array
}

void Solver::get_H_and_g_for_T(float *H_TT, float *g_T){
    // Get sizes
    int num_pose_params = _incremental_poses.size(0) * 6;

    // Copy values into the output pointers
    std::memcpy(H_TT, _H_TT, num_pose_params * num_pose_params * sizeof(float));
    std::memcpy(g_T, _g_T, num_pose_params * sizeof(float));
}

void Solver::loadInverseDepths(float *inverse_depths){
    // Write the inverse depths
    for (int row_idx = 0; row_idx < _inverse_depths.size(0); ++row_idx) {
        _inverse_depths.index_put_({row_idx}, inverse_depths[row_idx]);
    }
}

void Solver::getInverseDepths(float *inverse_depths) {
    // 1. Move tensors to CPU and ensure they're contiguous
    auto inverse_depths_cpu = _inverse_depths.to(torch::kCPU).contiguous();
    
    // 2. Get raw pointers to the CPU tensors
    float* inverse_depths_data = inverse_depths_cpu.data_ptr<float>();

    // 4. Copy values into the output pointers
    std::memcpy(inverse_depths, inverse_depths_data, _inverse_depths.size(0) * sizeof(float));
}