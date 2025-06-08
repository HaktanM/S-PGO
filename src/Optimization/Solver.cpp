#include "Solver.hpp"



void Solver::step(int iterations){
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
            _counter
        );
    }
}

void Solver::loadCalibration(float *intrinsics, float *T_r_to_l){
    // Load intrinsics and extrinsics into the device
    for(int row_idx = 0; row_idx<4; row_idx++){
        _intrinsics.index_put_({row_idx,0}, intrinsics[row_idx]);
        _intrinsics.index_put_({row_idx,1}, intrinsics[4 + row_idx]);
        for(int col_idx = 0; col_idx<4; col_idx++){
            _T_r_to_l.index_put_({row_idx, col_idx}, T_r_to_l[4*row_idx + col_idx]);
        }
    }
}

void Solver::writeObservations(int anchor_frame_ID, int target_frame_ID, int global_feat_ID, float *left_obs, float *right_obs){
    assert(anchor_frame_ID >= 0 && anchor_frame_ID < _observations.size(0));
    assert(target_frame_ID >= 0 && target_frame_ID < _observations.size(0));
    assert(global_feat_ID >= 0 && global_feat_ID < _observations.size(1));
    assert(_counter < static_cast<int>(_max_meas_size-1));
    
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