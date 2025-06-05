#include "Solver.hpp"



void Solver::writeObservations(int anchor_frame_ID, int target_frame_ID, int global_feat_ID, float *left_obs, float *right_obs){
    assert(anchor_frame_ID >= 0 && anchor_frame_ID < _observations.size(0));
    assert(target_frame_ID >= 0 && target_frame_ID < _observations.size(0));
    assert(global_feat_ID >= 0 && global_feat_ID < _observations.size(1));
    assert(_counter < static_cast<int>(_max_meas_size-1));
    
    // Write the pixel coordinates
    for (int i = 0; i < 3; ++i) {
        _observations.index_put_({target_frame_ID, global_feat_ID, 0, i}, left_obs[i]);
        _observations.index_put_({target_frame_ID, global_feat_ID, 1, i}, right_obs[i]);
    }

    _anchor_frame_id.index_put_({_counter, 0}, anchor_frame_ID);
    _target_frame_id.index_put_({_counter, 0}, target_frame_ID);
    _feat_glob_id.index_put_({_counter, 0}, global_feat_ID);
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