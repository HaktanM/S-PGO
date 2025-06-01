#include "Manager.hpp"


void Manager::addObservation(size_t landmark_idx, Eigen::Vector3d pixel_coordinate, int frame_idx, float weight) {
    // Check if the landmark exists
    auto it = _landmarks.find(landmark_idx);
    if (it == _landmarks.end()) {
        std::cerr << "Error: Landmark with index " << landmark_idx << " not found." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Create a new observation
    Observation obs;
    obs._pixel_coordinate = pixel_coordinate;
    obs._frame_idx = frame_idx;
    obs._weigth = weight;

    // Add the observation to the landmark
    it->second._observations.push_back(obs);
}

void Manager::createLandmark(int anchor_frame_idx, float inv_depth){
    // Initialize a new landmark
    Landmark landmark(anchor_frame_idx, inv_depth);

    // Append this landmark to our map
    _landmarks.insert({_landmark_counter, landmark});
    
    // Increase our landmark counter
    ++_landmark_counter;
};