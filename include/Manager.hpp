#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

struct Observation{
    // Pixel coordinate of the tracked feature
    Eigen::Vector3d _pixel_coordinate;

    // This is the index of the frame, where the observation belongs
    int _frame_idx;

    // Weigth of the observation
    float _weigth;
};


struct Landmark{
private:
    int _anchor_frame_idx;
public:
    float _inv_depth;
    std::vector<Observation> _observations;

    Landmark(int anchor_frame_idx, float inv_depth)
        : _anchor_frame_idx(anchor_frame_idx), _inv_depth(inv_depth){};

    // This is thw way to read the anchor frame index
    int anchorFrameIdx() const {
        return _anchor_frame_idx;
    }
};


struct Manager{
public:
    size_t _landmark_counter{0};

    // Map between the landmark index and landmark
    std::unordered_map<size_t, Landmark> _landmarks;

    void addObservation(size_t landmark_idx, Eigen::Vector3d pixel_coordinate, int frame_idx, float weight);
    void createLandmark(int anchor_frame_idx, float inv_depth);
};