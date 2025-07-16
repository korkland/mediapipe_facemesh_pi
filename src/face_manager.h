#pragma once

#include "face_detection.h"
#include "face_landmarks.h"

namespace pi {

class FaceManager {
public:
    explicit FaceManager(const FaceManagerConfig& config);
    bool Run(const cv::Mat& image_in, DetectionBox& detection_box_in_out, bool static_image_mode = false);
    const std::vector<Point3D>& GetLandmarks() const {return face_landmarks_.GetLandmarks();}

    // Debug
    static void draw_box_and_landmarks(cv::Mat& image, const DetectionBox& detection_box, const std::vector<Point3D>& landmarks,
                                        const cv::Scalar& box_color = cv::Scalar(0, 0, 255),
                                        const cv::Scalar& landmark_color = cv::Scalar(0, 255, 0),
                                        int thickness = 2);

private:
    FaceDetection face_detection_; // Face detection instance
    FaceLandmarks face_landmarks_; // Face landmarks instance
    bool run_first_detection_; // Flag to indicate if this is the first detection run
};
} // namespace pi