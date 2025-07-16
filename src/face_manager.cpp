#include "face_manager.h"

namespace pi {

FaceManager::FaceManager(const FaceManagerConfig& config)
    : face_detection_(config.detection_config), face_landmarks_(config.landmarks_config),
      run_first_detection_(true) {}

bool FaceManager::Run(const cv::Mat& image_in, DetectionBox& detection_box_in_out, bool static_image_mode) {
    // Run face detection
    if (run_first_detection_) {
        if (!face_detection_.Run(image_in, detection_box_in_out)) {
            return false; // Early exit if detection fails
        }
        run_first_detection_ = static_image_mode;
    }

    // Run face landmarks
    if (!face_landmarks_.Run(image_in, detection_box_in_out)) {
        run_first_detection_ = true; // Reset for next detection
        return false; // Early exit if landmarks detection fails
    }
    return true; // Successful detection and landmarks extraction
}

void FaceManager::draw_box_and_landmarks(cv::Mat& image, const DetectionBox& detection_box,
                                       const std::vector<Point3D>& landmarks,
                                       const cv::Scalar& box_color, const cv::Scalar& landmark_color,
                                       int thickness) {
    // Draw the detection box
    cv::RotatedRect rotated_rect(cv::Point2f(detection_box.center.x, detection_box.center.y),
                                cv::Size2f(detection_box.width, detection_box.height),
                                detection_box.rotation * RAD2DEG);
    cv::Point2f vertices[4];
    rotated_rect.points(vertices);
    for (int i = 0; i < 4; ++i) {
        cv::line(image, vertices[i], vertices[(i + 1) % 4], box_color, thickness);
    }

    // Draw landmarks
    for (const auto& landmark : landmarks) {
        cv::circle(image, cv::Point(static_cast<int>(landmark.x), static_cast<int>(landmark.y)),
                   1, landmark_color, -1);
    }
}

} // namespace pi