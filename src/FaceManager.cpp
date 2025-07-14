#include "FaceManager.h"

namespace pi {

FaceManager::FaceManager(const FaceManagerConfig& config)
    : m_faceDetection(config.detection_config), m_faceLandmarks(config.landmarks_config),
      m_run_first_detection(true) {}

bool FaceManager::run(const cv::Mat& imageIn, DetectionBox& detectionBoxInOut, bool static_image_mode) {
    // Run face detection
    if (m_run_first_detection) {
        if (!m_faceDetection.run(imageIn, detectionBoxInOut)) {
            return false; // Early exit if detection fails
        }
        m_run_first_detection = static_image_mode;
    }

    // Run face landmarks
    if (!m_faceLandmarks.run(imageIn, detectionBoxInOut)) {
        m_run_first_detection = true; // Reset for next detection
        return false; // Early exit if landmarks detection fails
    }
    return true; // Successful detection and landmarks extraction
}

void FaceManager::drawBoxAndLandmarks(cv::Mat& image, const DetectionBox& detectionBox,
                                       const std::vector<Point3D>& landmarks,
                                       const cv::Scalar& boxColor, const cv::Scalar& landmarkColor,
                                       int thickness) {
    // Draw the detection box
    cv::RotatedRect rotatedRect(cv::Point2f(detectionBox.center.x, detectionBox.center.y),
                                cv::Size2f(detectionBox.width, detectionBox.height),
                                detectionBox.rotation * RAD2DEG);
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);
    for (int i = 0; i < 4; ++i) {
        cv::line(image, vertices[i], vertices[(i + 1) % 4], boxColor, thickness);
    }

    // Draw landmarks
    for (const auto& landmark : landmarks) {
        cv::circle(image, cv::Point(static_cast<int>(landmark.x), static_cast<int>(landmark.y)),
                   1, landmarkColor, -1);
    }
}

} // namespace pi