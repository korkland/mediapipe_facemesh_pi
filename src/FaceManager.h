#pragma once

#include "FaceDetection.h"
#include "FaceLandmarks.h"

namespace pi {

class FaceManager {
public:
    explicit FaceManager(const FaceManagerConfig& config);
    bool run(const cv::Mat& imageIn, DetectionBox& detectionBoxInOut, bool static_image_mode = false);
    const std::vector<Point3D>& getLandmarks() const {return m_faceLandmarks.getLandmarks();}

    // Debug
    static void drawBoxAndLandmarks(cv::Mat& image, const DetectionBox& detectionBox, const std::vector<Point3D>& landmarks,
                                    const cv::Scalar& boxColor = cv::Scalar(0, 0, 255),
                                    const cv::Scalar& landmarkColor = cv::Scalar(0, 255, 0),
                                    int thickness = 2);

private:
    FaceDetection m_faceDetection; // Face detection instance
    FaceLandmarks m_faceLandmarks; // Face landmarks instance
    bool m_run_first_detection; // Flag to indicate if this is the first detection run
};
} // namespace pi