#pragma once

#include "TFliteModel.h"
#include "CommonDefines.h"

namespace pi {

class FaceDetection : public TFliteModel {
public:
    explicit FaceDetection(const std::string& model_path);
    virtual ~FaceDetection() = default;
    bool run(const cv::Mat& imageIn /* RGB */, DetectionBox& detectionBoxOut);

private:

    bool preprocessInput(const cv::Mat& input, std::vector<float>& input_data);
    bool postprocessOutput(const std::vector<float>& output_data, cv::Mat& output);

} // namespace pi