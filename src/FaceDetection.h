#pragma once

#include "TFliteModel.h"

namespace pi {

class FaceDetection : public TFliteModel {
public:
    explicit FaceDetection(const std::string& model_path);
    virtual ~FaceDetection() = default;
    bool invoke(const cv::Mat& input, cv::Mat& output) override;

private:

    bool preprocessInput(const cv::Mat& input, std::vector<float>& input_data);
    bool postprocessOutput(const std::vector<float>& output_data, cv::Mat& output

} // namespace pi