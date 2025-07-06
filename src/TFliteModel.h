#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace pi {

class TFliteModel {
public:
    explicit TFliteModel(const std::string& model_path);
    virtual ~TFliteModel();

    bool isLoaded() const { return m_is_loaded; }

protected:
    bool invoke(const cv::Mat& input, cv::Mat& output);
    std::unique_ptr<tflite::FlatBufferModel> m_model;
    std::unique_ptr<tflite::Interpreter> m_interpreter;

private:
    // XNNPack delegate for optimized inference on ARM devices
    TfLiteXNNPackDelegate* m_xnnpack_delegate = nullptr;
    bool m_is_loaded = false;
};

} // namespace pi