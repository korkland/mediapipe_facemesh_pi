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

    inline int GetInputTensorCount() const { return m_interpreter->inputs().size(); }
    inline int GetOutputTensorCount() const { return m_interpreter->outputs().size(); }

    // Get input tensor dimensions
    inline const TfLiteIntArray* GetInputTensorShape(int tensorIdx) const {
        int input_tensor_idx = m_interpreter->inputs()[tensorIdx];
        return m_interpreter->tensor(input_tensor_idx)->dims;
    }

    // Get output tensor dimensions
    inline const TfLiteIntArray* GetOutputTensorShape(int tensorIdx) const {
        int output_tensor_idx = m_interpreter->outputs()[tensorIdx];
        return m_interpreter->tensor(output_tensor_idx)->dims;
    }

    // Get input tensor data
    inline float* GetInputTensorData(int tensorIdx) {
        int input_tensor_idx = m_interpreter->inputs()[tensorIdx];
        return m_interpreter->typed_tensor<float>(input_tensor_idx);
    }

    // Get output tensor data
    inline float* GetOutputTensorData(int tensorIdx) {
        int output_tensor_idx = m_interpreter->outputs()[tensorIdx];
        return m_interpreter->typed_tensor<float>(output_tensor_idx);
    }

protected:
    std::unique_ptr<tflite::FlatBufferModel> m_model;
    std::unique_ptr<tflite::Interpreter> m_interpreter;

private:
    // XNNPack delegate for optimized inference on ARM devices
    TfLiteDelegate* m_xnnpack_delegate = nullptr;
};

} // namespace pi