#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace pi {

class TfLiteModel {
public:
    explicit TfLiteModel(const std::string& model_path);
    virtual ~TfLiteModel();

    inline int GetInputTensorCount() const { return interpreter_->inputs().size(); }
    inline int GetOutputTensorCount() const { return interpreter_->outputs().size(); }

    // Get input tensor dimensions
    inline const TfLiteIntArray* GetInputTensorShape(int tensor_idx) const {
        int input_tensor_idx = interpreter_->inputs()[tensor_idx];
        return interpreter_->tensor(input_tensor_idx)->dims;
    }

    // Get output tensor dimensions
    inline const TfLiteIntArray* GetOutputTensorShape(int tensor_idx) const {
        int output_tensor_idx = interpreter_->outputs()[tensor_idx];
        return interpreter_->tensor(output_tensor_idx)->dims;
    }

    // Get input tensor data
    inline float* GetInputTensorData(int tensor_idx) {
        int input_tensor_idx = interpreter_->inputs()[tensor_idx];
        return interpreter_->typed_tensor<float>(input_tensor_idx);
    }

    // Get output tensor data
    inline float* GetOutputTensorData(int tensor_idx) {
        int output_tensor_idx = interpreter_->outputs()[tensor_idx];
        return interpreter_->typed_tensor<float>(output_tensor_idx);
    }

protected:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;

private:
    // XNNPack delegate for optimized inference on ARM devices
    TfLiteDelegate* xnnpack_delegate_ = nullptr;
};

} // namespace pi