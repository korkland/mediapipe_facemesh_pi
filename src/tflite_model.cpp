#include "tflite_model.h"
#include "common_defines.h"
#include "tensorflow/lite/kernels/register.h"

namespace pi {

TfLiteModel::TfLiteModel(const std::string& model_path) {

    // Load the TFLite model from the specified path
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    ASSERT(model_ != nullptr, "Failed to load model from " + model_path);

    // Create the interpreter with the model
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    ASSERT(builder(&interpreter_) == kTfLiteOk && interpreter_ != nullptr,
           "Failed to create interpreter for model " + model_path);

    // Initialize the XNNPack delegate
    TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_delegate_ = TfLiteXNNPackDelegateCreate(&xnnpack_options);
    ASSERT(xnnpack_delegate_ != nullptr, "Failed to create XNNPack delegate for model " + model_path);

    // Modify the interpreter to use the XNNPack delegate
    ASSERT(interpreter_->ModifyGraphWithDelegate(xnnpack_delegate_) == kTfLiteOk,
           "Failed to modify graph with XNNPack delegate for model " + model_path);

    // Allocate tensors for the interpreter
    ASSERT(interpreter_->AllocateTensors() == kTfLiteOk,
           "Failed to allocate tensors for model " + model_path);

    std::cout << "TFLite model loaded successfully from " << model_path << std::endl;
}

TfLiteModel::~TfLiteModel() {
    // Clean up the XNNPack delegate
    if (xnnpack_delegate_) {
        TfLiteXNNPackDelegateDelete(xnnpack_delegate_);
    }
}

} // namespace pi