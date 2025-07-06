#include "TFliteModel.h"

namespace pi {

TFLiteModel::TFliteModel(const std::string& model_path) {

    // Load the TFLite model from the specified path
    m_model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!m_model) {
        throw std::runtime_error("Failed to load model from " + model_path);
    }

    // Create the interpreter with the model
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*m_model, resolver);
    if (builder(&m_interpreter) != kTfLiteOk || !m_interpreter) {
        throw std::runtime_error("Failed to create interpreter for model " + model_path);
    }

    // Initialize the XNNPack delegate
    TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
    m_xnnpack_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_options);
    if (!m_xnnpack_delegate) {
        throw std::runtime_error("Failed to create XNNPack delegate for model " + model_path);
    }

    // Modify the interpreter to use the XNNPack delegate
    if (m_interpreter->ModifyGraphWithDelegate(m_xnnpack_delegate) != kTfLiteOk) {
        throw std::runtime_error("Failed to modify graph with XNNPack delegate for model " + model_path);
    }

    // Allocate tensors for the interpreter
    if (m_interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors for model " + model_path);
    }

    m_is_loaded = true; // Model loaded successfully
    std::cout << "TFLite model loaded successfully from " << model_path << std::endl;
}

TFLiteModel::~TFLiteModel() {
    // Clean up the XNNPack delegate
    if (m_xnnpack_delegate) {
        TfLiteXNNPackDelegateDelete(m_xnnpack_delegate);
    }
}

} // namespace pi