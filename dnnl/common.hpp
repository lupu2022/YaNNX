template <tt::TensorDataType _DTYPE_>

tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_Flatten(tt::tensor_t input, tt::tensor_t output, int64_t axis) {
    tt::TensorDataType dtype = _DTYPE_;

    auto src_ptr = input->device_float()->plain_ptr();
    auto dst_ptr = input->device_float()->plain_ptr();

    if (dtype == tt::YNX_FLOAT) {
        memcpy(dst_ptr, src_ptr, sizeof(float) * num_items());
        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}
