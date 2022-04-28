template <DataType _DTYPE_>
Result<ShapeType, TensorError> CPUTensor<_DTYPE_>::amax() {
    DataType dtype = _DTYPE_;
    if ( dtype == DataType::Float) {
        auto data = (const float *)plain_ptr();

        
        float maxv = fabs(data[0]);
        size_t position = 1;
        for(size_t i = 1; i < items();  i++) {
            auto v = fabs(data[i]);
            if ( v > maxv) {
                maxv = v;
                position = i;
            }
        }

        std::vector<size_t> result = shape();
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = position / strides_[i];
            position = position % strides_[i];
        }
        return Ok(ShapeType(result));
    }
    return Err(TensorError("CPUTensor.amax: type error for cpu"));
}

template <DataType _DTYPE_>
Result<ScalarValue, TensorError> CPUTensor<_DTYPE_>::sum() {
    DataType dtype = _DTYPE_;
    if ( dtype == DataType::Float) {
        auto data = (const float *)plain_ptr();
        float result = 0.0;
        for(size_t i = 1; i < items();  i++) {
            result += data[i];
        }
        return Ok(ScalarValue(result));
    }
    return Err(TensorError("CPUTensor.sum type error for cpu"));
}