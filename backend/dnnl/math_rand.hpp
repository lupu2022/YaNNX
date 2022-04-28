#include <random>
template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::randn_(float mean, float scale) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.randn_: type error"));
    }

    auto total = items();
    float* mem = (float *)plain_ptr();

    std::normal_distribution<float> distribution(mean,  scale);
    for (size_t i = 0; i < total; i++) {
        mem[i] = distribution(tt::RANDOM_GENERATOR);
    }

    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::randu_(float low, float up) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.randu_: type error"));
    }

    auto total = items();
    float* mem = (float *)plain_ptr();

    std::uniform_real_distribution<float> distribution(low, up);
    for (size_t i = 0; i < total; i++) {
        mem[i] = distribution(tt::RANDOM_GENERATOR);
    }

    return Ok();
}
