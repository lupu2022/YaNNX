// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Abs
OperatorReturnType onnx_Abs(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Abs(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Acos
OperatorReturnType onnx_Acos(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Acos(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Acosh
OperatorReturnType onnx_Acosh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Acosh(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
OperatorReturnType onnx_Add(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_Add(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#And
OperatorReturnType onnx_And(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_And(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMax
OperatorReturnType onnx_ArgMax(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > keepdims, std::variant<void *, int64_t > select_last_index) override {
    return impl()->onnx_ArgMax(data, reduced, axis, keepdims, select_last_index);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMin
OperatorReturnType onnx_ArgMin(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > keepdims, std::variant<void *, int64_t > select_last_index) override {
    return impl()->onnx_ArgMin(data, reduced, axis, keepdims, select_last_index);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asin
OperatorReturnType onnx_Asin(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Asin(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asinh
OperatorReturnType onnx_Asinh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Asinh(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Atan
OperatorReturnType onnx_Atan(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Atan(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Atanh
OperatorReturnType onnx_Atanh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Atanh(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool
OperatorReturnType onnx_AveragePool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, int64_t > ceil_mode, std::variant<void *, int64_t > count_include_pad, std::vector<int64_t> kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) override {
    return impl()->onnx_AveragePool(X, Y, auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization
OperatorReturnType onnx_BatchNormalization(/*inputs:*/ tensor_t X, tensor_t scale, tensor_t B, tensor_t input_mean, tensor_t input_var, /*outputs:*/ tensor_t Y, std::variant<void *, tensor_t>& running_mean, std::variant<void *, tensor_t>& running_var, /*attributes:*/ std::variant<void *, float > epsilon, std::variant<void *, float > momentum, std::variant<void *, int64_t > training_mode) override {
    return impl()->onnx_BatchNormalization(X, scale, B, input_mean, input_var, Y, running_mean, running_var, epsilon, momentum, training_mode);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Bernoulli
OperatorReturnType onnx_Bernoulli(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, float > seed) override {
    return impl()->onnx_Bernoulli(input, output, dtype, seed);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitShift
OperatorReturnType onnx_BitShift(/*inputs:*/ tensor_t X, tensor_t Y, /*outputs:*/ tensor_t Z, /*attributes:*/ std::string direction) override {
    return impl()->onnx_BitShift(X, Y, Z, direction);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
OperatorReturnType onnx_Cast(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t to) override {
    return impl()->onnx_Cast(input, output, to);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#CastLike
OperatorReturnType onnx_CastLike(/*inputs:*/ tensor_t input, tensor_t target_type, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_CastLike(input, target_type, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Ceil
OperatorReturnType onnx_Ceil(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Ceil(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Celu
OperatorReturnType onnx_Celu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha) override {
    return impl()->onnx_Celu(X, Y, alpha);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip
OperatorReturnType onnx_Clip(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& min, std::variant<void *, tensor_t>& max, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Clip(input, min, max, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Compress
OperatorReturnType onnx_Compress(/*inputs:*/ tensor_t input, tensor_t condition, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_Compress(input, condition, output, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
OperatorReturnType onnx_Concat(/*inputs:*/ std::vector<tensor_t>& inputs, /*outputs:*/ tensor_t concat_result, /*attributes:*/ int64_t axis) override {
    return impl()->onnx_Concat(inputs, concat_result, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConcatFromSequence
OperatorReturnType onnx_ConcatFromSequence(/*inputs:*/ tensor_t input_sequence, /*outputs:*/ tensor_t concat_result, /*attributes:*/ int64_t axis, std::variant<void *, int64_t > new_axis) override {
    return impl()->onnx_ConcatFromSequence(input_sequence, concat_result, axis, new_axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
OperatorReturnType onnx_Conv(/*inputs:*/ tensor_t X, tensor_t W, std::variant<void *, tensor_t>& B, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, std::vector<int64_t> > dilations, std::variant<void *, int64_t > group, std::variant<void *, std::vector<int64_t> > kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) override {
    return impl()->onnx_Conv(X, W, B, Y, auto_pad, dilations, group, kernel_shape, pads, strides);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvInteger
OperatorReturnType onnx_ConvInteger(/*inputs:*/ tensor_t x, tensor_t w, std::variant<void *, tensor_t>& x_zero_point, std::variant<void *, tensor_t>& w_zero_point, /*outputs:*/ tensor_t y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, std::vector<int64_t> > dilations, std::variant<void *, int64_t > group, std::variant<void *, std::vector<int64_t> > kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) override {
    return impl()->onnx_ConvInteger(x, w, x_zero_point, w_zero_point, y, auto_pad, dilations, group, kernel_shape, pads, strides);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
OperatorReturnType onnx_ConvTranspose(/*inputs:*/ tensor_t X, tensor_t W, std::variant<void *, tensor_t>& B, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, std::vector<int64_t> > dilations, std::variant<void *, int64_t > group, std::variant<void *, std::vector<int64_t> > kernel_shape, std::variant<void *, std::vector<int64_t> > output_padding, std::variant<void *, std::vector<int64_t> > output_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) override {
    return impl()->onnx_ConvTranspose(X, W, B, Y, auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cos
OperatorReturnType onnx_Cos(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Cos(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cosh
OperatorReturnType onnx_Cosh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Cosh(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
OperatorReturnType onnx_CumSum(/*inputs:*/ tensor_t x, tensor_t axis, /*outputs:*/ tensor_t y, /*attributes:*/ std::variant<void *, int64_t > exclusive, std::variant<void *, int64_t > reverse) override {
    return impl()->onnx_CumSum(x, axis, y, exclusive, reverse);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#DepthToSpace
OperatorReturnType onnx_DepthToSpace(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t blocksize, std::variant<void *, std::string > mode) override {
    return impl()->onnx_DepthToSpace(input, output, blocksize, mode);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#DequantizeLinear
OperatorReturnType onnx_DequantizeLinear(/*inputs:*/ tensor_t x, tensor_t x_scale, std::variant<void *, tensor_t>& x_zero_point, /*outputs:*/ tensor_t y, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_DequantizeLinear(x, x_scale, x_zero_point, y, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Det
OperatorReturnType onnx_Det(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Det(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div
OperatorReturnType onnx_Div(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_Div(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout
OperatorReturnType onnx_Dropout(/*inputs:*/ tensor_t data, std::variant<void *, tensor_t>& ratio, std::variant<void *, tensor_t>& training_mode, /*outputs:*/ tensor_t output, std::variant<void *, tensor_t>& mask, /*attributes:*/ std::variant<void *, int64_t > seed) override {
    return impl()->onnx_Dropout(data, ratio, training_mode, output, mask, seed);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#DynamicQuantizeLinear
OperatorReturnType onnx_DynamicQuantizeLinear(/*inputs:*/ tensor_t x, /*outputs:*/ tensor_t y, tensor_t y_scale, tensor_t y_zero_point) override {
    return impl()->onnx_DynamicQuantizeLinear(x, y, y_scale, y_zero_point);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Einsum
OperatorReturnType onnx_Einsum(/*inputs:*/ std::vector<tensor_t>& Inputs, /*outputs:*/ tensor_t Output, /*attributes:*/ std::string equation) override {
    return impl()->onnx_Einsum(Inputs, Output, equation);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Elu
OperatorReturnType onnx_Elu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha) override {
    return impl()->onnx_Elu(X, Y, alpha);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Equal
OperatorReturnType onnx_Equal(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_Equal(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Erf
OperatorReturnType onnx_Erf(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Erf(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Exp
OperatorReturnType onnx_Exp(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Exp(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand
OperatorReturnType onnx_Expand(/*inputs:*/ tensor_t input, tensor_t shape, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Expand(input, shape, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#EyeLike
OperatorReturnType onnx_EyeLike(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, int64_t > k) override {
    return impl()->onnx_EyeLike(input, output, dtype, k);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten
OperatorReturnType onnx_Flatten(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_Flatten(input, output, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Floor
OperatorReturnType onnx_Floor(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Floor(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU
OperatorReturnType onnx_GRU(/*inputs:*/ tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, /*outputs:*/ std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, /*attributes:*/ std::variant<void *, std::vector<float> > activation_alpha, std::variant<void *, std::vector<float> > activation_beta, std::variant<void *, std::vector<std::string> > activations, std::variant<void *, float > clip, std::variant<void *, std::string > direction, std::variant<void *, int64_t > hidden_size, std::variant<void *, int64_t > layout, std::variant<void *, int64_t > linear_before_reset) override {
    return impl()->onnx_GRU(X, W, R, B, sequence_lens, initial_h, Y, Y_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout, linear_before_reset);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
OperatorReturnType onnx_Gather(/*inputs:*/ tensor_t data, tensor_t indices, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_Gather(data, indices, output, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements
OperatorReturnType onnx_GatherElements(/*inputs:*/ tensor_t data, tensor_t indices, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_GatherElements(data, indices, output, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
OperatorReturnType onnx_GatherND(/*inputs:*/ tensor_t data, tensor_t indices, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > batch_dims) override {
    return impl()->onnx_GatherND(data, indices, output, batch_dims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
OperatorReturnType onnx_Gemm(/*inputs:*/ tensor_t A, tensor_t B, std::variant<void *, tensor_t>& C, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha, std::variant<void *, float > beta, std::variant<void *, int64_t > transA, std::variant<void *, int64_t > transB) override {
    return impl()->onnx_Gemm(A, B, C, Y, alpha, beta, transA, transB);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool
OperatorReturnType onnx_GlobalAveragePool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_GlobalAveragePool(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalLpPool
OperatorReturnType onnx_GlobalLpPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, int64_t > p) override {
    return impl()->onnx_GlobalLpPool(X, Y, p);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool
OperatorReturnType onnx_GlobalMaxPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_GlobalMaxPool(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater
OperatorReturnType onnx_Greater(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_Greater(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GreaterOrEqual
OperatorReturnType onnx_GreaterOrEqual(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_GreaterOrEqual(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid
OperatorReturnType onnx_HardSigmoid(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha, std::variant<void *, float > beta) override {
    return impl()->onnx_HardSigmoid(X, Y, alpha, beta);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSwish
OperatorReturnType onnx_HardSwish(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_HardSwish(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Hardmax
OperatorReturnType onnx_Hardmax(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_Hardmax(input, output, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity
OperatorReturnType onnx_Identity(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Identity(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization
OperatorReturnType onnx_InstanceNormalization(/*inputs:*/ tensor_t input, tensor_t scale, tensor_t B, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, float > epsilon) override {
    return impl()->onnx_InstanceNormalization(input, scale, B, output, epsilon);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsInf
OperatorReturnType onnx_IsInf(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, int64_t > detect_negative, std::variant<void *, int64_t > detect_positive) override {
    return impl()->onnx_IsInf(X, Y, detect_negative, detect_positive);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsNaN
OperatorReturnType onnx_IsNaN(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_IsNaN(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN
OperatorReturnType onnx_LRN(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha, std::variant<void *, float > beta, std::variant<void *, float > bias, int64_t size) override {
    return impl()->onnx_LRN(X, Y, alpha, beta, bias, size);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM
OperatorReturnType onnx_LSTM(/*inputs:*/ tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, std::variant<void *, tensor_t>& initial_c, std::variant<void *, tensor_t>& P, /*outputs:*/ std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, std::variant<void *, tensor_t>& Y_c, /*attributes:*/ std::variant<void *, std::vector<float> > activation_alpha, std::variant<void *, std::vector<float> > activation_beta, std::variant<void *, std::vector<std::string> > activations, std::variant<void *, float > clip, std::variant<void *, std::string > direction, std::variant<void *, int64_t > hidden_size, std::variant<void *, int64_t > input_forget, std::variant<void *, int64_t > layout) override {
    return impl()->onnx_LSTM(X, W, R, B, sequence_lens, initial_h, initial_c, P, Y, Y_h, Y_c, activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget, layout);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LeakyRelu
OperatorReturnType onnx_LeakyRelu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha) override {
    return impl()->onnx_LeakyRelu(X, Y, alpha);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less
OperatorReturnType onnx_Less(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_Less(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LessOrEqual
OperatorReturnType onnx_LessOrEqual(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_LessOrEqual(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log
OperatorReturnType onnx_Log(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Log(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LogSoftmax
OperatorReturnType onnx_LogSoftmax(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_LogSoftmax(input, output, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpNormalization
OperatorReturnType onnx_LpNormalization(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > p) override {
    return impl()->onnx_LpNormalization(input, output, axis, p);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpPool
OperatorReturnType onnx_LpPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::vector<int64_t> kernel_shape, std::variant<void *, int64_t > p, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) override {
    return impl()->onnx_LpPool(X, Y, auto_pad, kernel_shape, p, pads, strides);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul
OperatorReturnType onnx_MatMul(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_MatMul(A, B, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMulInteger
OperatorReturnType onnx_MatMulInteger(/*inputs:*/ tensor_t A, tensor_t B, std::variant<void *, tensor_t>& a_zero_point, std::variant<void *, tensor_t>& b_zero_point, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_MatMulInteger(A, B, a_zero_point, b_zero_point, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Max
OperatorReturnType onnx_Max(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t max) override {
    return impl()->onnx_Max(data_0, max);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
OperatorReturnType onnx_MaxPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, std::variant<void *, tensor_t>& Indices, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, int64_t > ceil_mode, std::variant<void *, std::vector<int64_t> > dilations, std::vector<int64_t> kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, int64_t > storage_order, std::variant<void *, std::vector<int64_t> > strides) override {
    return impl()->onnx_MaxPool(X, Y, Indices, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxRoiPool
OperatorReturnType onnx_MaxRoiPool(/*inputs:*/ tensor_t X, tensor_t rois, /*outputs:*/ tensor_t Y, /*attributes:*/ std::vector<int64_t> pooled_shape, std::variant<void *, float > spatial_scale) override {
    return impl()->onnx_MaxRoiPool(X, rois, Y, pooled_shape, spatial_scale);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxUnpool
OperatorReturnType onnx_MaxUnpool(/*inputs:*/ tensor_t X, tensor_t I, std::variant<void *, tensor_t>& output_shape, /*outputs:*/ tensor_t output, /*attributes:*/ std::vector<int64_t> kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) override {
    return impl()->onnx_MaxUnpool(X, I, output_shape, output, kernel_shape, pads, strides);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mean
OperatorReturnType onnx_Mean(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t mean) override {
    return impl()->onnx_Mean(data_0, mean);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MeanVarianceNormalization
OperatorReturnType onnx_MeanVarianceNormalization(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes) override {
    return impl()->onnx_MeanVarianceNormalization(X, Y, axes);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min
OperatorReturnType onnx_Min(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t min) override {
    return impl()->onnx_Min(data_0, min);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mod
OperatorReturnType onnx_Mod(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C, /*attributes:*/ std::variant<void *, int64_t > fmod) override {
    return impl()->onnx_Mod(A, B, C, fmod);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
OperatorReturnType onnx_Mul(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_Mul(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Multinomial
OperatorReturnType onnx_Multinomial(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, int64_t > sample_size, std::variant<void *, float > seed) override {
    return impl()->onnx_Multinomial(input, output, dtype, sample_size, seed);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Neg
OperatorReturnType onnx_Neg(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Neg(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#NegativeLogLikelihoodLoss
OperatorReturnType onnx_NegativeLogLikelihoodLoss(/*inputs:*/ tensor_t input, tensor_t target, std::variant<void *, tensor_t>& weight, /*outputs:*/ tensor_t loss, /*attributes:*/ std::variant<void *, int64_t > ignore_index, std::variant<void *, std::string > reduction) override {
    return impl()->onnx_NegativeLogLikelihoodLoss(input, target, weight, loss, ignore_index, reduction);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression
OperatorReturnType onnx_NonMaxSuppression(/*inputs:*/ tensor_t boxes, tensor_t scores, std::variant<void *, tensor_t>& max_output_boxes_per_class, std::variant<void *, tensor_t>& iou_threshold, std::variant<void *, tensor_t>& score_threshold, /*outputs:*/ tensor_t selected_indices, /*attributes:*/ std::variant<void *, int64_t > center_point_box) override {
    return impl()->onnx_NonMaxSuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, selected_indices, center_point_box);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonZero
OperatorReturnType onnx_NonZero(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_NonZero(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Not
OperatorReturnType onnx_Not(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Not(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#OneHot
OperatorReturnType onnx_OneHot(/*inputs:*/ tensor_t indices, tensor_t depth, tensor_t values, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_OneHot(indices, depth, values, output, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Or
OperatorReturnType onnx_Or(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_Or(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu
OperatorReturnType onnx_PRelu(/*inputs:*/ tensor_t X, tensor_t slope, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_PRelu(X, slope, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad
OperatorReturnType onnx_Pad(/*inputs:*/ tensor_t data, tensor_t pads, std::variant<void *, tensor_t>& constant_value, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, std::string > mode) override {
    return impl()->onnx_Pad(data, pads, constant_value, output, mode);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow
OperatorReturnType onnx_Pow(/*inputs:*/ tensor_t X, tensor_t Y, /*outputs:*/ tensor_t Z) override {
    return impl()->onnx_Pow(X, Y, Z);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv
OperatorReturnType onnx_QLinearConv(/*inputs:*/ tensor_t x, tensor_t x_scale, tensor_t x_zero_point, tensor_t w, tensor_t w_scale, tensor_t w_zero_point, tensor_t y_scale, tensor_t y_zero_point, std::variant<void *, tensor_t>& B, /*outputs:*/ tensor_t y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, std::vector<int64_t> > dilations, std::variant<void *, int64_t > group, std::variant<void *, std::vector<int64_t> > kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) override {
    return impl()->onnx_QLinearConv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, y, auto_pad, dilations, group, kernel_shape, pads, strides);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearMatMul
OperatorReturnType onnx_QLinearMatMul(/*inputs:*/ tensor_t a, tensor_t a_scale, tensor_t a_zero_point, tensor_t b, tensor_t b_scale, tensor_t b_zero_point, tensor_t y_scale, tensor_t y_zero_point, /*outputs:*/ tensor_t y) override {
    return impl()->onnx_QLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear
OperatorReturnType onnx_QuantizeLinear(/*inputs:*/ tensor_t x, tensor_t y_scale, std::variant<void *, tensor_t>& y_zero_point, /*outputs:*/ tensor_t y, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_QuantizeLinear(x, y_scale, y_zero_point, y, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN
OperatorReturnType onnx_RNN(/*inputs:*/ tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, /*outputs:*/ std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, /*attributes:*/ std::variant<void *, std::vector<float> > activation_alpha, std::variant<void *, std::vector<float> > activation_beta, std::variant<void *, std::vector<std::string> > activations, std::variant<void *, float > clip, std::variant<void *, std::string > direction, std::variant<void *, int64_t > hidden_size, std::variant<void *, int64_t > layout) override {
    return impl()->onnx_RNN(X, W, R, B, sequence_lens, initial_h, Y, Y_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormal
OperatorReturnType onnx_RandomNormal(/*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, float > mean, std::variant<void *, float > scale, std::variant<void *, float > seed, std::vector<int64_t> shape) override {
    return impl()->onnx_RandomNormal(output, dtype, mean, scale, seed, shape);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormalLike
OperatorReturnType onnx_RandomNormalLike(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, float > mean, std::variant<void *, float > scale, std::variant<void *, float > seed) override {
    return impl()->onnx_RandomNormalLike(input, output, dtype, mean, scale, seed);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniform
OperatorReturnType onnx_RandomUniform(/*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, float > high, std::variant<void *, float > low, std::variant<void *, float > seed, std::vector<int64_t> shape) override {
    return impl()->onnx_RandomUniform(output, dtype, high, low, seed, shape);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniformLike
OperatorReturnType onnx_RandomUniformLike(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, float > high, std::variant<void *, float > low, std::variant<void *, float > seed) override {
    return impl()->onnx_RandomUniformLike(input, output, dtype, high, low, seed);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range
OperatorReturnType onnx_Range(/*inputs:*/ tensor_t start, tensor_t limit, tensor_t delta, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Range(start, limit, delta, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reciprocal
OperatorReturnType onnx_Reciprocal(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Reciprocal(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL1
OperatorReturnType onnx_ReduceL1(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) override {
    return impl()->onnx_ReduceL1(data, reduced, axes, keepdims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2
OperatorReturnType onnx_ReduceL2(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) override {
    return impl()->onnx_ReduceL2(data, reduced, axes, keepdims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSum
OperatorReturnType onnx_ReduceLogSum(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) override {
    return impl()->onnx_ReduceLogSum(data, reduced, axes, keepdims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSumExp
OperatorReturnType onnx_ReduceLogSumExp(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) override {
    return impl()->onnx_ReduceLogSumExp(data, reduced, axes, keepdims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMax
OperatorReturnType onnx_ReduceMax(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) override {
    return impl()->onnx_ReduceMax(data, reduced, axes, keepdims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
OperatorReturnType onnx_ReduceMean(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) override {
    return impl()->onnx_ReduceMean(data, reduced, axes, keepdims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMin
OperatorReturnType onnx_ReduceMin(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) override {
    return impl()->onnx_ReduceMin(data, reduced, axes, keepdims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceProd
OperatorReturnType onnx_ReduceProd(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) override {
    return impl()->onnx_ReduceProd(data, reduced, axes, keepdims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum
OperatorReturnType onnx_ReduceSum(/*inputs:*/ tensor_t data, std::variant<void *, tensor_t>& axes, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, int64_t > keepdims, std::variant<void *, int64_t > noop_with_empty_axes) override {
    return impl()->onnx_ReduceSum(data, axes, reduced, keepdims, noop_with_empty_axes);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSumSquare
OperatorReturnType onnx_ReduceSumSquare(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) override {
    return impl()->onnx_ReduceSumSquare(data, reduced, axes, keepdims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
OperatorReturnType onnx_Relu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Relu(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape
OperatorReturnType onnx_Reshape(/*inputs:*/ tensor_t data, tensor_t shape, /*outputs:*/ tensor_t reshaped, /*attributes:*/ std::variant<void *, int64_t > allowzero) override {
    return impl()->onnx_Reshape(data, shape, reshaped, allowzero);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
OperatorReturnType onnx_Resize(/*inputs:*/ tensor_t X, std::variant<void *, tensor_t>& roi, std::variant<void *, tensor_t>& scales, std::variant<void *, tensor_t>& sizes, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > coordinate_transformation_mode, std::variant<void *, float > cubic_coeff_a, std::variant<void *, int64_t > exclude_outside, std::variant<void *, float > extrapolation_value, std::variant<void *, std::string > mode, std::variant<void *, std::string > nearest_mode) override {
    return impl()->onnx_Resize(X, roi, scales, sizes, Y, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReverseSequence
OperatorReturnType onnx_ReverseSequence(/*inputs:*/ tensor_t input, tensor_t sequence_lens, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, int64_t > batch_axis, std::variant<void *, int64_t > time_axis) override {
    return impl()->onnx_ReverseSequence(input, sequence_lens, Y, batch_axis, time_axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RoiAlign
OperatorReturnType onnx_RoiAlign(/*inputs:*/ tensor_t X, tensor_t rois, tensor_t batch_indices, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > coordinate_transformation_mode, std::variant<void *, std::string > mode, std::variant<void *, int64_t > output_height, std::variant<void *, int64_t > output_width, std::variant<void *, int64_t > sampling_ratio, std::variant<void *, float > spatial_scale) override {
    return impl()->onnx_RoiAlign(X, rois, batch_indices, Y, coordinate_transformation_mode, mode, output_height, output_width, sampling_ratio, spatial_scale);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Round
OperatorReturnType onnx_Round(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Round(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements
OperatorReturnType onnx_ScatterElements(/*inputs:*/ tensor_t data, tensor_t indices, tensor_t updates, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_ScatterElements(data, indices, updates, output, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND
OperatorReturnType onnx_ScatterND(/*inputs:*/ tensor_t data, tensor_t indices, tensor_t updates, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_ScatterND(data, indices, updates, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Selu
OperatorReturnType onnx_Selu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha, std::variant<void *, float > gamma) override {
    return impl()->onnx_Selu(X, Y, alpha, gamma);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceAt
OperatorReturnType onnx_SequenceAt(/*inputs:*/ tensor_t input_sequence, tensor_t position, /*outputs:*/ tensor_t tensor) override {
    return impl()->onnx_SequenceAt(input_sequence, position, tensor);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceConstruct
OperatorReturnType onnx_SequenceConstruct(/*inputs:*/ std::vector<tensor_t>& inputs, /*outputs:*/ tensor_t output_sequence) override {
    return impl()->onnx_SequenceConstruct(inputs, output_sequence);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceEmpty
OperatorReturnType onnx_SequenceEmpty(/*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype) override {
    return impl()->onnx_SequenceEmpty(output, dtype);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceErase
OperatorReturnType onnx_SequenceErase(/*inputs:*/ tensor_t input_sequence, std::variant<void *, tensor_t>& position, /*outputs:*/ tensor_t output_sequence) override {
    return impl()->onnx_SequenceErase(input_sequence, position, output_sequence);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceInsert
OperatorReturnType onnx_SequenceInsert(/*inputs:*/ tensor_t input_sequence, tensor_t tensor, std::variant<void *, tensor_t>& position, /*outputs:*/ tensor_t output_sequence) override {
    return impl()->onnx_SequenceInsert(input_sequence, tensor, position, output_sequence);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceLength
OperatorReturnType onnx_SequenceLength(/*inputs:*/ tensor_t input_sequence, /*outputs:*/ tensor_t length) override {
    return impl()->onnx_SequenceLength(input_sequence, length);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape
OperatorReturnType onnx_Shape(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t shape, /*attributes:*/ std::variant<void *, int64_t > end, std::variant<void *, int64_t > start) override {
    return impl()->onnx_Shape(data, shape, end, start);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shrink
OperatorReturnType onnx_Shrink(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, float > bias, std::variant<void *, float > lambd) override {
    return impl()->onnx_Shrink(input, output, bias, lambd);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid
OperatorReturnType onnx_Sigmoid(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Sigmoid(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sign
OperatorReturnType onnx_Sign(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Sign(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sin
OperatorReturnType onnx_Sin(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Sin(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sinh
OperatorReturnType onnx_Sinh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Sinh(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size
OperatorReturnType onnx_Size(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t size) override {
    return impl()->onnx_Size(data, size);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice
OperatorReturnType onnx_Slice(/*inputs:*/ tensor_t data, tensor_t starts, tensor_t ends, std::variant<void *, tensor_t>& axes, std::variant<void *, tensor_t>& steps, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Slice(data, starts, ends, axes, steps, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
OperatorReturnType onnx_Softmax(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_Softmax(input, output, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SoftmaxCrossEntropyLoss
OperatorReturnType onnx_SoftmaxCrossEntropyLoss(/*inputs:*/ tensor_t scores, tensor_t labels, std::variant<void *, tensor_t>& weights, /*outputs:*/ tensor_t output, std::variant<void *, tensor_t>& log_prob, /*attributes:*/ std::variant<void *, int64_t > ignore_index, std::variant<void *, std::string > reduction) override {
    return impl()->onnx_SoftmaxCrossEntropyLoss(scores, labels, weights, output, log_prob, ignore_index, reduction);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softplus
OperatorReturnType onnx_Softplus(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Softplus(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softsign
OperatorReturnType onnx_Softsign(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Softsign(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SpaceToDepth
OperatorReturnType onnx_SpaceToDepth(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t blocksize) override {
    return impl()->onnx_SpaceToDepth(input, output, blocksize);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
OperatorReturnType onnx_Split(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& split, /*outputs:*/ std::vector<tensor_t>& outputs, /*attributes:*/ std::variant<void *, int64_t > axis) override {
    return impl()->onnx_Split(input, split, outputs, axis);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SplitToSequence
OperatorReturnType onnx_SplitToSequence(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& split, /*outputs:*/ tensor_t output_sequence, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > keepdims) override {
    return impl()->onnx_SplitToSequence(input, split, output_sequence, axis, keepdims);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt
OperatorReturnType onnx_Sqrt(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) override {
    return impl()->onnx_Sqrt(X, Y);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Squeeze
OperatorReturnType onnx_Squeeze(/*inputs:*/ tensor_t data, std::variant<void *, tensor_t>& axes, /*outputs:*/ tensor_t squeezed) override {
    return impl()->onnx_Squeeze(data, axes, squeezed);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#StringNormalizer
OperatorReturnType onnx_StringNormalizer(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > case_change_action, std::variant<void *, int64_t > is_case_sensitive, std::variant<void *, std::string > locale, std::variant<void *, std::vector<std::string> > stopwords) override {
    return impl()->onnx_StringNormalizer(X, Y, case_change_action, is_case_sensitive, locale, stopwords);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub
OperatorReturnType onnx_Sub(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_Sub(A, B, C);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sum
OperatorReturnType onnx_Sum(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t sum) override {
    return impl()->onnx_Sum(data_0, sum);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tan
OperatorReturnType onnx_Tan(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Tan(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh
OperatorReturnType onnx_Tanh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Tanh(input, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#TfIdfVectorizer
OperatorReturnType onnx_TfIdfVectorizer(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ int64_t max_gram_length, int64_t max_skip_count, int64_t min_gram_length, std::string mode, std::vector<int64_t> ngram_counts, std::vector<int64_t> ngram_indexes, std::variant<void *, std::vector<int64_t> > pool_int64s, std::variant<void *, std::vector<std::string> > pool_strings, std::variant<void *, std::vector<float> > weights) override {
    return impl()->onnx_TfIdfVectorizer(X, Y, max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings, weights);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ThresholdedRelu
OperatorReturnType onnx_ThresholdedRelu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha) override {
    return impl()->onnx_ThresholdedRelu(X, Y, alpha);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile
OperatorReturnType onnx_Tile(/*inputs:*/ tensor_t input, tensor_t repeats, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Tile(input, repeats, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK
OperatorReturnType onnx_TopK(/*inputs:*/ tensor_t X, tensor_t K, /*outputs:*/ tensor_t Values, tensor_t Indices, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > largest, std::variant<void *, int64_t > sorted) override {
    return impl()->onnx_TopK(X, K, Values, Indices, axis, largest, sorted);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose
OperatorReturnType onnx_Transpose(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t transposed, /*attributes:*/ std::variant<void *, std::vector<int64_t> > perm) override {
    return impl()->onnx_Transpose(data, transposed, perm);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Trilu
OperatorReturnType onnx_Trilu(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& k, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > upper) override {
    return impl()->onnx_Trilu(input, k, output, upper);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unique
OperatorReturnType onnx_Unique(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, std::variant<void *, tensor_t>& indices, std::variant<void *, tensor_t>& inverse_indices, std::variant<void *, tensor_t>& counts, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > sorted) override {
    return impl()->onnx_Unique(X, Y, indices, inverse_indices, counts, axis, sorted);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze
OperatorReturnType onnx_Unsqueeze(/*inputs:*/ tensor_t data, tensor_t axes, /*outputs:*/ tensor_t expanded) override {
    return impl()->onnx_Unsqueeze(data, axes, expanded);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where
OperatorReturnType onnx_Where(/*inputs:*/ tensor_t condition, tensor_t X, tensor_t Y, /*outputs:*/ tensor_t output) override {
    return impl()->onnx_Where(condition, X, Y, output);
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor
OperatorReturnType onnx_Xor(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) override {
    return impl()->onnx_Xor(A, B, C);
}

