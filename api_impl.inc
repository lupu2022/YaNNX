OperatorReturnType onnx_Abs(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Abs(X, Y);
}

OperatorReturnType onnx_Acos(tensor_t input, tensor_t output) override {
    return impl()->onnx_Acos(input, output);
}

OperatorReturnType onnx_Acosh(tensor_t input, tensor_t output) override {
    return impl()->onnx_Acosh(input, output);
}

OperatorReturnType onnx_Add(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_Add(A, B, C);
}

OperatorReturnType onnx_And(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_And(A, B, C);
}

OperatorReturnType onnx_ArgMax(tensor_t data, tensor_t reduced, int64_t axis, int64_t keepdims, int64_t select_last_index) override {
    return impl()->onnx_ArgMax(data, reduced, axis, keepdims, select_last_index);
}

OperatorReturnType onnx_ArgMin(tensor_t data, tensor_t reduced, int64_t axis, int64_t keepdims, int64_t select_last_index) override {
    return impl()->onnx_ArgMin(data, reduced, axis, keepdims, select_last_index);
}

OperatorReturnType onnx_Asin(tensor_t input, tensor_t output) override {
    return impl()->onnx_Asin(input, output);
}

OperatorReturnType onnx_Asinh(tensor_t input, tensor_t output) override {
    return impl()->onnx_Asinh(input, output);
}

OperatorReturnType onnx_Atan(tensor_t input, tensor_t output) override {
    return impl()->onnx_Atan(input, output);
}

OperatorReturnType onnx_Atanh(tensor_t input, tensor_t output) override {
    return impl()->onnx_Atanh(input, output);
}

OperatorReturnType onnx_AveragePool(tensor_t X, tensor_t Y, std::string auto_pad, int64_t ceil_mode, int64_t count_include_pad, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) override {
    return impl()->onnx_AveragePool(X, Y, auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides);
}

OperatorReturnType onnx_BatchNormalization(tensor_t X, tensor_t scale, tensor_t B, tensor_t input_mean, tensor_t input_var, tensor_t Y, std::variant<void *, tensor_t>& running_mean, std::variant<void *, tensor_t>& running_var, float epsilon, float momentum, int64_t training_mode) override {
    return impl()->onnx_BatchNormalization(X, scale, B, input_mean, input_var, Y, running_mean, running_var, epsilon, momentum, training_mode);
}

OperatorReturnType onnx_Bernoulli(tensor_t input, tensor_t output, int64_t dtype, float seed) override {
    return impl()->onnx_Bernoulli(input, output, dtype, seed);
}

OperatorReturnType onnx_BitShift(tensor_t X, tensor_t Y, tensor_t Z, std::string direction) override {
    return impl()->onnx_BitShift(X, Y, Z, direction);
}

OperatorReturnType onnx_Cast(tensor_t input, tensor_t output, int64_t to) override {
    return impl()->onnx_Cast(input, output, to);
}

OperatorReturnType onnx_CastLike(tensor_t input, tensor_t target_type, tensor_t output) override {
    return impl()->onnx_CastLike(input, target_type, output);
}

OperatorReturnType onnx_Ceil(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Ceil(X, Y);
}

OperatorReturnType onnx_Celu(tensor_t X, tensor_t Y, float alpha) override {
    return impl()->onnx_Celu(X, Y, alpha);
}

OperatorReturnType onnx_Clip(tensor_t input, std::variant<void *, tensor_t>& min, std::variant<void *, tensor_t>& max, tensor_t output) override {
    return impl()->onnx_Clip(input, min, max, output);
}

OperatorReturnType onnx_Compress(tensor_t input, tensor_t condition, tensor_t output, int64_t axis) override {
    return impl()->onnx_Compress(input, condition, output, axis);
}

OperatorReturnType onnx_Concat(std::vector<tensor_t>& inputs, tensor_t concat_result, int64_t axis) override {
    return impl()->onnx_Concat(inputs, concat_result, axis);
}

OperatorReturnType onnx_ConcatFromSequence(tensor_t input_sequence, tensor_t concat_result, int64_t axis, int64_t new_axis) override {
    return impl()->onnx_ConcatFromSequence(input_sequence, concat_result, axis, new_axis);
}

OperatorReturnType onnx_Conv(tensor_t X, tensor_t W, std::variant<void *, tensor_t>& B, tensor_t Y, std::string auto_pad, std::vector<int64_t> dilations, int64_t group, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) override {
    return impl()->onnx_Conv(X, W, B, Y, auto_pad, dilations, group, kernel_shape, pads, strides);
}

OperatorReturnType onnx_ConvInteger(tensor_t x, tensor_t w, std::variant<void *, tensor_t>& x_zero_point, std::variant<void *, tensor_t>& w_zero_point, tensor_t y, std::string auto_pad, std::vector<int64_t> dilations, int64_t group, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) override {
    return impl()->onnx_ConvInteger(x, w, x_zero_point, w_zero_point, y, auto_pad, dilations, group, kernel_shape, pads, strides);
}

OperatorReturnType onnx_ConvTranspose(tensor_t X, tensor_t W, std::variant<void *, tensor_t>& B, tensor_t Y, std::string auto_pad, std::vector<int64_t> dilations, int64_t group, std::vector<int64_t> kernel_shape, std::vector<int64_t> output_padding, std::vector<int64_t> output_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) override {
    return impl()->onnx_ConvTranspose(X, W, B, Y, auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides);
}

OperatorReturnType onnx_Cos(tensor_t input, tensor_t output) override {
    return impl()->onnx_Cos(input, output);
}

OperatorReturnType onnx_Cosh(tensor_t input, tensor_t output) override {
    return impl()->onnx_Cosh(input, output);
}

OperatorReturnType onnx_CumSum(tensor_t x, tensor_t axis, tensor_t y, int64_t exclusive, int64_t reverse) override {
    return impl()->onnx_CumSum(x, axis, y, exclusive, reverse);
}

OperatorReturnType onnx_DepthToSpace(tensor_t input, tensor_t output, int64_t blocksize, std::string mode) override {
    return impl()->onnx_DepthToSpace(input, output, blocksize, mode);
}

OperatorReturnType onnx_DequantizeLinear(tensor_t x, tensor_t x_scale, std::variant<void *, tensor_t>& x_zero_point, tensor_t y, int64_t axis) override {
    return impl()->onnx_DequantizeLinear(x, x_scale, x_zero_point, y, axis);
}

OperatorReturnType onnx_Det(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Det(X, Y);
}

OperatorReturnType onnx_Div(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_Div(A, B, C);
}

OperatorReturnType onnx_Dropout(tensor_t data, std::variant<void *, tensor_t>& ratio, std::variant<void *, tensor_t>& training_mode, tensor_t output, std::variant<void *, tensor_t>& mask, int64_t seed) override {
    return impl()->onnx_Dropout(data, ratio, training_mode, output, mask, seed);
}

OperatorReturnType onnx_DynamicQuantizeLinear(tensor_t x, tensor_t y, tensor_t y_scale, tensor_t y_zero_point) override {
    return impl()->onnx_DynamicQuantizeLinear(x, y, y_scale, y_zero_point);
}

OperatorReturnType onnx_Einsum(std::vector<tensor_t>& Inputs, tensor_t Output, std::string equation) override {
    return impl()->onnx_Einsum(Inputs, Output, equation);
}

OperatorReturnType onnx_Elu(tensor_t X, tensor_t Y, float alpha) override {
    return impl()->onnx_Elu(X, Y, alpha);
}

OperatorReturnType onnx_Equal(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_Equal(A, B, C);
}

OperatorReturnType onnx_Erf(tensor_t input, tensor_t output) override {
    return impl()->onnx_Erf(input, output);
}

OperatorReturnType onnx_Exp(tensor_t input, tensor_t output) override {
    return impl()->onnx_Exp(input, output);
}

OperatorReturnType onnx_Expand(tensor_t input, tensor_t shape, tensor_t output) override {
    return impl()->onnx_Expand(input, shape, output);
}

OperatorReturnType onnx_EyeLike(tensor_t input, tensor_t output, int64_t dtype, int64_t k) override {
    return impl()->onnx_EyeLike(input, output, dtype, k);
}

OperatorReturnType onnx_Flatten(tensor_t input, tensor_t output, int64_t axis) override {
    return impl()->onnx_Flatten(input, output, axis);
}

OperatorReturnType onnx_Floor(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Floor(X, Y);
}

OperatorReturnType onnx_GRU(tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, std::vector<float> activation_alpha, std::vector<float> activation_beta, std::vector<std::string> activations, float clip, std::string direction, int64_t hidden_size, int64_t layout, int64_t linear_before_reset) override {
    return impl()->onnx_GRU(X, W, R, B, sequence_lens, initial_h, Y, Y_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout, linear_before_reset);
}

OperatorReturnType onnx_Gather(tensor_t data, tensor_t indices, tensor_t output, int64_t axis) override {
    return impl()->onnx_Gather(data, indices, output, axis);
}

OperatorReturnType onnx_GatherElements(tensor_t data, tensor_t indices, tensor_t output, int64_t axis) override {
    return impl()->onnx_GatherElements(data, indices, output, axis);
}

OperatorReturnType onnx_GatherND(tensor_t data, tensor_t indices, tensor_t output, int64_t batch_dims) override {
    return impl()->onnx_GatherND(data, indices, output, batch_dims);
}

OperatorReturnType onnx_Gemm(tensor_t A, tensor_t B, std::variant<void *, tensor_t>& C, tensor_t Y, float alpha, float beta, int64_t transA, int64_t transB) override {
    return impl()->onnx_Gemm(A, B, C, Y, alpha, beta, transA, transB);
}

OperatorReturnType onnx_GlobalAveragePool(tensor_t X, tensor_t Y) override {
    return impl()->onnx_GlobalAveragePool(X, Y);
}

OperatorReturnType onnx_GlobalLpPool(tensor_t X, tensor_t Y, int64_t p) override {
    return impl()->onnx_GlobalLpPool(X, Y, p);
}

OperatorReturnType onnx_GlobalMaxPool(tensor_t X, tensor_t Y) override {
    return impl()->onnx_GlobalMaxPool(X, Y);
}

OperatorReturnType onnx_Greater(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_Greater(A, B, C);
}

OperatorReturnType onnx_GreaterOrEqual(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_GreaterOrEqual(A, B, C);
}

OperatorReturnType onnx_HardSigmoid(tensor_t X, tensor_t Y, float alpha, float beta) override {
    return impl()->onnx_HardSigmoid(X, Y, alpha, beta);
}

OperatorReturnType onnx_HardSwish(tensor_t X, tensor_t Y) override {
    return impl()->onnx_HardSwish(X, Y);
}

OperatorReturnType onnx_Hardmax(tensor_t input, tensor_t output, int64_t axis) override {
    return impl()->onnx_Hardmax(input, output, axis);
}

OperatorReturnType onnx_Identity(tensor_t input, tensor_t output) override {
    return impl()->onnx_Identity(input, output);
}

OperatorReturnType onnx_InstanceNormalization(tensor_t input, tensor_t scale, tensor_t B, tensor_t output, float epsilon) override {
    return impl()->onnx_InstanceNormalization(input, scale, B, output, epsilon);
}

OperatorReturnType onnx_IsInf(tensor_t X, tensor_t Y, int64_t detect_negative, int64_t detect_positive) override {
    return impl()->onnx_IsInf(X, Y, detect_negative, detect_positive);
}

OperatorReturnType onnx_IsNaN(tensor_t X, tensor_t Y) override {
    return impl()->onnx_IsNaN(X, Y);
}

OperatorReturnType onnx_LRN(tensor_t X, tensor_t Y, float alpha, float beta, float bias, int64_t size) override {
    return impl()->onnx_LRN(X, Y, alpha, beta, bias, size);
}

OperatorReturnType onnx_LSTM(tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, std::variant<void *, tensor_t>& initial_c, std::variant<void *, tensor_t>& P, std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, std::variant<void *, tensor_t>& Y_c, std::vector<float> activation_alpha, std::vector<float> activation_beta, std::vector<std::string> activations, float clip, std::string direction, int64_t hidden_size, int64_t input_forget, int64_t layout) override {
    return impl()->onnx_LSTM(X, W, R, B, sequence_lens, initial_h, initial_c, P, Y, Y_h, Y_c, activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget, layout);
}

OperatorReturnType onnx_LeakyRelu(tensor_t X, tensor_t Y, float alpha) override {
    return impl()->onnx_LeakyRelu(X, Y, alpha);
}

OperatorReturnType onnx_Less(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_Less(A, B, C);
}

OperatorReturnType onnx_LessOrEqual(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_LessOrEqual(A, B, C);
}

OperatorReturnType onnx_Log(tensor_t input, tensor_t output) override {
    return impl()->onnx_Log(input, output);
}

OperatorReturnType onnx_LogSoftmax(tensor_t input, tensor_t output, int64_t axis) override {
    return impl()->onnx_LogSoftmax(input, output, axis);
}

OperatorReturnType onnx_LpNormalization(tensor_t input, tensor_t output, int64_t axis, int64_t p) override {
    return impl()->onnx_LpNormalization(input, output, axis, p);
}

OperatorReturnType onnx_LpPool(tensor_t X, tensor_t Y, std::string auto_pad, std::vector<int64_t> kernel_shape, int64_t p, std::vector<int64_t> pads, std::vector<int64_t> strides) override {
    return impl()->onnx_LpPool(X, Y, auto_pad, kernel_shape, p, pads, strides);
}

OperatorReturnType onnx_MatMul(tensor_t A, tensor_t B, tensor_t Y) override {
    return impl()->onnx_MatMul(A, B, Y);
}

OperatorReturnType onnx_MatMulInteger(tensor_t A, tensor_t B, std::variant<void *, tensor_t>& a_zero_point, std::variant<void *, tensor_t>& b_zero_point, tensor_t Y) override {
    return impl()->onnx_MatMulInteger(A, B, a_zero_point, b_zero_point, Y);
}

OperatorReturnType onnx_Max(std::vector<tensor_t>& data_0, tensor_t max) override {
    return impl()->onnx_Max(data_0, max);
}

OperatorReturnType onnx_MaxPool(tensor_t X, tensor_t Y, std::variant<void *, tensor_t>& Indices, std::string auto_pad, int64_t ceil_mode, std::vector<int64_t> dilations, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, int64_t storage_order, std::vector<int64_t> strides) override {
    return impl()->onnx_MaxPool(X, Y, Indices, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides);
}

OperatorReturnType onnx_MaxRoiPool(tensor_t X, tensor_t rois, tensor_t Y, std::vector<int64_t> pooled_shape, float spatial_scale) override {
    return impl()->onnx_MaxRoiPool(X, rois, Y, pooled_shape, spatial_scale);
}

OperatorReturnType onnx_MaxUnpool(tensor_t X, tensor_t I, std::variant<void *, tensor_t>& output_shape, tensor_t output, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) override {
    return impl()->onnx_MaxUnpool(X, I, output_shape, output, kernel_shape, pads, strides);
}

OperatorReturnType onnx_Mean(std::vector<tensor_t>& data_0, tensor_t mean) override {
    return impl()->onnx_Mean(data_0, mean);
}

OperatorReturnType onnx_MeanVarianceNormalization(tensor_t X, tensor_t Y, std::vector<int64_t> axes) override {
    return impl()->onnx_MeanVarianceNormalization(X, Y, axes);
}

OperatorReturnType onnx_Min(std::vector<tensor_t>& data_0, tensor_t min) override {
    return impl()->onnx_Min(data_0, min);
}

OperatorReturnType onnx_Mod(tensor_t A, tensor_t B, tensor_t C, int64_t fmod) override {
    return impl()->onnx_Mod(A, B, C, fmod);
}

OperatorReturnType onnx_Mul(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_Mul(A, B, C);
}

OperatorReturnType onnx_Multinomial(tensor_t input, tensor_t output, int64_t dtype, int64_t sample_size, float seed) override {
    return impl()->onnx_Multinomial(input, output, dtype, sample_size, seed);
}

OperatorReturnType onnx_Neg(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Neg(X, Y);
}

OperatorReturnType onnx_NegativeLogLikelihoodLoss(tensor_t input, tensor_t target, std::variant<void *, tensor_t>& weight, tensor_t loss, int64_t ignore_index, std::string reduction) override {
    return impl()->onnx_NegativeLogLikelihoodLoss(input, target, weight, loss, ignore_index, reduction);
}

OperatorReturnType onnx_NonMaxSuppression(tensor_t boxes, tensor_t scores, std::variant<void *, tensor_t>& max_output_boxes_per_class, std::variant<void *, tensor_t>& iou_threshold, std::variant<void *, tensor_t>& score_threshold, tensor_t selected_indices, int64_t center_point_box) override {
    return impl()->onnx_NonMaxSuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, selected_indices, center_point_box);
}

OperatorReturnType onnx_NonZero(tensor_t X, tensor_t Y) override {
    return impl()->onnx_NonZero(X, Y);
}

OperatorReturnType onnx_Not(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Not(X, Y);
}

OperatorReturnType onnx_OneHot(tensor_t indices, tensor_t depth, tensor_t values, tensor_t output, int64_t axis) override {
    return impl()->onnx_OneHot(indices, depth, values, output, axis);
}

OperatorReturnType onnx_Or(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_Or(A, B, C);
}

OperatorReturnType onnx_PRelu(tensor_t X, tensor_t slope, tensor_t Y) override {
    return impl()->onnx_PRelu(X, slope, Y);
}

OperatorReturnType onnx_Pad(tensor_t data, tensor_t pads, std::variant<void *, tensor_t>& constant_value, tensor_t output, std::string mode) override {
    return impl()->onnx_Pad(data, pads, constant_value, output, mode);
}

OperatorReturnType onnx_Pow(tensor_t X, tensor_t Y, tensor_t Z) override {
    return impl()->onnx_Pow(X, Y, Z);
}

OperatorReturnType onnx_QLinearConv(tensor_t x, tensor_t x_scale, tensor_t x_zero_point, tensor_t w, tensor_t w_scale, tensor_t w_zero_point, tensor_t y_scale, tensor_t y_zero_point, std::variant<void *, tensor_t>& B, tensor_t y, std::string auto_pad, std::vector<int64_t> dilations, int64_t group, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) override {
    return impl()->onnx_QLinearConv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, y, auto_pad, dilations, group, kernel_shape, pads, strides);
}

OperatorReturnType onnx_QLinearMatMul(tensor_t a, tensor_t a_scale, tensor_t a_zero_point, tensor_t b, tensor_t b_scale, tensor_t b_zero_point, tensor_t y_scale, tensor_t y_zero_point, tensor_t y) override {
    return impl()->onnx_QLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, y);
}

OperatorReturnType onnx_QuantizeLinear(tensor_t x, tensor_t y_scale, std::variant<void *, tensor_t>& y_zero_point, tensor_t y, int64_t axis) override {
    return impl()->onnx_QuantizeLinear(x, y_scale, y_zero_point, y, axis);
}

OperatorReturnType onnx_RNN(tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, std::vector<float> activation_alpha, std::vector<float> activation_beta, std::vector<std::string> activations, float clip, std::string direction, int64_t hidden_size, int64_t layout) override {
    return impl()->onnx_RNN(X, W, R, B, sequence_lens, initial_h, Y, Y_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout);
}

OperatorReturnType onnx_RandomNormal(tensor_t output, int64_t dtype, float mean, float scale, float seed, std::vector<int64_t> shape) override {
    return impl()->onnx_RandomNormal(output, dtype, mean, scale, seed, shape);
}

OperatorReturnType onnx_RandomNormalLike(tensor_t input, tensor_t output, int64_t dtype, float mean, float scale, float seed) override {
    return impl()->onnx_RandomNormalLike(input, output, dtype, mean, scale, seed);
}

OperatorReturnType onnx_RandomUniform(tensor_t output, int64_t dtype, float high, float low, float seed, std::vector<int64_t> shape) override {
    return impl()->onnx_RandomUniform(output, dtype, high, low, seed, shape);
}

OperatorReturnType onnx_RandomUniformLike(tensor_t input, tensor_t output, int64_t dtype, float high, float low, float seed) override {
    return impl()->onnx_RandomUniformLike(input, output, dtype, high, low, seed);
}

OperatorReturnType onnx_Range(tensor_t start, tensor_t limit, tensor_t delta, tensor_t output) override {
    return impl()->onnx_Range(start, limit, delta, output);
}

OperatorReturnType onnx_Reciprocal(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Reciprocal(X, Y);
}

OperatorReturnType onnx_ReduceL1(tensor_t data, tensor_t reduced, std::vector<int64_t> axes, int64_t keepdims) override {
    return impl()->onnx_ReduceL1(data, reduced, axes, keepdims);
}

OperatorReturnType onnx_ReduceL2(tensor_t data, tensor_t reduced, std::vector<int64_t> axes, int64_t keepdims) override {
    return impl()->onnx_ReduceL2(data, reduced, axes, keepdims);
}

OperatorReturnType onnx_ReduceLogSum(tensor_t data, tensor_t reduced, std::vector<int64_t> axes, int64_t keepdims) override {
    return impl()->onnx_ReduceLogSum(data, reduced, axes, keepdims);
}

OperatorReturnType onnx_ReduceLogSumExp(tensor_t data, tensor_t reduced, std::vector<int64_t> axes, int64_t keepdims) override {
    return impl()->onnx_ReduceLogSumExp(data, reduced, axes, keepdims);
}

OperatorReturnType onnx_ReduceMax(tensor_t data, tensor_t reduced, std::vector<int64_t> axes, int64_t keepdims) override {
    return impl()->onnx_ReduceMax(data, reduced, axes, keepdims);
}

OperatorReturnType onnx_ReduceMean(tensor_t data, tensor_t reduced, std::vector<int64_t> axes, int64_t keepdims) override {
    return impl()->onnx_ReduceMean(data, reduced, axes, keepdims);
}

OperatorReturnType onnx_ReduceMin(tensor_t data, tensor_t reduced, std::vector<int64_t> axes, int64_t keepdims) override {
    return impl()->onnx_ReduceMin(data, reduced, axes, keepdims);
}

OperatorReturnType onnx_ReduceProd(tensor_t data, tensor_t reduced, std::vector<int64_t> axes, int64_t keepdims) override {
    return impl()->onnx_ReduceProd(data, reduced, axes, keepdims);
}

OperatorReturnType onnx_ReduceSum(tensor_t data, std::variant<void *, tensor_t>& axes, tensor_t reduced, int64_t keepdims, int64_t noop_with_empty_axes) override {
    return impl()->onnx_ReduceSum(data, axes, reduced, keepdims, noop_with_empty_axes);
}

OperatorReturnType onnx_ReduceSumSquare(tensor_t data, tensor_t reduced, std::vector<int64_t> axes, int64_t keepdims) override {
    return impl()->onnx_ReduceSumSquare(data, reduced, axes, keepdims);
}

OperatorReturnType onnx_Relu(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Relu(X, Y);
}

OperatorReturnType onnx_Reshape(tensor_t data, tensor_t shape, tensor_t reshaped, int64_t allowzero) override {
    return impl()->onnx_Reshape(data, shape, reshaped, allowzero);
}

OperatorReturnType onnx_Resize(tensor_t X, std::variant<void *, tensor_t>& roi, std::variant<void *, tensor_t>& scales, std::variant<void *, tensor_t>& sizes, tensor_t Y, std::string coordinate_transformation_mode, float cubic_coeff_a, int64_t exclude_outside, float extrapolation_value, std::string mode, std::string nearest_mode) override {
    return impl()->onnx_Resize(X, roi, scales, sizes, Y, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
}

OperatorReturnType onnx_ReverseSequence(tensor_t input, tensor_t sequence_lens, tensor_t Y, int64_t batch_axis, int64_t time_axis) override {
    return impl()->onnx_ReverseSequence(input, sequence_lens, Y, batch_axis, time_axis);
}

OperatorReturnType onnx_RoiAlign(tensor_t X, tensor_t rois, tensor_t batch_indices, tensor_t Y, std::string coordinate_transformation_mode, std::string mode, int64_t output_height, int64_t output_width, int64_t sampling_ratio, float spatial_scale) override {
    return impl()->onnx_RoiAlign(X, rois, batch_indices, Y, coordinate_transformation_mode, mode, output_height, output_width, sampling_ratio, spatial_scale);
}

OperatorReturnType onnx_Round(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Round(X, Y);
}

OperatorReturnType onnx_ScatterElements(tensor_t data, tensor_t indices, tensor_t updates, tensor_t output, int64_t axis) override {
    return impl()->onnx_ScatterElements(data, indices, updates, output, axis);
}

OperatorReturnType onnx_ScatterND(tensor_t data, tensor_t indices, tensor_t updates, tensor_t output) override {
    return impl()->onnx_ScatterND(data, indices, updates, output);
}

OperatorReturnType onnx_Selu(tensor_t X, tensor_t Y, float alpha, float gamma) override {
    return impl()->onnx_Selu(X, Y, alpha, gamma);
}

OperatorReturnType onnx_SequenceAt(tensor_t input_sequence, tensor_t position, tensor_t tensor) override {
    return impl()->onnx_SequenceAt(input_sequence, position, tensor);
}

OperatorReturnType onnx_SequenceConstruct(std::vector<tensor_t>& inputs, tensor_t output_sequence) override {
    return impl()->onnx_SequenceConstruct(inputs, output_sequence);
}

OperatorReturnType onnx_SequenceEmpty(tensor_t output, int64_t dtype) override {
    return impl()->onnx_SequenceEmpty(output, dtype);
}

OperatorReturnType onnx_SequenceErase(tensor_t input_sequence, std::variant<void *, tensor_t>& position, tensor_t output_sequence) override {
    return impl()->onnx_SequenceErase(input_sequence, position, output_sequence);
}

OperatorReturnType onnx_SequenceInsert(tensor_t input_sequence, tensor_t tensor, std::variant<void *, tensor_t>& position, tensor_t output_sequence) override {
    return impl()->onnx_SequenceInsert(input_sequence, tensor, position, output_sequence);
}

OperatorReturnType onnx_SequenceLength(tensor_t input_sequence, tensor_t length) override {
    return impl()->onnx_SequenceLength(input_sequence, length);
}

OperatorReturnType onnx_Shape(tensor_t data, tensor_t shape, int64_t end, int64_t start) override {
    return impl()->onnx_Shape(data, shape, end, start);
}

OperatorReturnType onnx_Shrink(tensor_t input, tensor_t output, float bias, float lambd) override {
    return impl()->onnx_Shrink(input, output, bias, lambd);
}

OperatorReturnType onnx_Sigmoid(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Sigmoid(X, Y);
}

OperatorReturnType onnx_Sign(tensor_t input, tensor_t output) override {
    return impl()->onnx_Sign(input, output);
}

OperatorReturnType onnx_Sin(tensor_t input, tensor_t output) override {
    return impl()->onnx_Sin(input, output);
}

OperatorReturnType onnx_Sinh(tensor_t input, tensor_t output) override {
    return impl()->onnx_Sinh(input, output);
}

OperatorReturnType onnx_Size(tensor_t data, tensor_t size) override {
    return impl()->onnx_Size(data, size);
}

OperatorReturnType onnx_Slice(tensor_t data, tensor_t starts, tensor_t ends, std::variant<void *, tensor_t>& axes, std::variant<void *, tensor_t>& steps, tensor_t output) override {
    return impl()->onnx_Slice(data, starts, ends, axes, steps, output);
}

OperatorReturnType onnx_Softmax(tensor_t input, tensor_t output, int64_t axis) override {
    return impl()->onnx_Softmax(input, output, axis);
}

OperatorReturnType onnx_SoftmaxCrossEntropyLoss(tensor_t scores, tensor_t labels, std::variant<void *, tensor_t>& weights, tensor_t output, std::variant<void *, tensor_t>& log_prob, int64_t ignore_index, std::string reduction) override {
    return impl()->onnx_SoftmaxCrossEntropyLoss(scores, labels, weights, output, log_prob, ignore_index, reduction);
}

OperatorReturnType onnx_Softplus(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Softplus(X, Y);
}

OperatorReturnType onnx_Softsign(tensor_t input, tensor_t output) override {
    return impl()->onnx_Softsign(input, output);
}

OperatorReturnType onnx_SpaceToDepth(tensor_t input, tensor_t output, int64_t blocksize) override {
    return impl()->onnx_SpaceToDepth(input, output, blocksize);
}

OperatorReturnType onnx_Split(tensor_t input, std::variant<void *, tensor_t>& split, std::vector<tensor_t>& outputs, int64_t axis) override {
    return impl()->onnx_Split(input, split, outputs, axis);
}

OperatorReturnType onnx_SplitToSequence(tensor_t input, std::variant<void *, tensor_t>& split, tensor_t output_sequence, int64_t axis, int64_t keepdims) override {
    return impl()->onnx_SplitToSequence(input, split, output_sequence, axis, keepdims);
}

OperatorReturnType onnx_Sqrt(tensor_t X, tensor_t Y) override {
    return impl()->onnx_Sqrt(X, Y);
}

OperatorReturnType onnx_Squeeze(tensor_t data, std::variant<void *, tensor_t>& axes, tensor_t squeezed) override {
    return impl()->onnx_Squeeze(data, axes, squeezed);
}

OperatorReturnType onnx_StringNormalizer(tensor_t X, tensor_t Y, std::string case_change_action, int64_t is_case_sensitive, std::string locale, std::vector<std::string> stopwords) override {
    return impl()->onnx_StringNormalizer(X, Y, case_change_action, is_case_sensitive, locale, stopwords);
}

OperatorReturnType onnx_Sub(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_Sub(A, B, C);
}

OperatorReturnType onnx_Sum(std::vector<tensor_t>& data_0, tensor_t sum) override {
    return impl()->onnx_Sum(data_0, sum);
}

OperatorReturnType onnx_Tan(tensor_t input, tensor_t output) override {
    return impl()->onnx_Tan(input, output);
}

OperatorReturnType onnx_Tanh(tensor_t input, tensor_t output) override {
    return impl()->onnx_Tanh(input, output);
}

OperatorReturnType onnx_TfIdfVectorizer(tensor_t X, tensor_t Y, int64_t max_gram_length, int64_t max_skip_count, int64_t min_gram_length, std::string mode, std::vector<int64_t> ngram_counts, std::vector<int64_t> ngram_indexes, std::vector<int64_t> pool_int64s, std::vector<std::string> pool_strings, std::vector<float> weights) override {
    return impl()->onnx_TfIdfVectorizer(X, Y, max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings, weights);
}

OperatorReturnType onnx_ThresholdedRelu(tensor_t X, tensor_t Y, float alpha) override {
    return impl()->onnx_ThresholdedRelu(X, Y, alpha);
}

OperatorReturnType onnx_Tile(tensor_t input, tensor_t repeats, tensor_t output) override {
    return impl()->onnx_Tile(input, repeats, output);
}

OperatorReturnType onnx_TopK(tensor_t X, tensor_t K, tensor_t Values, tensor_t Indices, int64_t axis, int64_t largest, int64_t sorted) override {
    return impl()->onnx_TopK(X, K, Values, Indices, axis, largest, sorted);
}

OperatorReturnType onnx_Transpose(tensor_t data, tensor_t transposed, std::vector<int64_t> perm) override {
    return impl()->onnx_Transpose(data, transposed, perm);
}

OperatorReturnType onnx_Trilu(tensor_t input, std::variant<void *, tensor_t>& k, tensor_t output, int64_t upper) override {
    return impl()->onnx_Trilu(input, k, output, upper);
}

OperatorReturnType onnx_Unique(tensor_t X, tensor_t Y, std::variant<void *, tensor_t>& indices, std::variant<void *, tensor_t>& inverse_indices, std::variant<void *, tensor_t>& counts, int64_t axis, int64_t sorted) override {
    return impl()->onnx_Unique(X, Y, indices, inverse_indices, counts, axis, sorted);
}

OperatorReturnType onnx_Unsqueeze(tensor_t data, tensor_t axes, tensor_t expanded) override {
    return impl()->onnx_Unsqueeze(data, axes, expanded);
}

OperatorReturnType onnx_Where(tensor_t condition, tensor_t X, tensor_t Y, tensor_t output) override {
    return impl()->onnx_Where(condition, X, Y, output);
}

OperatorReturnType onnx_Xor(tensor_t A, tensor_t B, tensor_t C) override {
    return impl()->onnx_Xor(A, B, C);
}

