//
//  this file is geneated by autogen
//

#ifndef _YANNX_TENSORAPI_HPP_
#define _YANNX_TENSORAPI_HPP_

#include <vector>
#include <variant>

namespace yannx { namespace tt {

struct TensorType;
using tensor_t = std::shared_ptr<TensorType>;

enum OperatorReturnType {
    YNX_OK = 0,
    YNX_TODO_ERROR = -1,
    YNX_INPUT_ERROR = -2,
    YNX_OUTPUT_ERROR = -3,
    YNX_ATTR_ERROR = -4,
};

//
//  ONNX based tensor computing API
//  https://github.com/onnx/onnx/blob/main/docs/IR.md
//  https://github.com/onnx/onnx/blob/main/docs/Operators.md
//
struct OnnxOperatorSet {
    // some must be common operator
    virtual const char* genre() = 0;

    // following is ONNX operator set
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Abs
    virtual OperatorReturnType onnx_Abs(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Acos
    virtual OperatorReturnType onnx_Acos(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Acosh
    virtual OperatorReturnType onnx_Acosh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
    virtual OperatorReturnType onnx_Add(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#And
    virtual OperatorReturnType onnx_And(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMax
    virtual OperatorReturnType onnx_ArgMax(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ int64_t axis, int64_t keepdims, int64_t select_last_index) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMin
    virtual OperatorReturnType onnx_ArgMin(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ int64_t axis, int64_t keepdims, int64_t select_last_index) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asin
    virtual OperatorReturnType onnx_Asin(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asinh
    virtual OperatorReturnType onnx_Asinh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Atan
    virtual OperatorReturnType onnx_Atan(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Atanh
    virtual OperatorReturnType onnx_Atanh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool
    virtual OperatorReturnType onnx_AveragePool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::string auto_pad, int64_t ceil_mode, int64_t count_include_pad, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization
    virtual OperatorReturnType onnx_BatchNormalization(/*inputs:*/ tensor_t X, tensor_t scale, tensor_t B, tensor_t input_mean, tensor_t input_var, /*outputs:*/ tensor_t Y, std::variant<void *, tensor_t>& running_mean, std::variant<void *, tensor_t>& running_var, /*attributes:*/ float epsilon, float momentum, int64_t training_mode) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Bernoulli
    virtual OperatorReturnType onnx_Bernoulli(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t dtype, float seed) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitShift
    virtual OperatorReturnType onnx_BitShift(/*inputs:*/ tensor_t X, tensor_t Y, /*outputs:*/ tensor_t Z, /*attributes:*/ std::string direction) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
    virtual OperatorReturnType onnx_Cast(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t to) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#CastLike
    virtual OperatorReturnType onnx_CastLike(/*inputs:*/ tensor_t input, tensor_t target_type, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Ceil
    virtual OperatorReturnType onnx_Ceil(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Celu
    virtual OperatorReturnType onnx_Celu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ float alpha) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip
    virtual OperatorReturnType onnx_Clip(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& min, std::variant<void *, tensor_t>& max, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Compress
    virtual OperatorReturnType onnx_Compress(/*inputs:*/ tensor_t input, tensor_t condition, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
    virtual OperatorReturnType onnx_Concat(/*inputs:*/ std::vector<tensor_t>& inputs, /*outputs:*/ tensor_t concat_result, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConcatFromSequence
    virtual OperatorReturnType onnx_ConcatFromSequence(/*inputs:*/ tensor_t input_sequence, /*outputs:*/ tensor_t concat_result, /*attributes:*/ int64_t axis, int64_t new_axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
    virtual OperatorReturnType onnx_Conv(/*inputs:*/ tensor_t X, tensor_t W, std::variant<void *, tensor_t>& B, /*outputs:*/ tensor_t Y, /*attributes:*/ std::string auto_pad, std::vector<int64_t> dilations, int64_t group, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvInteger
    virtual OperatorReturnType onnx_ConvInteger(/*inputs:*/ tensor_t x, tensor_t w, std::variant<void *, tensor_t>& x_zero_point, std::variant<void *, tensor_t>& w_zero_point, /*outputs:*/ tensor_t y, /*attributes:*/ std::string auto_pad, std::vector<int64_t> dilations, int64_t group, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
    virtual OperatorReturnType onnx_ConvTranspose(/*inputs:*/ tensor_t X, tensor_t W, std::variant<void *, tensor_t>& B, /*outputs:*/ tensor_t Y, /*attributes:*/ std::string auto_pad, std::vector<int64_t> dilations, int64_t group, std::vector<int64_t> kernel_shape, std::vector<int64_t> output_padding, std::vector<int64_t> output_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cos
    virtual OperatorReturnType onnx_Cos(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cosh
    virtual OperatorReturnType onnx_Cosh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
    virtual OperatorReturnType onnx_CumSum(/*inputs:*/ tensor_t x, tensor_t axis, /*outputs:*/ tensor_t y, /*attributes:*/ int64_t exclusive, int64_t reverse) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#DepthToSpace
    virtual OperatorReturnType onnx_DepthToSpace(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t blocksize, std::string mode) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#DequantizeLinear
    virtual OperatorReturnType onnx_DequantizeLinear(/*inputs:*/ tensor_t x, tensor_t x_scale, std::variant<void *, tensor_t>& x_zero_point, /*outputs:*/ tensor_t y, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Det
    virtual OperatorReturnType onnx_Det(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div
    virtual OperatorReturnType onnx_Div(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout
    virtual OperatorReturnType onnx_Dropout(/*inputs:*/ tensor_t data, std::variant<void *, tensor_t>& ratio, std::variant<void *, tensor_t>& training_mode, /*outputs:*/ tensor_t output, std::variant<void *, tensor_t>& mask, /*attributes:*/ int64_t seed) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#DynamicQuantizeLinear
    virtual OperatorReturnType onnx_DynamicQuantizeLinear(/*inputs:*/ tensor_t x, /*outputs:*/ tensor_t y, tensor_t y_scale, tensor_t y_zero_point) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Einsum
    virtual OperatorReturnType onnx_Einsum(/*inputs:*/ std::vector<tensor_t>& Inputs, /*outputs:*/ tensor_t Output, /*attributes:*/ std::string equation) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Elu
    virtual OperatorReturnType onnx_Elu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ float alpha) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Equal
    virtual OperatorReturnType onnx_Equal(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Erf
    virtual OperatorReturnType onnx_Erf(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Exp
    virtual OperatorReturnType onnx_Exp(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand
    virtual OperatorReturnType onnx_Expand(/*inputs:*/ tensor_t input, tensor_t shape, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#EyeLike
    virtual OperatorReturnType onnx_EyeLike(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t dtype, int64_t k) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten
    virtual OperatorReturnType onnx_Flatten(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Floor
    virtual OperatorReturnType onnx_Floor(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU
    virtual OperatorReturnType onnx_GRU(/*inputs:*/ tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, /*outputs:*/ std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, /*attributes:*/ std::vector<float> activation_alpha, std::vector<float> activation_beta, std::vector<std::string> activations, float clip, std::string direction, int64_t hidden_size, int64_t layout, int64_t linear_before_reset) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
    virtual OperatorReturnType onnx_Gather(/*inputs:*/ tensor_t data, tensor_t indices, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements
    virtual OperatorReturnType onnx_GatherElements(/*inputs:*/ tensor_t data, tensor_t indices, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
    virtual OperatorReturnType onnx_GatherND(/*inputs:*/ tensor_t data, tensor_t indices, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t batch_dims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
    virtual OperatorReturnType onnx_Gemm(/*inputs:*/ tensor_t A, tensor_t B, std::variant<void *, tensor_t>& C, /*outputs:*/ tensor_t Y, /*attributes:*/ float alpha, float beta, int64_t transA, int64_t transB) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool
    virtual OperatorReturnType onnx_GlobalAveragePool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalLpPool
    virtual OperatorReturnType onnx_GlobalLpPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ int64_t p) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool
    virtual OperatorReturnType onnx_GlobalMaxPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater
    virtual OperatorReturnType onnx_Greater(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#GreaterOrEqual
    virtual OperatorReturnType onnx_GreaterOrEqual(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid
    virtual OperatorReturnType onnx_HardSigmoid(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ float alpha, float beta) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSwish
    virtual OperatorReturnType onnx_HardSwish(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Hardmax
    virtual OperatorReturnType onnx_Hardmax(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity
    virtual OperatorReturnType onnx_Identity(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization
    virtual OperatorReturnType onnx_InstanceNormalization(/*inputs:*/ tensor_t input, tensor_t scale, tensor_t B, /*outputs:*/ tensor_t output, /*attributes:*/ float epsilon) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsInf
    virtual OperatorReturnType onnx_IsInf(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ int64_t detect_negative, int64_t detect_positive) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsNaN
    virtual OperatorReturnType onnx_IsNaN(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN
    virtual OperatorReturnType onnx_LRN(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ float alpha, float beta, float bias, int64_t size) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM
    virtual OperatorReturnType onnx_LSTM(/*inputs:*/ tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, std::variant<void *, tensor_t>& initial_c, std::variant<void *, tensor_t>& P, /*outputs:*/ std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, std::variant<void *, tensor_t>& Y_c, /*attributes:*/ std::vector<float> activation_alpha, std::vector<float> activation_beta, std::vector<std::string> activations, float clip, std::string direction, int64_t hidden_size, int64_t input_forget, int64_t layout) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LeakyRelu
    virtual OperatorReturnType onnx_LeakyRelu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ float alpha) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less
    virtual OperatorReturnType onnx_Less(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LessOrEqual
    virtual OperatorReturnType onnx_LessOrEqual(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log
    virtual OperatorReturnType onnx_Log(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LogSoftmax
    virtual OperatorReturnType onnx_LogSoftmax(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpNormalization
    virtual OperatorReturnType onnx_LpNormalization(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t axis, int64_t p) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpPool
    virtual OperatorReturnType onnx_LpPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::string auto_pad, std::vector<int64_t> kernel_shape, int64_t p, std::vector<int64_t> pads, std::vector<int64_t> strides) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul
    virtual OperatorReturnType onnx_MatMul(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMulInteger
    virtual OperatorReturnType onnx_MatMulInteger(/*inputs:*/ tensor_t A, tensor_t B, std::variant<void *, tensor_t>& a_zero_point, std::variant<void *, tensor_t>& b_zero_point, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Max
    virtual OperatorReturnType onnx_Max(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t max) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
    virtual OperatorReturnType onnx_MaxPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, std::variant<void *, tensor_t>& Indices, /*attributes:*/ std::string auto_pad, int64_t ceil_mode, std::vector<int64_t> dilations, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, int64_t storage_order, std::vector<int64_t> strides) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxRoiPool
    virtual OperatorReturnType onnx_MaxRoiPool(/*inputs:*/ tensor_t X, tensor_t rois, /*outputs:*/ tensor_t Y, /*attributes:*/ std::vector<int64_t> pooled_shape, float spatial_scale) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxUnpool
    virtual OperatorReturnType onnx_MaxUnpool(/*inputs:*/ tensor_t X, tensor_t I, std::variant<void *, tensor_t>& output_shape, /*outputs:*/ tensor_t output, /*attributes:*/ std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mean
    virtual OperatorReturnType onnx_Mean(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t mean) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MeanVarianceNormalization
    virtual OperatorReturnType onnx_MeanVarianceNormalization(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::vector<int64_t> axes) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min
    virtual OperatorReturnType onnx_Min(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t min) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mod
    virtual OperatorReturnType onnx_Mod(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C, /*attributes:*/ int64_t fmod) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
    virtual OperatorReturnType onnx_Mul(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Multinomial
    virtual OperatorReturnType onnx_Multinomial(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t dtype, int64_t sample_size, float seed) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Neg
    virtual OperatorReturnType onnx_Neg(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#NegativeLogLikelihoodLoss
    virtual OperatorReturnType onnx_NegativeLogLikelihoodLoss(/*inputs:*/ tensor_t input, tensor_t target, std::variant<void *, tensor_t>& weight, /*outputs:*/ tensor_t loss, /*attributes:*/ int64_t ignore_index, std::string reduction) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression
    virtual OperatorReturnType onnx_NonMaxSuppression(/*inputs:*/ tensor_t boxes, tensor_t scores, std::variant<void *, tensor_t>& max_output_boxes_per_class, std::variant<void *, tensor_t>& iou_threshold, std::variant<void *, tensor_t>& score_threshold, /*outputs:*/ tensor_t selected_indices, /*attributes:*/ int64_t center_point_box) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonZero
    virtual OperatorReturnType onnx_NonZero(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Not
    virtual OperatorReturnType onnx_Not(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#OneHot
    virtual OperatorReturnType onnx_OneHot(/*inputs:*/ tensor_t indices, tensor_t depth, tensor_t values, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Or
    virtual OperatorReturnType onnx_Or(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu
    virtual OperatorReturnType onnx_PRelu(/*inputs:*/ tensor_t X, tensor_t slope, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad
    virtual OperatorReturnType onnx_Pad(/*inputs:*/ tensor_t data, tensor_t pads, std::variant<void *, tensor_t>& constant_value, /*outputs:*/ tensor_t output, /*attributes:*/ std::string mode) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow
    virtual OperatorReturnType onnx_Pow(/*inputs:*/ tensor_t X, tensor_t Y, /*outputs:*/ tensor_t Z) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv
    virtual OperatorReturnType onnx_QLinearConv(/*inputs:*/ tensor_t x, tensor_t x_scale, tensor_t x_zero_point, tensor_t w, tensor_t w_scale, tensor_t w_zero_point, tensor_t y_scale, tensor_t y_zero_point, std::variant<void *, tensor_t>& B, /*outputs:*/ tensor_t y, /*attributes:*/ std::string auto_pad, std::vector<int64_t> dilations, int64_t group, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearMatMul
    virtual OperatorReturnType onnx_QLinearMatMul(/*inputs:*/ tensor_t a, tensor_t a_scale, tensor_t a_zero_point, tensor_t b, tensor_t b_scale, tensor_t b_zero_point, tensor_t y_scale, tensor_t y_zero_point, /*outputs:*/ tensor_t y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear
    virtual OperatorReturnType onnx_QuantizeLinear(/*inputs:*/ tensor_t x, tensor_t y_scale, std::variant<void *, tensor_t>& y_zero_point, /*outputs:*/ tensor_t y, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN
    virtual OperatorReturnType onnx_RNN(/*inputs:*/ tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, /*outputs:*/ std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, /*attributes:*/ std::vector<float> activation_alpha, std::vector<float> activation_beta, std::vector<std::string> activations, float clip, std::string direction, int64_t hidden_size, int64_t layout) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormal
    virtual OperatorReturnType onnx_RandomNormal(/*outputs:*/ tensor_t output, /*attributes:*/ int64_t dtype, float mean, float scale, float seed, std::vector<int64_t> shape) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormalLike
    virtual OperatorReturnType onnx_RandomNormalLike(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t dtype, float mean, float scale, float seed) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniform
    virtual OperatorReturnType onnx_RandomUniform(/*outputs:*/ tensor_t output, /*attributes:*/ int64_t dtype, float high, float low, float seed, std::vector<int64_t> shape) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniformLike
    virtual OperatorReturnType onnx_RandomUniformLike(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t dtype, float high, float low, float seed) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range
    virtual OperatorReturnType onnx_Range(/*inputs:*/ tensor_t start, tensor_t limit, tensor_t delta, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reciprocal
    virtual OperatorReturnType onnx_Reciprocal(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL1
    virtual OperatorReturnType onnx_ReduceL1(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::vector<int64_t> axes, int64_t keepdims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2
    virtual OperatorReturnType onnx_ReduceL2(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::vector<int64_t> axes, int64_t keepdims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSum
    virtual OperatorReturnType onnx_ReduceLogSum(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::vector<int64_t> axes, int64_t keepdims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSumExp
    virtual OperatorReturnType onnx_ReduceLogSumExp(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::vector<int64_t> axes, int64_t keepdims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMax
    virtual OperatorReturnType onnx_ReduceMax(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::vector<int64_t> axes, int64_t keepdims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
    virtual OperatorReturnType onnx_ReduceMean(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::vector<int64_t> axes, int64_t keepdims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMin
    virtual OperatorReturnType onnx_ReduceMin(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::vector<int64_t> axes, int64_t keepdims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceProd
    virtual OperatorReturnType onnx_ReduceProd(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::vector<int64_t> axes, int64_t keepdims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum
    virtual OperatorReturnType onnx_ReduceSum(/*inputs:*/ tensor_t data, std::variant<void *, tensor_t>& axes, /*outputs:*/ tensor_t reduced, /*attributes:*/ int64_t keepdims, int64_t noop_with_empty_axes) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSumSquare
    virtual OperatorReturnType onnx_ReduceSumSquare(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::vector<int64_t> axes, int64_t keepdims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
    virtual OperatorReturnType onnx_Relu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape
    virtual OperatorReturnType onnx_Reshape(/*inputs:*/ tensor_t data, tensor_t shape, /*outputs:*/ tensor_t reshaped, /*attributes:*/ int64_t allowzero) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
    virtual OperatorReturnType onnx_Resize(/*inputs:*/ tensor_t X, std::variant<void *, tensor_t>& roi, std::variant<void *, tensor_t>& scales, std::variant<void *, tensor_t>& sizes, /*outputs:*/ tensor_t Y, /*attributes:*/ std::string coordinate_transformation_mode, float cubic_coeff_a, int64_t exclude_outside, float extrapolation_value, std::string mode, std::string nearest_mode) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReverseSequence
    virtual OperatorReturnType onnx_ReverseSequence(/*inputs:*/ tensor_t input, tensor_t sequence_lens, /*outputs:*/ tensor_t Y, /*attributes:*/ int64_t batch_axis, int64_t time_axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#RoiAlign
    virtual OperatorReturnType onnx_RoiAlign(/*inputs:*/ tensor_t X, tensor_t rois, tensor_t batch_indices, /*outputs:*/ tensor_t Y, /*attributes:*/ std::string coordinate_transformation_mode, std::string mode, int64_t output_height, int64_t output_width, int64_t sampling_ratio, float spatial_scale) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Round
    virtual OperatorReturnType onnx_Round(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements
    virtual OperatorReturnType onnx_ScatterElements(/*inputs:*/ tensor_t data, tensor_t indices, tensor_t updates, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND
    virtual OperatorReturnType onnx_ScatterND(/*inputs:*/ tensor_t data, tensor_t indices, tensor_t updates, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Selu
    virtual OperatorReturnType onnx_Selu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ float alpha, float gamma) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceAt
    virtual OperatorReturnType onnx_SequenceAt(/*inputs:*/ tensor_t input_sequence, tensor_t position, /*outputs:*/ tensor_t tensor) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceConstruct
    virtual OperatorReturnType onnx_SequenceConstruct(/*inputs:*/ std::vector<tensor_t>& inputs, /*outputs:*/ tensor_t output_sequence) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceEmpty
    virtual OperatorReturnType onnx_SequenceEmpty(/*outputs:*/ tensor_t output, /*attributes:*/ int64_t dtype) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceErase
    virtual OperatorReturnType onnx_SequenceErase(/*inputs:*/ tensor_t input_sequence, std::variant<void *, tensor_t>& position, /*outputs:*/ tensor_t output_sequence) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceInsert
    virtual OperatorReturnType onnx_SequenceInsert(/*inputs:*/ tensor_t input_sequence, tensor_t tensor, std::variant<void *, tensor_t>& position, /*outputs:*/ tensor_t output_sequence) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceLength
    virtual OperatorReturnType onnx_SequenceLength(/*inputs:*/ tensor_t input_sequence, /*outputs:*/ tensor_t length) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape
    virtual OperatorReturnType onnx_Shape(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t shape, /*attributes:*/ int64_t end, int64_t start) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shrink
    virtual OperatorReturnType onnx_Shrink(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ float bias, float lambd) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid
    virtual OperatorReturnType onnx_Sigmoid(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sign
    virtual OperatorReturnType onnx_Sign(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sin
    virtual OperatorReturnType onnx_Sin(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sinh
    virtual OperatorReturnType onnx_Sinh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size
    virtual OperatorReturnType onnx_Size(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t size) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice
    virtual OperatorReturnType onnx_Slice(/*inputs:*/ tensor_t data, tensor_t starts, tensor_t ends, std::variant<void *, tensor_t>& axes, std::variant<void *, tensor_t>& steps, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
    virtual OperatorReturnType onnx_Softmax(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#SoftmaxCrossEntropyLoss
    virtual OperatorReturnType onnx_SoftmaxCrossEntropyLoss(/*inputs:*/ tensor_t scores, tensor_t labels, std::variant<void *, tensor_t>& weights, /*outputs:*/ tensor_t output, std::variant<void *, tensor_t>& log_prob, /*attributes:*/ int64_t ignore_index, std::string reduction) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softplus
    virtual OperatorReturnType onnx_Softplus(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softsign
    virtual OperatorReturnType onnx_Softsign(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#SpaceToDepth
    virtual OperatorReturnType onnx_SpaceToDepth(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t blocksize) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
    virtual OperatorReturnType onnx_Split(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& split, /*outputs:*/ std::vector<tensor_t>& outputs, /*attributes:*/ int64_t axis) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#SplitToSequence
    virtual OperatorReturnType onnx_SplitToSequence(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& split, /*outputs:*/ tensor_t output_sequence, /*attributes:*/ int64_t axis, int64_t keepdims) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt
    virtual OperatorReturnType onnx_Sqrt(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Squeeze
    virtual OperatorReturnType onnx_Squeeze(/*inputs:*/ tensor_t data, std::variant<void *, tensor_t>& axes, /*outputs:*/ tensor_t squeezed) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#StringNormalizer
    virtual OperatorReturnType onnx_StringNormalizer(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::string case_change_action, int64_t is_case_sensitive, std::string locale, std::vector<std::string> stopwords) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub
    virtual OperatorReturnType onnx_Sub(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sum
    virtual OperatorReturnType onnx_Sum(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t sum) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tan
    virtual OperatorReturnType onnx_Tan(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh
    virtual OperatorReturnType onnx_Tanh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#TfIdfVectorizer
    virtual OperatorReturnType onnx_TfIdfVectorizer(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ int64_t max_gram_length, int64_t max_skip_count, int64_t min_gram_length, std::string mode, std::vector<int64_t> ngram_counts, std::vector<int64_t> ngram_indexes, std::vector<int64_t> pool_int64s, std::vector<std::string> pool_strings, std::vector<float> weights) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ThresholdedRelu
    virtual OperatorReturnType onnx_ThresholdedRelu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ float alpha) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile
    virtual OperatorReturnType onnx_Tile(/*inputs:*/ tensor_t input, tensor_t repeats, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK
    virtual OperatorReturnType onnx_TopK(/*inputs:*/ tensor_t X, tensor_t K, /*outputs:*/ tensor_t Values, tensor_t Indices, /*attributes:*/ int64_t axis, int64_t largest, int64_t sorted) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose
    virtual OperatorReturnType onnx_Transpose(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t transposed, /*attributes:*/ std::vector<int64_t> perm) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Trilu
    virtual OperatorReturnType onnx_Trilu(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& k, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t upper) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unique
    virtual OperatorReturnType onnx_Unique(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, std::variant<void *, tensor_t>& indices, std::variant<void *, tensor_t>& inverse_indices, std::variant<void *, tensor_t>& counts, /*attributes:*/ int64_t axis, int64_t sorted) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze
    virtual OperatorReturnType onnx_Unsqueeze(/*inputs:*/ tensor_t data, tensor_t axes, /*outputs:*/ tensor_t expanded) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where
    virtual OperatorReturnType onnx_Where(/*inputs:*/ tensor_t condition, tensor_t X, tensor_t Y, /*outputs:*/ tensor_t output) {
        return YNX_TODO_ERROR;
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor
    virtual OperatorReturnType onnx_Xor(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
        return YNX_TODO_ERROR;
    }



};

}}

#endif
