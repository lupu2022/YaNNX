//
//  this file is geneated by onnx~/autogen
//

#ifndef _YANXX_TENSORTYPE_HPP_
#define _YANNX_TENSORTYPE_HPP_

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#include <yannx.hpp>

#ifdef USING_ONNX_IMPL
#include <onnx/onnx_pb.h>
#include <onnx/defs/schema.h>
#include <onnx/defs/attr_proto_util.h>
#include <onnx/shape_inference/implementation.h>

using namespace onnx;
#endif


//
//  A simple onnx based (type and shape inference only, or a pure dummy tensor ) Tensor.following ONNX IR
//  https://github.com/onnx/onnx/blob/main/docs/IR.md
//  https://github.com/onnx/onnx/blob/main/docs/Operators.md
//

using namespace yannx;

namespace yannx_tt {

enum OperatorReturnType {
    YNX_OK = 0,
    YNX_TODO_ERROR = -1,
    YNX_INPUT_ERROR = -2,
    YNX_OUTPUT_ERROR = -3,
    YNX_ATTR_ERROR = -4,
};

enum TensorDataType {
    YNX_UNDEFINED = 0,
    YNX_FLOAT,
    YNX_UINT8,
    YNX_INT8,
    YNX_UINT16,
    YNX_INT16,
    YNX_INT32,
    YNX_INT64,
    YNX_STRING,
    YNX_BOOL,
    YNX_FLOAT16,
    YNX_DOUBLE,
    YNX_UINT32,
    YNX_UINT64,
    YNX_COMPLEX64,
    YNX_COMPLEX128,
    YNX_BFLOAT16
};

static const char* TensorDataTypeString[] = {
    "YNX_UNDEFINED",
    "YNX_FLOAT",
    "YNX_UINT8",
    "YNX_INT8",
    "YNX_UINT16",
    "YNX_INT16",
    "YNX_INT32",
    "YNX_INT64",
    "YNX_STRING",
    "YNX_BOOL",
    "YNX_FLOAT16",
    "YNX_DOUBLE",
    "YNX_UINT32",
    "YNX_UINT64",
    "YNX_COMPLEX64",
    "YNX_COMPLEX128",
    "YNX_BFLOAT16"
};

TensorDataType datatype_from_string(const std::string& dt_str ) {
    for (size_t i = 0; i < YNX_BFLOAT16; i++) {
        if ( dt_str == TensorDataTypeString[i] ) {
            return (TensorDataType)i;
        }
    }
    return YNX_UNDEFINED;
};


#ifdef USING_ONNX_IMPL
TensorDataType datatype_from_onnx( int dt ) {
    switch( dt ) {
        case TensorProto_DataType_UNDEFINED:
            return YNX_UNDEFINED;
        case TensorProto_DataType_FLOAT:
            return YNX_FLOAT;
        case TensorProto_DataType_UINT8:
            return YNX_UINT8;
        case TensorProto_DataType_UINT16:
            return YNX_UINT16;
        case TensorProto_DataType_INT16:
            return YNX_INT16;
        case TensorProto_DataType_INT32:
            return YNX_INT32;
        case TensorProto_DataType_INT64:
            return YNX_INT64;
        case TensorProto_DataType_STRING:
            return YNX_STRING;
        case TensorProto_DataType_BOOL:
            return YNX_BOOL;
        case TensorProto_DataType_FLOAT16:
            return YNX_FLOAT16;
        case TensorProto_DataType_DOUBLE:
            return YNX_DOUBLE;
        case TensorProto_DataType_UINT32:
            return YNX_UINT32;
        case TensorProto_DataType_UINT64:
            return YNX_UINT64;
        case TensorProto_DataType_COMPLEX64:
            return YNX_COMPLEX64;
        case TensorProto_DataType_COMPLEX128:
            return YNX_COMPLEX128;
        case TensorProto_DataType_BFLOAT16:
            return YNX_BFLOAT16;
    }
    return YNX_UNDEFINED;
}
#endif


/*
 *  https://github.com/onnx/onnx/blob/main/docs/IR.md#tensor-definition
 *  scalar:         an empty shape with a defined data type
 *  tensor:         shape dimention > 0
 *  undefined:      empty shape with a undefined data type, used for type_shape inference.
 */

struct TensorType;
using tensor_t = std::shared_ptr<TensorType>;

struct TensorType {
    TensorDataType dtype_;
    std::vector<size_t> shape_;

    TensorType() {
        dtype_ = TensorDataType::YNX_UNDEFINED;
    }
    TensorType(TensorDataType dtype, std::vector<size_t>& shape) {
        reset(dtype, shape);
    }
    virtual void reset(TensorDataType dtype, std::vector<size_t>& shape) {
        dtype_ = dtype;
        shape_ = shape;
    }
    virtual std::string to_string() {
        std::ostringstream ss;
        ss << TensorDataTypeString[dtype_];
        ss << ":[";
        for (size_t i = 0; i < shape_.size(); i++) {
            ss << shape_[i];
            if (i != shape_.size() - 1) {
                ss << " ";
            }
        }
        ss << "]";
        return ss.str();
    }

    //
    //  User must be re-implement, return user side undefined tensor!
    //
    enum TensorFlag {
        TT_Constant,
        TT_Variable,
        TT_Parameter,
    };
    static tensor_t create_undefined_user_tensor();
    static void register_user_tensor(tensor_t, TensorFlag flag = TT_Constant);

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Abs
	virtual OperatorReturnType onnx_Abs(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Acos
	virtual OperatorReturnType onnx_Acos(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Acosh
	virtual OperatorReturnType onnx_Acosh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
	virtual OperatorReturnType onnx_Add(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#And
	virtual OperatorReturnType onnx_And(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMax
	virtual OperatorReturnType onnx_ArgMax(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > keepdims, std::variant<void *, int64_t > select_last_index) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMin
	virtual OperatorReturnType onnx_ArgMin(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > keepdims, std::variant<void *, int64_t > select_last_index) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asin
	virtual OperatorReturnType onnx_Asin(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asinh
	virtual OperatorReturnType onnx_Asinh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Atan
	virtual OperatorReturnType onnx_Atan(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Atanh
	virtual OperatorReturnType onnx_Atanh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool
	virtual OperatorReturnType onnx_AveragePool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, int64_t > ceil_mode, std::variant<void *, int64_t > count_include_pad, std::vector<int64_t> kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization
	virtual OperatorReturnType onnx_BatchNormalization(/*inputs:*/ tensor_t X, tensor_t scale, tensor_t B, tensor_t input_mean, tensor_t input_var, /*outputs:*/ tensor_t Y, std::variant<void *, tensor_t>& running_mean, std::variant<void *, tensor_t>& running_var, /*attributes:*/ std::variant<void *, float > epsilon, std::variant<void *, float > momentum, std::variant<void *, int64_t > training_mode) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Bernoulli
	virtual OperatorReturnType onnx_Bernoulli(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, float > seed) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitShift
	virtual OperatorReturnType onnx_BitShift(/*inputs:*/ tensor_t X, tensor_t Y, /*outputs:*/ tensor_t Z, /*attributes:*/ std::string direction) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
	virtual OperatorReturnType onnx_Cast(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t to) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#CastLike
	virtual OperatorReturnType onnx_CastLike(/*inputs:*/ tensor_t input, tensor_t target_type, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Ceil
	virtual OperatorReturnType onnx_Ceil(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Celu
	virtual OperatorReturnType onnx_Celu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip
	virtual OperatorReturnType onnx_Clip(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& min, std::variant<void *, tensor_t>& max, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Compress
	virtual OperatorReturnType onnx_Compress(/*inputs:*/ tensor_t input, tensor_t condition, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
	virtual OperatorReturnType onnx_Concat(/*inputs:*/ std::vector<tensor_t>& inputs, /*outputs:*/ tensor_t concat_result, /*attributes:*/ int64_t axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConcatFromSequence
	virtual OperatorReturnType onnx_ConcatFromSequence(/*inputs:*/ tensor_t input_sequence, /*outputs:*/ tensor_t concat_result, /*attributes:*/ int64_t axis, std::variant<void *, int64_t > new_axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
	virtual OperatorReturnType onnx_Conv(/*inputs:*/ tensor_t X, tensor_t W, std::variant<void *, tensor_t>& B, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, std::vector<int64_t> > dilations, std::variant<void *, int64_t > group, std::variant<void *, std::vector<int64_t> > kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvInteger
	virtual OperatorReturnType onnx_ConvInteger(/*inputs:*/ tensor_t x, tensor_t w, std::variant<void *, tensor_t>& x_zero_point, std::variant<void *, tensor_t>& w_zero_point, /*outputs:*/ tensor_t y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, std::vector<int64_t> > dilations, std::variant<void *, int64_t > group, std::variant<void *, std::vector<int64_t> > kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
	virtual OperatorReturnType onnx_ConvTranspose(/*inputs:*/ tensor_t X, tensor_t W, std::variant<void *, tensor_t>& B, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, std::vector<int64_t> > dilations, std::variant<void *, int64_t > group, std::variant<void *, std::vector<int64_t> > kernel_shape, std::variant<void *, std::vector<int64_t> > output_padding, std::variant<void *, std::vector<int64_t> > output_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cos
	virtual OperatorReturnType onnx_Cos(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cosh
	virtual OperatorReturnType onnx_Cosh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
	virtual OperatorReturnType onnx_CumSum(/*inputs:*/ tensor_t x, tensor_t axis, /*outputs:*/ tensor_t y, /*attributes:*/ std::variant<void *, int64_t > exclusive, std::variant<void *, int64_t > reverse) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#DepthToSpace
	virtual OperatorReturnType onnx_DepthToSpace(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t blocksize, std::variant<void *, std::string > mode) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#DequantizeLinear
	virtual OperatorReturnType onnx_DequantizeLinear(/*inputs:*/ tensor_t x, tensor_t x_scale, std::variant<void *, tensor_t>& x_zero_point, /*outputs:*/ tensor_t y, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Det
	virtual OperatorReturnType onnx_Det(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div
	virtual OperatorReturnType onnx_Div(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout
	virtual OperatorReturnType onnx_Dropout(/*inputs:*/ tensor_t data, std::variant<void *, tensor_t>& ratio, std::variant<void *, tensor_t>& training_mode, /*outputs:*/ tensor_t output, std::variant<void *, tensor_t>& mask, /*attributes:*/ std::variant<void *, int64_t > seed) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#DynamicQuantizeLinear
	virtual OperatorReturnType onnx_DynamicQuantizeLinear(/*inputs:*/ tensor_t x, /*outputs:*/ tensor_t y, tensor_t y_scale, tensor_t y_zero_point) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Einsum
	virtual OperatorReturnType onnx_Einsum(/*inputs:*/ std::vector<tensor_t>& Inputs, /*outputs:*/ tensor_t Output, /*attributes:*/ std::string equation) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Elu
	virtual OperatorReturnType onnx_Elu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Equal
	virtual OperatorReturnType onnx_Equal(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Erf
	virtual OperatorReturnType onnx_Erf(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Exp
	virtual OperatorReturnType onnx_Exp(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand
	virtual OperatorReturnType onnx_Expand(/*inputs:*/ tensor_t input, tensor_t shape, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#EyeLike
	virtual OperatorReturnType onnx_EyeLike(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, int64_t > k) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten
	virtual OperatorReturnType onnx_Flatten(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Floor
	virtual OperatorReturnType onnx_Floor(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU
	virtual OperatorReturnType onnx_GRU(/*inputs:*/ tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, /*outputs:*/ std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, /*attributes:*/ std::variant<void *, std::vector<float> > activation_alpha, std::variant<void *, std::vector<float> > activation_beta, std::variant<void *, std::vector<std::string> > activations, std::variant<void *, float > clip, std::variant<void *, std::string > direction, std::variant<void *, int64_t > hidden_size, std::variant<void *, int64_t > layout, std::variant<void *, int64_t > linear_before_reset) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
	virtual OperatorReturnType onnx_Gather(/*inputs:*/ tensor_t data, tensor_t indices, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements
	virtual OperatorReturnType onnx_GatherElements(/*inputs:*/ tensor_t data, tensor_t indices, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
	virtual OperatorReturnType onnx_GatherND(/*inputs:*/ tensor_t data, tensor_t indices, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > batch_dims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
	virtual OperatorReturnType onnx_Gemm(/*inputs:*/ tensor_t A, tensor_t B, std::variant<void *, tensor_t>& C, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha, std::variant<void *, float > beta, std::variant<void *, int64_t > transA, std::variant<void *, int64_t > transB) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool
	virtual OperatorReturnType onnx_GlobalAveragePool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalLpPool
	virtual OperatorReturnType onnx_GlobalLpPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, int64_t > p) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool
	virtual OperatorReturnType onnx_GlobalMaxPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater
	virtual OperatorReturnType onnx_Greater(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GreaterOrEqual
	virtual OperatorReturnType onnx_GreaterOrEqual(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid
	virtual OperatorReturnType onnx_HardSigmoid(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha, std::variant<void *, float > beta) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSwish
	virtual OperatorReturnType onnx_HardSwish(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Hardmax
	virtual OperatorReturnType onnx_Hardmax(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity
	virtual OperatorReturnType onnx_Identity(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization
	virtual OperatorReturnType onnx_InstanceNormalization(/*inputs:*/ tensor_t input, tensor_t scale, tensor_t B, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, float > epsilon) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsInf
	virtual OperatorReturnType onnx_IsInf(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, int64_t > detect_negative, std::variant<void *, int64_t > detect_positive) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsNaN
	virtual OperatorReturnType onnx_IsNaN(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN
	virtual OperatorReturnType onnx_LRN(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha, std::variant<void *, float > beta, std::variant<void *, float > bias, int64_t size) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM
	virtual OperatorReturnType onnx_LSTM(/*inputs:*/ tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, std::variant<void *, tensor_t>& initial_c, std::variant<void *, tensor_t>& P, /*outputs:*/ std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, std::variant<void *, tensor_t>& Y_c, /*attributes:*/ std::variant<void *, std::vector<float> > activation_alpha, std::variant<void *, std::vector<float> > activation_beta, std::variant<void *, std::vector<std::string> > activations, std::variant<void *, float > clip, std::variant<void *, std::string > direction, std::variant<void *, int64_t > hidden_size, std::variant<void *, int64_t > input_forget, std::variant<void *, int64_t > layout) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LeakyRelu
	virtual OperatorReturnType onnx_LeakyRelu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less
	virtual OperatorReturnType onnx_Less(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LessOrEqual
	virtual OperatorReturnType onnx_LessOrEqual(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log
	virtual OperatorReturnType onnx_Log(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LogSoftmax
	virtual OperatorReturnType onnx_LogSoftmax(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpNormalization
	virtual OperatorReturnType onnx_LpNormalization(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > p) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpPool
	virtual OperatorReturnType onnx_LpPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::vector<int64_t> kernel_shape, std::variant<void *, int64_t > p, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul
	virtual OperatorReturnType onnx_MatMul(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMulInteger
	virtual OperatorReturnType onnx_MatMulInteger(/*inputs:*/ tensor_t A, tensor_t B, std::variant<void *, tensor_t>& a_zero_point, std::variant<void *, tensor_t>& b_zero_point, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Max
	virtual OperatorReturnType onnx_Max(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t max) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
	virtual OperatorReturnType onnx_MaxPool(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, std::variant<void *, tensor_t>& Indices, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, int64_t > ceil_mode, std::variant<void *, std::vector<int64_t> > dilations, std::vector<int64_t> kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, int64_t > storage_order, std::variant<void *, std::vector<int64_t> > strides) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxRoiPool
	virtual OperatorReturnType onnx_MaxRoiPool(/*inputs:*/ tensor_t X, tensor_t rois, /*outputs:*/ tensor_t Y, /*attributes:*/ std::vector<int64_t> pooled_shape, std::variant<void *, float > spatial_scale) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxUnpool
	virtual OperatorReturnType onnx_MaxUnpool(/*inputs:*/ tensor_t X, tensor_t I, std::variant<void *, tensor_t>& output_shape, /*outputs:*/ tensor_t output, /*attributes:*/ std::vector<int64_t> kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mean
	virtual OperatorReturnType onnx_Mean(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t mean) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MeanVarianceNormalization
	virtual OperatorReturnType onnx_MeanVarianceNormalization(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min
	virtual OperatorReturnType onnx_Min(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t min) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mod
	virtual OperatorReturnType onnx_Mod(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C, /*attributes:*/ std::variant<void *, int64_t > fmod) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
	virtual OperatorReturnType onnx_Mul(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Multinomial
	virtual OperatorReturnType onnx_Multinomial(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, int64_t > sample_size, std::variant<void *, float > seed) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Neg
	virtual OperatorReturnType onnx_Neg(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#NegativeLogLikelihoodLoss
	virtual OperatorReturnType onnx_NegativeLogLikelihoodLoss(/*inputs:*/ tensor_t input, tensor_t target, std::variant<void *, tensor_t>& weight, /*outputs:*/ tensor_t loss, /*attributes:*/ std::variant<void *, int64_t > ignore_index, std::variant<void *, std::string > reduction) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression
	virtual OperatorReturnType onnx_NonMaxSuppression(/*inputs:*/ tensor_t boxes, tensor_t scores, std::variant<void *, tensor_t>& max_output_boxes_per_class, std::variant<void *, tensor_t>& iou_threshold, std::variant<void *, tensor_t>& score_threshold, /*outputs:*/ tensor_t selected_indices, /*attributes:*/ std::variant<void *, int64_t > center_point_box) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonZero
	virtual OperatorReturnType onnx_NonZero(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Not
	virtual OperatorReturnType onnx_Not(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#OneHot
	virtual OperatorReturnType onnx_OneHot(/*inputs:*/ tensor_t indices, tensor_t depth, tensor_t values, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Or
	virtual OperatorReturnType onnx_Or(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu
	virtual OperatorReturnType onnx_PRelu(/*inputs:*/ tensor_t X, tensor_t slope, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad
	virtual OperatorReturnType onnx_Pad(/*inputs:*/ tensor_t data, tensor_t pads, std::variant<void *, tensor_t>& constant_value, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, std::string > mode) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow
	virtual OperatorReturnType onnx_Pow(/*inputs:*/ tensor_t X, tensor_t Y, /*outputs:*/ tensor_t Z) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv
	virtual OperatorReturnType onnx_QLinearConv(/*inputs:*/ tensor_t x, tensor_t x_scale, tensor_t x_zero_point, tensor_t w, tensor_t w_scale, tensor_t w_zero_point, tensor_t y_scale, tensor_t y_zero_point, std::variant<void *, tensor_t>& B, /*outputs:*/ tensor_t y, /*attributes:*/ std::variant<void *, std::string > auto_pad, std::variant<void *, std::vector<int64_t> > dilations, std::variant<void *, int64_t > group, std::variant<void *, std::vector<int64_t> > kernel_shape, std::variant<void *, std::vector<int64_t> > pads, std::variant<void *, std::vector<int64_t> > strides) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearMatMul
	virtual OperatorReturnType onnx_QLinearMatMul(/*inputs:*/ tensor_t a, tensor_t a_scale, tensor_t a_zero_point, tensor_t b, tensor_t b_scale, tensor_t b_zero_point, tensor_t y_scale, tensor_t y_zero_point, /*outputs:*/ tensor_t y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear
	virtual OperatorReturnType onnx_QuantizeLinear(/*inputs:*/ tensor_t x, tensor_t y_scale, std::variant<void *, tensor_t>& y_zero_point, /*outputs:*/ tensor_t y, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN
	virtual OperatorReturnType onnx_RNN(/*inputs:*/ tensor_t X, tensor_t W, tensor_t R, std::variant<void *, tensor_t>& B, std::variant<void *, tensor_t>& sequence_lens, std::variant<void *, tensor_t>& initial_h, /*outputs:*/ std::variant<void *, tensor_t>& Y, std::variant<void *, tensor_t>& Y_h, /*attributes:*/ std::variant<void *, std::vector<float> > activation_alpha, std::variant<void *, std::vector<float> > activation_beta, std::variant<void *, std::vector<std::string> > activations, std::variant<void *, float > clip, std::variant<void *, std::string > direction, std::variant<void *, int64_t > hidden_size, std::variant<void *, int64_t > layout) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormal
	virtual OperatorReturnType onnx_RandomNormal(/*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, float > mean, std::variant<void *, float > scale, std::variant<void *, float > seed, std::vector<int64_t> shape) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormalLike
	virtual OperatorReturnType onnx_RandomNormalLike(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, float > mean, std::variant<void *, float > scale, std::variant<void *, float > seed) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniform
	virtual OperatorReturnType onnx_RandomUniform(/*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, float > high, std::variant<void *, float > low, std::variant<void *, float > seed, std::vector<int64_t> shape) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniformLike
	virtual OperatorReturnType onnx_RandomUniformLike(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype, std::variant<void *, float > high, std::variant<void *, float > low, std::variant<void *, float > seed) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range
	virtual OperatorReturnType onnx_Range(/*inputs:*/ tensor_t start, tensor_t limit, tensor_t delta, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reciprocal
	virtual OperatorReturnType onnx_Reciprocal(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL1
	virtual OperatorReturnType onnx_ReduceL1(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2
	virtual OperatorReturnType onnx_ReduceL2(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSum
	virtual OperatorReturnType onnx_ReduceLogSum(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSumExp
	virtual OperatorReturnType onnx_ReduceLogSumExp(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMax
	virtual OperatorReturnType onnx_ReduceMax(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
	virtual OperatorReturnType onnx_ReduceMean(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMin
	virtual OperatorReturnType onnx_ReduceMin(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceProd
	virtual OperatorReturnType onnx_ReduceProd(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum
	virtual OperatorReturnType onnx_ReduceSum(/*inputs:*/ tensor_t data, std::variant<void *, tensor_t>& axes, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, int64_t > keepdims, std::variant<void *, int64_t > noop_with_empty_axes) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSumSquare
	virtual OperatorReturnType onnx_ReduceSumSquare(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t reduced, /*attributes:*/ std::variant<void *, std::vector<int64_t> > axes, std::variant<void *, int64_t > keepdims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
	virtual OperatorReturnType onnx_Relu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape
	virtual OperatorReturnType onnx_Reshape(/*inputs:*/ tensor_t data, tensor_t shape, /*outputs:*/ tensor_t reshaped, /*attributes:*/ std::variant<void *, int64_t > allowzero) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
	virtual OperatorReturnType onnx_Resize(/*inputs:*/ tensor_t X, std::variant<void *, tensor_t>& roi, std::variant<void *, tensor_t>& scales, std::variant<void *, tensor_t>& sizes, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > coordinate_transformation_mode, std::variant<void *, float > cubic_coeff_a, std::variant<void *, int64_t > exclude_outside, std::variant<void *, float > extrapolation_value, std::variant<void *, std::string > mode, std::variant<void *, std::string > nearest_mode) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReverseSequence
	virtual OperatorReturnType onnx_ReverseSequence(/*inputs:*/ tensor_t input, tensor_t sequence_lens, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, int64_t > batch_axis, std::variant<void *, int64_t > time_axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#RoiAlign
	virtual OperatorReturnType onnx_RoiAlign(/*inputs:*/ tensor_t X, tensor_t rois, tensor_t batch_indices, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > mode, std::variant<void *, int64_t > output_height, std::variant<void *, int64_t > output_width, std::variant<void *, int64_t > sampling_ratio, std::variant<void *, float > spatial_scale) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Round
	virtual OperatorReturnType onnx_Round(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements
	virtual OperatorReturnType onnx_ScatterElements(/*inputs:*/ tensor_t data, tensor_t indices, tensor_t updates, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND
	virtual OperatorReturnType onnx_ScatterND(/*inputs:*/ tensor_t data, tensor_t indices, tensor_t updates, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Selu
	virtual OperatorReturnType onnx_Selu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha, std::variant<void *, float > gamma) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceAt
	virtual OperatorReturnType onnx_SequenceAt(/*inputs:*/ tensor_t input_sequence, tensor_t position, /*outputs:*/ tensor_t tensor) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceConstruct
	virtual OperatorReturnType onnx_SequenceConstruct(/*inputs:*/ std::vector<tensor_t>& inputs, /*outputs:*/ tensor_t output_sequence) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceEmpty
	virtual OperatorReturnType onnx_SequenceEmpty(/*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > dtype) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceErase
	virtual OperatorReturnType onnx_SequenceErase(/*inputs:*/ tensor_t input_sequence, std::variant<void *, tensor_t>& position, /*outputs:*/ tensor_t output_sequence) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceInsert
	virtual OperatorReturnType onnx_SequenceInsert(/*inputs:*/ tensor_t input_sequence, tensor_t tensor, std::variant<void *, tensor_t>& position, /*outputs:*/ tensor_t output_sequence) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceLength
	virtual OperatorReturnType onnx_SequenceLength(/*inputs:*/ tensor_t input_sequence, /*outputs:*/ tensor_t length) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape
	virtual OperatorReturnType onnx_Shape(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t shape, /*attributes:*/ std::variant<void *, int64_t > end, std::variant<void *, int64_t > start) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shrink
	virtual OperatorReturnType onnx_Shrink(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, float > bias, std::variant<void *, float > lambd) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid
	virtual OperatorReturnType onnx_Sigmoid(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sign
	virtual OperatorReturnType onnx_Sign(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sin
	virtual OperatorReturnType onnx_Sin(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sinh
	virtual OperatorReturnType onnx_Sinh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size
	virtual OperatorReturnType onnx_Size(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t size) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice
	virtual OperatorReturnType onnx_Slice(/*inputs:*/ tensor_t data, tensor_t starts, tensor_t ends, std::variant<void *, tensor_t>& axes, std::variant<void *, tensor_t>& steps, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
	virtual OperatorReturnType onnx_Softmax(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SoftmaxCrossEntropyLoss
	virtual OperatorReturnType onnx_SoftmaxCrossEntropyLoss(/*inputs:*/ tensor_t scores, tensor_t labels, std::variant<void *, tensor_t>& weights, /*outputs:*/ tensor_t output, std::variant<void *, tensor_t>& log_prob, /*attributes:*/ std::variant<void *, int64_t > ignore_index, std::variant<void *, std::string > reduction) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softplus
	virtual OperatorReturnType onnx_Softplus(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softsign
	virtual OperatorReturnType onnx_Softsign(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SpaceToDepth
	virtual OperatorReturnType onnx_SpaceToDepth(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output, /*attributes:*/ int64_t blocksize) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
	virtual OperatorReturnType onnx_Split(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& split, /*outputs:*/ std::vector<tensor_t>& outputs, /*attributes:*/ std::variant<void *, int64_t > axis) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#SplitToSequence
	virtual OperatorReturnType onnx_SplitToSequence(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& split, /*outputs:*/ tensor_t output_sequence, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > keepdims) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt
	virtual OperatorReturnType onnx_Sqrt(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Squeeze
	virtual OperatorReturnType onnx_Squeeze(/*inputs:*/ tensor_t data, std::variant<void *, tensor_t>& axes, /*outputs:*/ tensor_t squeezed) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#StringNormalizer
	virtual OperatorReturnType onnx_StringNormalizer(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, std::string > case_change_action, std::variant<void *, int64_t > is_case_sensitive, std::variant<void *, std::string > locale, std::variant<void *, std::vector<std::string> > stopwords) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub
	virtual OperatorReturnType onnx_Sub(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sum
	virtual OperatorReturnType onnx_Sum(/*inputs:*/ std::vector<tensor_t>& data_0, /*outputs:*/ tensor_t sum) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tan
	virtual OperatorReturnType onnx_Tan(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh
	virtual OperatorReturnType onnx_Tanh(/*inputs:*/ tensor_t input, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#TfIdfVectorizer
	virtual OperatorReturnType onnx_TfIdfVectorizer(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ int64_t max_gram_length, int64_t max_skip_count, int64_t min_gram_length, std::string mode, std::vector<int64_t> ngram_counts, std::vector<int64_t> ngram_indexes, std::variant<void *, std::vector<int64_t> > pool_int64s, std::variant<void *, std::vector<std::string> > pool_strings, std::variant<void *, std::vector<float> > weights) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ThresholdedRelu
	virtual OperatorReturnType onnx_ThresholdedRelu(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, /*attributes:*/ std::variant<void *, float > alpha) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile
	virtual OperatorReturnType onnx_Tile(/*inputs:*/ tensor_t input, tensor_t repeats, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK
	virtual OperatorReturnType onnx_TopK(/*inputs:*/ tensor_t X, tensor_t K, /*outputs:*/ tensor_t Values, tensor_t Indices, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > largest, std::variant<void *, int64_t > sorted) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose
	virtual OperatorReturnType onnx_Transpose(/*inputs:*/ tensor_t data, /*outputs:*/ tensor_t transposed, /*attributes:*/ std::variant<void *, std::vector<int64_t> > perm) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Trilu
	virtual OperatorReturnType onnx_Trilu(/*inputs:*/ tensor_t input, std::variant<void *, tensor_t>& k, /*outputs:*/ tensor_t output, /*attributes:*/ std::variant<void *, int64_t > upper) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unique
	virtual OperatorReturnType onnx_Unique(/*inputs:*/ tensor_t X, /*outputs:*/ tensor_t Y, std::variant<void *, tensor_t>& indices, std::variant<void *, tensor_t>& inverse_indices, std::variant<void *, tensor_t>& counts, /*attributes:*/ std::variant<void *, int64_t > axis, std::variant<void *, int64_t > sorted) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze
	virtual OperatorReturnType onnx_Unsqueeze(/*inputs:*/ tensor_t data, tensor_t axes, /*outputs:*/ tensor_t expanded) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where
	virtual OperatorReturnType onnx_Where(/*inputs:*/ tensor_t condition, tensor_t X, tensor_t Y, /*outputs:*/ tensor_t output) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor
	virtual OperatorReturnType onnx_Xor(/*inputs:*/ tensor_t A, tensor_t B, /*outputs:*/ tensor_t C) {
#ifndef USING_ONNX_IMPL
	    return YNX_TODO_ERROR;
#else
	    return YNX_OK;
#endif
	}



};

#ifdef USING_ONNX_IMPL
InferenceFunction query_inference_function(const std::string& op_name) {
    static std::map<const std::string, InferenceFunction> allInferenceFunctions;

    if ( allInferenceFunctions.size() == 0) {
        const std::vector<OpSchema> schemas = OpSchemaRegistry::get_all_schemas();

        for(size_t i = 0; i < schemas.size(); i++) {
            std::string name = schemas[i].Name();
            auto f = schemas[i].GetTypeAndShapeInferenceFunction();

            allInferenceFunctions[name] = f;
        }
    }

    if ( allInferenceFunctions.find(op_name) == allInferenceFunctions.end() ) {
        yannx_panic("Can't find InferenceFunction!");
    }

    return allInferenceFunctions[op_name];
}

struct YNXInferenceContextImpl : public InferenceContext {
    std::map<std::string, AttributeProto> attrs_;
    std::map<size_t, TypeProto> input_types_;
    std::vector<TypeProto> output_types_;

    size_t input_num_;
    const size_t output_num_;

    YNXInferenceContextImpl(size_t output_num) : output_num_(output_num) {
        input_num_ = 0;
        output_types_.resize( output_num_);
    }

    // setup interfaces
    void new_attr(const std::string& name, const float v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, const int64_t v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, const std::string& v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, const std::vector<float>& v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, const std::vector<int64_t>& v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, const std::vector<std::string>& v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }

    void new_input(tensor_t t) {
        TypeProto proto;

        TypeProto_Tensor* p_tensor = proto.mutable_tensor_type();
        p_tensor->set_elem_type( t->dtype_ );
        auto* shape = p_tensor->mutable_shape();

        shape->clear_dim();
        for (size_t i = 0; i < t->shape_.size(); i++) {
            shape->add_dim();
            auto dim = shape->mutable_dim(i);
            dim->set_dim_value( t->shape_[i] );
        }
        size_t index = input_num_;
        input_types_[index] = proto;

        input_num_ ++;
    }
    void new_input(std::variant<void *, tensor_t> v) {
        if ( v.index() == 1 ) {
            new_input(std::get<1>(v) );
        } else {
            input_num_ ++;
        }
    }
    void new_input(std::vector<tensor_t> v) {
        for (size_t i = 0; i < v.size(); i++) {
            new_input(v);
        }
    }

    // call InferenceFunction
    void do_inference(InferenceFunction f) {
        f( *this );
    }

    int check_output(size_t index, tensor_t t) {
        auto* proto = getOutputType(index);
        auto p_tensor = proto->tensor_type();

        if (! p_tensor.has_elem_type() ) {
            return YNX_OUTPUT_ERROR;
        }
        TensorDataType dtype =  datatype_from_onnx(p_tensor.elem_type());

        std::vector<size_t> shape;
        auto shape_proto = p_tensor.shape();

        for (int i = 0; i < shape_proto.dim_size(); i++ ) {
            if ( !shape_proto.dim(i).has_dim_value() ) {
                return YNX_OUTPUT_ERROR;
            }
            shape.push_back( shape_proto.dim(i).dim_value() );
        }

        t->reset(dtype, shape);

        return YNX_OK;
    }
    int check_output(size_t index, std::variant<void *, tensor_t>& v) {
        tensor_t output = TensorType::create_undefined_user_tensor();
        if ( check_output(index, output) == YNX_OK ) {
            std::get<1>(v) = output;
        }
        return YNX_OK;
    }
    int check_output(size_t index, std::vector<tensor_t>& v) {
        yannx_panic("FIXME: how to check Variadic type");
        return YNX_OUTPUT_ERROR;
    }

    // InferenceContext apis
    size_t getNumInputs() const override {
        return input_num_;
    }
    size_t getNumOutputs() const override {
        return output_types_.size();
    }
    const AttributeProto* getAttribute(const std::string& name) const override {
        if ( attrs_.find(name) != attrs_.end() ) {
            return &( attrs_.find(name)->second );
        }
        return nullptr;
    }
    const TypeProto* getInputType(size_t index) const override {
        if ( input_types_.find(index) != input_types_.end() ) {
            return &( input_types_.find(index)->second );
        }
        return nullptr;
    }
    TypeProto* getOutputType(size_t index) override {
        if ( index < output_types_.size() ) {
            return  &output_types_[index];
        }
        return nullptr;
    }

    // Skipping these impl, FIXME TensorProto seems not be used by Type&Shape inference
    const TensorProto* getInputData(size_t index) const override {
        return nullptr;
    }
    const TensorShapeProto* getSymbolicInput(size_t index) const override {
        return nullptr;
    }
    const SparseTensorProto* getInputSparseData(size_t index) const override {
        return nullptr;
    }
    GraphInferencer* getGraphAttributeInferencer( const std::string& attr_name) override {
        return nullptr;
    }
};
#endif

//
//  some common help functions, and none-auto operators
//
static float fetch_float(ValueStack<TensorType>& stack) {
    float v = stack.pop_number();
    return v;
}

static int64_t fetch_int(ValueStack<TensorType>& stack) {
    int64_t v = stack.pop_number();
    return v;
}

static std::string fetch_string(ValueStack<TensorType>& stack) {
    std::string v = stack.pop_string();
    return v;
}

static tensor_t fetch_tensor(ValueStack<TensorType>& stack) {
    auto v = stack.pop_tensor();
    return v;
}

static std::vector<float> fetch_floats(ValueStack<TensorType>& stack) {
    auto v = stack.pop_number_tuple();
    std::vector<float> ret;
    for (size_t i = 0; i < v.size(); i++) {
        ret.push_back( v[i] );
    }
    return ret;
}

static std::vector<int64_t> fetch_ints(ValueStack<TensorType>& stack) {
    auto v = stack.pop_number_tuple();
    std::vector<int64_t> ret;
    for (size_t i = 0; i < v.size(); i++) {
        ret.push_back( v[i] );
    }
    return ret;
}

static std::vector<std::string> fetch_strings(ValueStack<TensorType>& stack) {
    auto v = stack.pop_string_tuple();
    return v;
}

static std::vector<tensor_t> fetch_tensors(ValueStack<TensorType>& stack) {
    auto v = stack.pop_tensor_tuple();
    return v;
}

static std::variant<void *, float> fetch_optional_float(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, float>(nullptr);
    }
    return std::variant<void *, float>( fetch_float(stack) );
}

static std::variant<void *, int64_t> fetch_optional_int(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, int64_t>(nullptr);
    }
    return std::variant<void *, int64_t>( fetch_int(stack) );
}

static std::variant<void *, std::string> fetch_optional_string(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::string>(nullptr);
    }
    return std::variant<void *, std::string>( fetch_string(stack) );
}

static std::variant<void *, tensor_t> fetch_optional_tensor(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, tensor_t>(nullptr);
    }
    return std::variant<void *, tensor_t>( fetch_tensor(stack) );
}

static std::variant<void *, std::vector<float> > fetch_optional_floats(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::vector<float> >(nullptr);
    }
    return std::variant<void *, std::vector<float> >( fetch_floats(stack) );
}

static std::variant<void *, std::vector<int64_t> > fetch_optional_ints(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::vector<int64_t> >(nullptr);
    }
    return std::variant<void *, std::vector<int64_t> >( fetch_ints(stack) );
}

static std::variant<void *, std::vector<std::string> > fetch_optional_strings(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::vector<std::string>> (nullptr);
    }
    return std::variant<void *, std::vector<std::string> >( fetch_strings(stack) );
}

static void put_tensor(ValueStack<TensorType>& stack, tensor_t t) {
    stack.push_tensor(t);
}

static void put_tensors(ValueStack<TensorType>& stack, std::vector<tensor_t>& ts) {
    for (size_t i = 0; i < ts.size(); i++) {
        stack.push_tensor(ts[i]);
    }
    stack.push_number(ts.size());
}

static void put_optional_tensor(ValueStack<TensorType>& stack, std::variant<void*, tensor_t>& ot) {
    if ( ot.index() == 0) {
        stack.push_none();
        return;
    }
    stack.push_tensor( std::get<1>(ot) );
}

// some help words, like Constant Parameter Variable
#define NWORD_CREATOR_DEFINE_TENSORTYPE(CLS)                                                \
    static std::shared_ptr<NativeWord<TensorType> >   creator(Runtime<TensorType>& rt ) {   \
        std::shared_ptr<NativeWord<TensorType> > wd(new CLS());                             \
        return wd;                                                                          \
    }

namespace common {
    struct Constant : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto shape_ = fetch_ints(stack);
            std::vector<size_t> shape;
            for(size_t i = 0; i < shape_.size(); i++) {
                shape.push_back( shape_[i] );
            }

            std::string dtype_string = fetch_string(stack);
            auto dtype = datatype_from_string(dtype_string);

            output = TensorType::create_undefined_user_tensor();
            TensorType::register_user_tensor(output, TensorType::TT_Constant);
            output->reset(dtype, shape);

            put_tensor(stack, output);
        }
        virtual void run(ValueStack<TensorType>& stack) {
            fetch_ints(stack);
            fetch_string(stack);

            put_tensor(stack, output);
        }

        NWORD_CREATOR_DEFINE_TENSORTYPE(Constant)
    };

    struct Variable : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto shape_ = fetch_ints(stack);
            std::vector<size_t> shape;
            for(size_t i = 0; i < shape_.size(); i++) {
                shape.push_back( shape_[i] );
            }

            std::string dtype_string = fetch_string(stack);
            auto dtype = datatype_from_string(dtype_string);

            output = TensorType::create_undefined_user_tensor();
            output->reset(dtype, shape);

            put_tensor(stack, output);
        }
        virtual void run(ValueStack<TensorType>& stack) {
            put_tensor(stack, output);
        }

        NWORD_CREATOR_DEFINE_TENSORTYPE(Variable)
    };

}

namespace generator {

    struct RandomNormalLike : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto seed = fetch_optional_float(stack);
            auto scale = fetch_optional_float(stack);
            auto mean = fetch_optional_float(stack);
            auto dtype = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( seed.index() != 0) {
                infer_.new_attr("seed", std::get<1>(seed) );
            }
            if ( scale.index() != 0) {
                infer_.new_attr("scale", std::get<1>(scale) );
            }
            if ( mean.index() != 0) {
                infer_.new_attr("mean", std::get<1>(mean) );
            }
            if ( dtype.index() != 0) {
                infer_.new_attr("dtype", std::get<1>(dtype) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("RandomNormalLike");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_RandomNormalLike(input, output, dtype, mean, scale, seed) != YNX_OK ) {
                yannx_panic("API: RandomNormalLike  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto seed = fetch_optional_float(stack);
            auto scale = fetch_optional_float(stack);
            auto mean = fetch_optional_float(stack);
            auto dtype = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_RandomNormalLike(input, output, dtype, mean, scale, seed) != YNX_OK ) {
                yannx_panic("API: RandomNormalLike  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RandomNormalLike)
    };


    struct RandomNormal : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto shape = fetch_ints(stack);
            auto seed = fetch_optional_float(stack);
            auto scale = fetch_optional_float(stack);
            auto mean = fetch_optional_float(stack);
            auto dtype = fetch_optional_int(stack);


            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            infer_.new_attr("shape", shape);
            if ( seed.index() != 0) {
                infer_.new_attr("seed", std::get<1>(seed) );
            }
            if ( scale.index() != 0) {
                infer_.new_attr("scale", std::get<1>(scale) );
            }
            if ( mean.index() != 0) {
                infer_.new_attr("mean", std::get<1>(mean) );
            }
            if ( dtype.index() != 0) {
                infer_.new_attr("dtype", std::get<1>(dtype) );
            }


            auto f = query_inference_function("RandomNormal");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( output->onnx_RandomNormal(output, dtype, mean, scale, seed, shape) != YNX_OK ) {
                yannx_panic("API: RandomNormal  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto shape = fetch_ints(stack);
            auto seed = fetch_optional_float(stack);
            auto scale = fetch_optional_float(stack);
            auto mean = fetch_optional_float(stack);
            auto dtype = fetch_optional_int(stack);



            if ( output->onnx_RandomNormal(output, dtype, mean, scale, seed, shape) != YNX_OK ) {
                yannx_panic("API: RandomNormal  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RandomNormal)
    };


    struct RandomUniform : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto shape = fetch_ints(stack);
            auto seed = fetch_optional_float(stack);
            auto low = fetch_optional_float(stack);
            auto high = fetch_optional_float(stack);
            auto dtype = fetch_optional_int(stack);


            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            infer_.new_attr("shape", shape);
            if ( seed.index() != 0) {
                infer_.new_attr("seed", std::get<1>(seed) );
            }
            if ( low.index() != 0) {
                infer_.new_attr("low", std::get<1>(low) );
            }
            if ( high.index() != 0) {
                infer_.new_attr("high", std::get<1>(high) );
            }
            if ( dtype.index() != 0) {
                infer_.new_attr("dtype", std::get<1>(dtype) );
            }


            auto f = query_inference_function("RandomUniform");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( output->onnx_RandomUniform(output, dtype, high, low, seed, shape) != YNX_OK ) {
                yannx_panic("API: RandomUniform  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto shape = fetch_ints(stack);
            auto seed = fetch_optional_float(stack);
            auto low = fetch_optional_float(stack);
            auto high = fetch_optional_float(stack);
            auto dtype = fetch_optional_int(stack);



            if ( output->onnx_RandomUniform(output, dtype, high, low, seed, shape) != YNX_OK ) {
                yannx_panic("API: RandomUniform  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RandomUniform)
    };


    struct EyeLike : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto k = fetch_optional_int(stack);
            auto dtype = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( k.index() != 0) {
                infer_.new_attr("k", std::get<1>(k) );
            }
            if ( dtype.index() != 0) {
                infer_.new_attr("dtype", std::get<1>(dtype) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("EyeLike");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_EyeLike(input, output, dtype, k) != YNX_OK ) {
                yannx_panic("API: EyeLike  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto k = fetch_optional_int(stack);
            auto dtype = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_EyeLike(input, output, dtype, k) != YNX_OK ) {
                yannx_panic("API: EyeLike  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(EyeLike)
    };


    struct Bernoulli : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto seed = fetch_optional_float(stack);
            auto dtype = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( seed.index() != 0) {
                infer_.new_attr("seed", std::get<1>(seed) );
            }
            if ( dtype.index() != 0) {
                infer_.new_attr("dtype", std::get<1>(dtype) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("Bernoulli");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Bernoulli(input, output, dtype, seed) != YNX_OK ) {
                yannx_panic("API: Bernoulli  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto seed = fetch_optional_float(stack);
            auto dtype = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Bernoulli(input, output, dtype, seed) != YNX_OK ) {
                yannx_panic("API: Bernoulli  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Bernoulli)
    };


    struct Multinomial : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto seed = fetch_optional_float(stack);
            auto sample_size = fetch_optional_int(stack);
            auto dtype = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( seed.index() != 0) {
                infer_.new_attr("seed", std::get<1>(seed) );
            }
            if ( sample_size.index() != 0) {
                infer_.new_attr("sample_size", std::get<1>(sample_size) );
            }
            if ( dtype.index() != 0) {
                infer_.new_attr("dtype", std::get<1>(dtype) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("Multinomial");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Multinomial(input, output, dtype, sample_size, seed) != YNX_OK ) {
                yannx_panic("API: Multinomial  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto seed = fetch_optional_float(stack);
            auto sample_size = fetch_optional_int(stack);
            auto dtype = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Multinomial(input, output, dtype, sample_size, seed) != YNX_OK ) {
                yannx_panic("API: Multinomial  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Multinomial)
    };


    struct RandomUniformLike : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto seed = fetch_optional_float(stack);
            auto low = fetch_optional_float(stack);
            auto high = fetch_optional_float(stack);
            auto dtype = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( seed.index() != 0) {
                infer_.new_attr("seed", std::get<1>(seed) );
            }
            if ( low.index() != 0) {
                infer_.new_attr("low", std::get<1>(low) );
            }
            if ( high.index() != 0) {
                infer_.new_attr("high", std::get<1>(high) );
            }
            if ( dtype.index() != 0) {
                infer_.new_attr("dtype", std::get<1>(dtype) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("RandomUniformLike");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_RandomUniformLike(input, output, dtype, high, low, seed) != YNX_OK ) {
                yannx_panic("API: RandomUniformLike  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto seed = fetch_optional_float(stack);
            auto low = fetch_optional_float(stack);
            auto high = fetch_optional_float(stack);
            auto dtype = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_RandomUniformLike(input, output, dtype, high, low, seed) != YNX_OK ) {
                yannx_panic("API: RandomUniformLike  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RandomUniformLike)
    };


    struct Range : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto delta = fetch_tensor(stack);
            auto limit = fetch_tensor(stack);
            auto start = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(delta);
            infer_.new_input(limit);
            infer_.new_input(start);

            auto f = query_inference_function("Range");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( start->onnx_Range(start, limit, delta, output) != YNX_OK ) {
                yannx_panic("API: Range  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto delta = fetch_tensor(stack);
            auto limit = fetch_tensor(stack);
            auto start = fetch_tensor(stack);


            if ( start->onnx_Range(start, limit, delta, output) != YNX_OK ) {
                yannx_panic("API: Range  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Range)
    };

}
namespace logical {

    struct GreaterOrEqual : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("GreaterOrEqual");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_GreaterOrEqual(A, B, C) != YNX_OK ) {
                yannx_panic("API: GreaterOrEqual  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_GreaterOrEqual(A, B, C) != YNX_OK ) {
                yannx_panic("API: GreaterOrEqual  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GreaterOrEqual)
    };


    struct Or : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Or");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_Or(A, B, C) != YNX_OK ) {
                yannx_panic("API: Or  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Or(A, B, C) != YNX_OK ) {
                yannx_panic("API: Or  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Or)
    };


    struct BitShift : NativeWord<TensorType> {
        tensor_t Z;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto direction = fetch_string(stack);

            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);

            Z = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            infer_.new_attr("direction", direction);

            infer_.new_input(Y);
            infer_.new_input(X);

            auto f = query_inference_function("BitShift");
            infer_.do_inference(f);
            infer_.check_output(0, Z);

#endif

            if ( X->onnx_BitShift(X, Y, Z, direction) != YNX_OK ) {
                yannx_panic("API: BitShift  return error!");
            }

            put_tensor(stack, Z);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto direction = fetch_string(stack);

            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_BitShift(X, Y, Z, direction) != YNX_OK ) {
                yannx_panic("API: BitShift  return error!");
            }

            put_tensor(stack, Z);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(BitShift)
    };


    struct Greater : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Greater");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_Greater(A, B, C) != YNX_OK ) {
                yannx_panic("API: Greater  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Greater(A, B, C) != YNX_OK ) {
                yannx_panic("API: Greater  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Greater)
    };


    struct Xor : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Xor");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_Xor(A, B, C) != YNX_OK ) {
                yannx_panic("API: Xor  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Xor(A, B, C) != YNX_OK ) {
                yannx_panic("API: Xor  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Xor)
    };


    struct And : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("And");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_And(A, B, C) != YNX_OK ) {
                yannx_panic("API: And  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_And(A, B, C) != YNX_OK ) {
                yannx_panic("API: And  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(And)
    };


    struct LessOrEqual : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("LessOrEqual");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_LessOrEqual(A, B, C) != YNX_OK ) {
                yannx_panic("API: LessOrEqual  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_LessOrEqual(A, B, C) != YNX_OK ) {
                yannx_panic("API: LessOrEqual  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LessOrEqual)
    };


    struct Not : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Not");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Not(X, Y) != YNX_OK ) {
                yannx_panic("API: Not  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Not(X, Y) != YNX_OK ) {
                yannx_panic("API: Not  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Not)
    };


    struct Equal : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Equal");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_Equal(A, B, C) != YNX_OK ) {
                yannx_panic("API: Equal  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Equal(A, B, C) != YNX_OK ) {
                yannx_panic("API: Equal  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Equal)
    };


    struct Less : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Less");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_Less(A, B, C) != YNX_OK ) {
                yannx_panic("API: Less  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Less(A, B, C) != YNX_OK ) {
                yannx_panic("API: Less  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Less)
    };

}
namespace math {

    struct Reciprocal : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Reciprocal");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Reciprocal(X, Y) != YNX_OK ) {
                yannx_panic("API: Reciprocal  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Reciprocal(X, Y) != YNX_OK ) {
                yannx_panic("API: Reciprocal  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Reciprocal)
    };


    struct LeakyRelu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( alpha.index() != 0) {
                infer_.new_attr("alpha", std::get<1>(alpha) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("LeakyRelu");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_LeakyRelu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: LeakyRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_LeakyRelu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: LeakyRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LeakyRelu)
    };


    struct HardSigmoid : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto beta = fetch_optional_float(stack);
            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( beta.index() != 0) {
                infer_.new_attr("beta", std::get<1>(beta) );
            }
            if ( alpha.index() != 0) {
                infer_.new_attr("alpha", std::get<1>(alpha) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("HardSigmoid");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_HardSigmoid(X, Y, alpha, beta) != YNX_OK ) {
                yannx_panic("API: HardSigmoid  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto beta = fetch_optional_float(stack);
            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_HardSigmoid(X, Y, alpha, beta) != YNX_OK ) {
                yannx_panic("API: HardSigmoid  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(HardSigmoid)
    };


    struct Div : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Div");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_Div(A, B, C) != YNX_OK ) {
                yannx_panic("API: Div  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Div(A, B, C) != YNX_OK ) {
                yannx_panic("API: Div  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Div)
    };


    struct Pow : NativeWord<TensorType> {
        tensor_t Z;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);

            Z = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(Y);
            infer_.new_input(X);

            auto f = query_inference_function("Pow");
            infer_.do_inference(f);
            infer_.check_output(0, Z);

#endif

            if ( X->onnx_Pow(X, Y, Z) != YNX_OK ) {
                yannx_panic("API: Pow  return error!");
            }

            put_tensor(stack, Z);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_Pow(X, Y, Z) != YNX_OK ) {
                yannx_panic("API: Pow  return error!");
            }

            put_tensor(stack, Z);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Pow)
    };


    struct Mul : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Mul");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_Mul(A, B, C) != YNX_OK ) {
                yannx_panic("API: Mul  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Mul(A, B, C) != YNX_OK ) {
                yannx_panic("API: Mul  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Mul)
    };


    struct Min : NativeWord<TensorType> {
        tensor_t min;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto data_0 = fetch_tensors(stack);

            min = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(data_0);

            auto f = query_inference_function("Min");
            infer_.do_inference(f);
            infer_.check_output(0, min);

#endif

            if ( min->onnx_Min(data_0, min) != YNX_OK ) {
                yannx_panic("API: Min  return error!");
            }

            put_tensor(stack, min);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto data_0 = fetch_tensors(stack);


            if ( min->onnx_Min(data_0, min) != YNX_OK ) {
                yannx_panic("API: Min  return error!");
            }

            put_tensor(stack, min);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Min)
    };


    struct Floor : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Floor");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Floor(X, Y) != YNX_OK ) {
                yannx_panic("API: Floor  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Floor(X, Y) != YNX_OK ) {
                yannx_panic("API: Floor  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Floor)
    };


    struct Mean : NativeWord<TensorType> {
        tensor_t mean;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto data_0 = fetch_tensors(stack);

            mean = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(data_0);

            auto f = query_inference_function("Mean");
            infer_.do_inference(f);
            infer_.check_output(0, mean);

#endif

            if ( mean->onnx_Mean(data_0, mean) != YNX_OK ) {
                yannx_panic("API: Mean  return error!");
            }

            put_tensor(stack, mean);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto data_0 = fetch_tensors(stack);


            if ( mean->onnx_Mean(data_0, mean) != YNX_OK ) {
                yannx_panic("API: Mean  return error!");
            }

            put_tensor(stack, mean);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Mean)
    };


    struct Max : NativeWord<TensorType> {
        tensor_t max;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto data_0 = fetch_tensors(stack);

            max = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(data_0);

            auto f = query_inference_function("Max");
            infer_.do_inference(f);
            infer_.check_output(0, max);

#endif

            if ( max->onnx_Max(data_0, max) != YNX_OK ) {
                yannx_panic("API: Max  return error!");
            }

            put_tensor(stack, max);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto data_0 = fetch_tensors(stack);


            if ( max->onnx_Max(data_0, max) != YNX_OK ) {
                yannx_panic("API: Max  return error!");
            }

            put_tensor(stack, max);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Max)
    };


    struct Round : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Round");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Round(X, Y) != YNX_OK ) {
                yannx_panic("API: Round  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Round(X, Y) != YNX_OK ) {
                yannx_panic("API: Round  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Round)
    };


    struct Sigmoid : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Sigmoid");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Sigmoid(X, Y) != YNX_OK ) {
                yannx_panic("API: Sigmoid  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Sigmoid(X, Y) != YNX_OK ) {
                yannx_panic("API: Sigmoid  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sigmoid)
    };


    struct Relu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Relu");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Relu(X, Y) != YNX_OK ) {
                yannx_panic("API: Relu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Relu(X, Y) != YNX_OK ) {
                yannx_panic("API: Relu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Relu)
    };


    struct LogSoftmax : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("LogSoftmax");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_LogSoftmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: LogSoftmax  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_LogSoftmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: LogSoftmax  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LogSoftmax)
    };


    struct Ceil : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Ceil");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Ceil(X, Y) != YNX_OK ) {
                yannx_panic("API: Ceil  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Ceil(X, Y) != YNX_OK ) {
                yannx_panic("API: Ceil  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Ceil)
    };


    struct Log : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Log");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Log(input, output) != YNX_OK ) {
                yannx_panic("API: Log  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Log(input, output) != YNX_OK ) {
                yannx_panic("API: Log  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Log)
    };


    struct Neg : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Neg");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Neg(X, Y) != YNX_OK ) {
                yannx_panic("API: Neg  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Neg(X, Y) != YNX_OK ) {
                yannx_panic("API: Neg  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Neg)
    };


    struct Sub : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Sub");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_Sub(A, B, C) != YNX_OK ) {
                yannx_panic("API: Sub  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Sub(A, B, C) != YNX_OK ) {
                yannx_panic("API: Sub  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sub)
    };


    struct PRelu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto slope = fetch_tensor(stack);
            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(slope);
            infer_.new_input(X);

            auto f = query_inference_function("PRelu");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_PRelu(X, slope, Y) != YNX_OK ) {
                yannx_panic("API: PRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto slope = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_PRelu(X, slope, Y) != YNX_OK ) {
                yannx_panic("API: PRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(PRelu)
    };


    struct Add : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Add");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_Add(A, B, C) != YNX_OK ) {
                yannx_panic("API: Add  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Add(A, B, C) != YNX_OK ) {
                yannx_panic("API: Add  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Add)
    };


    struct Selu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto gamma = fetch_optional_float(stack);
            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( gamma.index() != 0) {
                infer_.new_attr("gamma", std::get<1>(gamma) );
            }
            if ( alpha.index() != 0) {
                infer_.new_attr("alpha", std::get<1>(alpha) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("Selu");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Selu(X, Y, alpha, gamma) != YNX_OK ) {
                yannx_panic("API: Selu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto gamma = fetch_optional_float(stack);
            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Selu(X, Y, alpha, gamma) != YNX_OK ) {
                yannx_panic("API: Selu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Selu)
    };


    struct Abs : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Abs");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Abs(X, Y) != YNX_OK ) {
                yannx_panic("API: Abs  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Abs(X, Y) != YNX_OK ) {
                yannx_panic("API: Abs  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Abs)
    };


    struct QLinearMatMul : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto y_zero_point = fetch_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto b_zero_point = fetch_tensor(stack);
            auto b_scale = fetch_tensor(stack);
            auto b = fetch_tensor(stack);
            auto a_zero_point = fetch_tensor(stack);
            auto a_scale = fetch_tensor(stack);
            auto a = fetch_tensor(stack);

            y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(y_zero_point);
            infer_.new_input(y_scale);
            infer_.new_input(b_zero_point);
            infer_.new_input(b_scale);
            infer_.new_input(b);
            infer_.new_input(a_zero_point);
            infer_.new_input(a_scale);
            infer_.new_input(a);

            auto f = query_inference_function("QLinearMatMul");
            infer_.do_inference(f);
            infer_.check_output(0, y);

#endif

            if ( a->onnx_QLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, y) != YNX_OK ) {
                yannx_panic("API: QLinearMatMul  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto y_zero_point = fetch_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto b_zero_point = fetch_tensor(stack);
            auto b_scale = fetch_tensor(stack);
            auto b = fetch_tensor(stack);
            auto a_zero_point = fetch_tensor(stack);
            auto a_scale = fetch_tensor(stack);
            auto a = fetch_tensor(stack);


            if ( a->onnx_QLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, y) != YNX_OK ) {
                yannx_panic("API: QLinearMatMul  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(QLinearMatMul)
    };


    struct Clip : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto max = fetch_optional_tensor(stack);
            auto min = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(max);
            infer_.new_input(min);
            infer_.new_input(input);

            auto f = query_inference_function("Clip");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Clip(input, min, max, output) != YNX_OK ) {
                yannx_panic("API: Clip  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto max = fetch_optional_tensor(stack);
            auto min = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Clip(input, min, max, output) != YNX_OK ) {
                yannx_panic("API: Clip  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Clip)
    };


    struct Einsum : NativeWord<TensorType> {
        tensor_t Output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto equation = fetch_string(stack);

            auto Inputs = fetch_tensors(stack);

            Output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            infer_.new_attr("equation", equation);

            infer_.new_input(Inputs);

            auto f = query_inference_function("Einsum");
            infer_.do_inference(f);
            infer_.check_output(0, Output);

#endif

            if ( Output->onnx_Einsum(Inputs, Output, equation) != YNX_OK ) {
                yannx_panic("API: Einsum  return error!");
            }

            put_tensor(stack, Output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto equation = fetch_string(stack);

            auto Inputs = fetch_tensors(stack);


            if ( Output->onnx_Einsum(Inputs, Output, equation) != YNX_OK ) {
                yannx_panic("API: Einsum  return error!");
            }

            put_tensor(stack, Output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Einsum)
    };


    struct Hardmax : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("Hardmax");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Hardmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Hardmax  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Hardmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Hardmax  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Hardmax)
    };


    struct Sqrt : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Sqrt");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Sqrt(X, Y) != YNX_OK ) {
                yannx_panic("API: Sqrt  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Sqrt(X, Y) != YNX_OK ) {
                yannx_panic("API: Sqrt  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sqrt)
    };


    struct Gemm : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto transB = fetch_optional_int(stack);
            auto transA = fetch_optional_int(stack);
            auto beta = fetch_optional_float(stack);
            auto alpha = fetch_optional_float(stack);

            auto C = fetch_optional_tensor(stack);
            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( transB.index() != 0) {
                infer_.new_attr("transB", std::get<1>(transB) );
            }
            if ( transA.index() != 0) {
                infer_.new_attr("transA", std::get<1>(transA) );
            }
            if ( beta.index() != 0) {
                infer_.new_attr("beta", std::get<1>(beta) );
            }
            if ( alpha.index() != 0) {
                infer_.new_attr("alpha", std::get<1>(alpha) );
            }

            infer_.new_input(C);
            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Gemm");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( A->onnx_Gemm(A, B, C, Y, alpha, beta, transA, transB) != YNX_OK ) {
                yannx_panic("API: Gemm  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto transB = fetch_optional_int(stack);
            auto transA = fetch_optional_int(stack);
            auto beta = fetch_optional_float(stack);
            auto alpha = fetch_optional_float(stack);

            auto C = fetch_optional_tensor(stack);
            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Gemm(A, B, C, Y, alpha, beta, transA, transB) != YNX_OK ) {
                yannx_panic("API: Gemm  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Gemm)
    };


    struct Cos : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Cos");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Cos(input, output) != YNX_OK ) {
                yannx_panic("API: Cos  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Cos(input, output) != YNX_OK ) {
                yannx_panic("API: Cos  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Cos)
    };


    struct Exp : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Exp");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Exp(input, output) != YNX_OK ) {
                yannx_panic("API: Exp  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Exp(input, output) != YNX_OK ) {
                yannx_panic("API: Exp  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Exp)
    };


    struct Tan : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Tan");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Tan(input, output) != YNX_OK ) {
                yannx_panic("API: Tan  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Tan(input, output) != YNX_OK ) {
                yannx_panic("API: Tan  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Tan)
    };


    struct Softmax : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("Softmax");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Softmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Softmax  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Softmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Softmax  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Softmax)
    };


    struct SoftmaxCrossEntropyLoss : NativeWord<TensorType> {
        tensor_t output;
        std::variant<void *, tensor_t> log_prob;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto reduction = fetch_optional_string(stack);
            auto ignore_index = fetch_optional_int(stack);

            auto weights = fetch_optional_tensor(stack);
            auto labels = fetch_tensor(stack);
            auto scores = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(2);
            if ( reduction.index() != 0) {
                infer_.new_attr("reduction", std::get<1>(reduction) );
            }
            if ( ignore_index.index() != 0) {
                infer_.new_attr("ignore_index", std::get<1>(ignore_index) );
            }

            infer_.new_input(weights);
            infer_.new_input(labels);
            infer_.new_input(scores);

            auto f = query_inference_function("SoftmaxCrossEntropyLoss");
            infer_.do_inference(f);
            infer_.check_output(0, output);
            infer_.check_output(1, log_prob);

#endif

            if ( scores->onnx_SoftmaxCrossEntropyLoss(scores, labels, weights, output, log_prob, ignore_index, reduction) != YNX_OK ) {
                yannx_panic("API: SoftmaxCrossEntropyLoss  return error!");
            }

            put_tensor(stack, output);
            put_optional_tensor(stack, log_prob);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto reduction = fetch_optional_string(stack);
            auto ignore_index = fetch_optional_int(stack);

            auto weights = fetch_optional_tensor(stack);
            auto labels = fetch_tensor(stack);
            auto scores = fetch_tensor(stack);


            if ( scores->onnx_SoftmaxCrossEntropyLoss(scores, labels, weights, output, log_prob, ignore_index, reduction) != YNX_OK ) {
                yannx_panic("API: SoftmaxCrossEntropyLoss  return error!");
            }

            put_tensor(stack, output);
            put_optional_tensor(stack, log_prob);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SoftmaxCrossEntropyLoss)
    };


    struct Softsign : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Softsign");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Softsign(input, output) != YNX_OK ) {
                yannx_panic("API: Softsign  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Softsign(input, output) != YNX_OK ) {
                yannx_panic("API: Softsign  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Softsign)
    };


    struct Sum : NativeWord<TensorType> {
        tensor_t sum;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto data_0 = fetch_tensors(stack);

            sum = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(data_0);

            auto f = query_inference_function("Sum");
            infer_.do_inference(f);
            infer_.check_output(0, sum);

#endif

            if ( sum->onnx_Sum(data_0, sum) != YNX_OK ) {
                yannx_panic("API: Sum  return error!");
            }

            put_tensor(stack, sum);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto data_0 = fetch_tensors(stack);


            if ( sum->onnx_Sum(data_0, sum) != YNX_OK ) {
                yannx_panic("API: Sum  return error!");
            }

            put_tensor(stack, sum);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sum)
    };


    struct Sinh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Sinh");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Sinh(input, output) != YNX_OK ) {
                yannx_panic("API: Sinh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Sinh(input, output) != YNX_OK ) {
                yannx_panic("API: Sinh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sinh)
    };


    struct Tanh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Tanh");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Tanh(input, output) != YNX_OK ) {
                yannx_panic("API: Tanh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Tanh(input, output) != YNX_OK ) {
                yannx_panic("API: Tanh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Tanh)
    };


    struct TopK : NativeWord<TensorType> {
        tensor_t Values;
        tensor_t Indices;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto sorted = fetch_optional_int(stack);
            auto largest = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto K = fetch_tensor(stack);
            auto X = fetch_tensor(stack);

            Values = TensorType::create_undefined_user_tensor();
            Indices = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(2);
            if ( sorted.index() != 0) {
                infer_.new_attr("sorted", std::get<1>(sorted) );
            }
            if ( largest.index() != 0) {
                infer_.new_attr("largest", std::get<1>(largest) );
            }
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(K);
            infer_.new_input(X);

            auto f = query_inference_function("TopK");
            infer_.do_inference(f);
            infer_.check_output(0, Values);
            infer_.check_output(1, Indices);

#endif

            if ( X->onnx_TopK(X, K, Values, Indices, axis, largest, sorted) != YNX_OK ) {
                yannx_panic("API: TopK  return error!");
            }

            put_tensor(stack, Values);
            put_tensor(stack, Indices);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto sorted = fetch_optional_int(stack);
            auto largest = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto K = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_TopK(X, K, Values, Indices, axis, largest, sorted) != YNX_OK ) {
                yannx_panic("API: TopK  return error!");
            }

            put_tensor(stack, Values);
            put_tensor(stack, Indices);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(TopK)
    };


    struct Acos : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Acos");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Acos(input, output) != YNX_OK ) {
                yannx_panic("API: Acos  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Acos(input, output) != YNX_OK ) {
                yannx_panic("API: Acos  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Acos)
    };


    struct Asin : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Asin");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Asin(input, output) != YNX_OK ) {
                yannx_panic("API: Asin  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Asin(input, output) != YNX_OK ) {
                yannx_panic("API: Asin  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Asin)
    };


    struct Atan : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Atan");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Atan(input, output) != YNX_OK ) {
                yannx_panic("API: Atan  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Atan(input, output) != YNX_OK ) {
                yannx_panic("API: Atan  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Atan)
    };


    struct Sign : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Sign");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Sign(input, output) != YNX_OK ) {
                yannx_panic("API: Sign  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Sign(input, output) != YNX_OK ) {
                yannx_panic("API: Sign  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sign)
    };


    struct Sin : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Sin");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Sin(input, output) != YNX_OK ) {
                yannx_panic("API: Sin  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Sin(input, output) != YNX_OK ) {
                yannx_panic("API: Sin  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sin)
    };


    struct MatMul : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("MatMul");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( A->onnx_MatMul(A, B, Y) != YNX_OK ) {
                yannx_panic("API: MatMul  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_MatMul(A, B, Y) != YNX_OK ) {
                yannx_panic("API: MatMul  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MatMul)
    };


    struct Expand : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto shape = fetch_tensor(stack);
            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(shape);
            infer_.new_input(input);

            auto f = query_inference_function("Expand");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Expand(input, shape, output) != YNX_OK ) {
                yannx_panic("API: Expand  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto shape = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Expand(input, shape, output) != YNX_OK ) {
                yannx_panic("API: Expand  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Expand)
    };


    struct Elu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( alpha.index() != 0) {
                infer_.new_attr("alpha", std::get<1>(alpha) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("Elu");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Elu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: Elu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Elu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: Elu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Elu)
    };


    struct Cosh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Cosh");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Cosh(input, output) != YNX_OK ) {
                yannx_panic("API: Cosh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Cosh(input, output) != YNX_OK ) {
                yannx_panic("API: Cosh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Cosh)
    };


    struct Asinh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Asinh");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Asinh(input, output) != YNX_OK ) {
                yannx_panic("API: Asinh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Asinh(input, output) != YNX_OK ) {
                yannx_panic("API: Asinh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Asinh)
    };


    struct Acosh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Acosh");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Acosh(input, output) != YNX_OK ) {
                yannx_panic("API: Acosh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Acosh(input, output) != YNX_OK ) {
                yannx_panic("API: Acosh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Acosh)
    };


    struct Atanh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Atanh");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Atanh(input, output) != YNX_OK ) {
                yannx_panic("API: Atanh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Atanh(input, output) != YNX_OK ) {
                yannx_panic("API: Atanh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Atanh)
    };


    struct Erf : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Erf");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Erf(input, output) != YNX_OK ) {
                yannx_panic("API: Erf  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Erf(input, output) != YNX_OK ) {
                yannx_panic("API: Erf  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Erf)
    };


    struct Mod : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto fmod = fetch_optional_int(stack);

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            C = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( fmod.index() != 0) {
                infer_.new_attr("fmod", std::get<1>(fmod) );
            }

            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("Mod");
            infer_.do_inference(f);
            infer_.check_output(0, C);

#endif

            if ( A->onnx_Mod(A, B, C, fmod) != YNX_OK ) {
                yannx_panic("API: Mod  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto fmod = fetch_optional_int(stack);

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Mod(A, B, C, fmod) != YNX_OK ) {
                yannx_panic("API: Mod  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Mod)
    };


    struct ThresholdedRelu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( alpha.index() != 0) {
                infer_.new_attr("alpha", std::get<1>(alpha) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("ThresholdedRelu");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_ThresholdedRelu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: ThresholdedRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_ThresholdedRelu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: ThresholdedRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ThresholdedRelu)
    };


    struct MatMulInteger : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto b_zero_point = fetch_optional_tensor(stack);
            auto a_zero_point = fetch_optional_tensor(stack);
            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(b_zero_point);
            infer_.new_input(a_zero_point);
            infer_.new_input(B);
            infer_.new_input(A);

            auto f = query_inference_function("MatMulInteger");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( A->onnx_MatMulInteger(A, B, a_zero_point, b_zero_point, Y) != YNX_OK ) {
                yannx_panic("API: MatMulInteger  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto b_zero_point = fetch_optional_tensor(stack);
            auto a_zero_point = fetch_optional_tensor(stack);
            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_MatMulInteger(A, B, a_zero_point, b_zero_point, Y) != YNX_OK ) {
                yannx_panic("API: MatMulInteger  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MatMulInteger)
    };


    struct Celu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( alpha.index() != 0) {
                infer_.new_attr("alpha", std::get<1>(alpha) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("Celu");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Celu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: Celu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Celu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: Celu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Celu)
    };


    struct CumSum : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto reverse = fetch_optional_int(stack);
            auto exclusive = fetch_optional_int(stack);

            auto axis = fetch_tensor(stack);
            auto x = fetch_tensor(stack);

            y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( reverse.index() != 0) {
                infer_.new_attr("reverse", std::get<1>(reverse) );
            }
            if ( exclusive.index() != 0) {
                infer_.new_attr("exclusive", std::get<1>(exclusive) );
            }

            infer_.new_input(axis);
            infer_.new_input(x);

            auto f = query_inference_function("CumSum");
            infer_.do_inference(f);
            infer_.check_output(0, y);

#endif

            if ( x->onnx_CumSum(x, axis, y, exclusive, reverse) != YNX_OK ) {
                yannx_panic("API: CumSum  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto reverse = fetch_optional_int(stack);
            auto exclusive = fetch_optional_int(stack);

            auto axis = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_CumSum(x, axis, y, exclusive, reverse) != YNX_OK ) {
                yannx_panic("API: CumSum  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(CumSum)
    };


    struct Softplus : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Softplus");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Softplus(X, Y) != YNX_OK ) {
                yannx_panic("API: Softplus  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Softplus(X, Y) != YNX_OK ) {
                yannx_panic("API: Softplus  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Softplus)
    };


    struct NegativeLogLikelihoodLoss : NativeWord<TensorType> {
        tensor_t loss;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto reduction = fetch_optional_string(stack);
            auto ignore_index = fetch_optional_int(stack);

            auto weight = fetch_optional_tensor(stack);
            auto target = fetch_tensor(stack);
            auto input = fetch_tensor(stack);

            loss = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( reduction.index() != 0) {
                infer_.new_attr("reduction", std::get<1>(reduction) );
            }
            if ( ignore_index.index() != 0) {
                infer_.new_attr("ignore_index", std::get<1>(ignore_index) );
            }

            infer_.new_input(weight);
            infer_.new_input(target);
            infer_.new_input(input);

            auto f = query_inference_function("NegativeLogLikelihoodLoss");
            infer_.do_inference(f);
            infer_.check_output(0, loss);

#endif

            if ( input->onnx_NegativeLogLikelihoodLoss(input, target, weight, loss, ignore_index, reduction) != YNX_OK ) {
                yannx_panic("API: NegativeLogLikelihoodLoss  return error!");
            }

            put_tensor(stack, loss);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto reduction = fetch_optional_string(stack);
            auto ignore_index = fetch_optional_int(stack);

            auto weight = fetch_optional_tensor(stack);
            auto target = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_NegativeLogLikelihoodLoss(input, target, weight, loss, ignore_index, reduction) != YNX_OK ) {
                yannx_panic("API: NegativeLogLikelihoodLoss  return error!");
            }

            put_tensor(stack, loss);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(NegativeLogLikelihoodLoss)
    };


    struct Det : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("Det");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Det(X, Y) != YNX_OK ) {
                yannx_panic("API: Det  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_Det(X, Y) != YNX_OK ) {
                yannx_panic("API: Det  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Det)
    };


    struct HardSwish : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("HardSwish");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_HardSwish(X, Y) != YNX_OK ) {
                yannx_panic("API: HardSwish  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_HardSwish(X, Y) != YNX_OK ) {
                yannx_panic("API: HardSwish  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(HardSwish)
    };

}
namespace nn {

    struct LRN : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto size = fetch_int(stack);
            auto bias = fetch_optional_float(stack);
            auto beta = fetch_optional_float(stack);
            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            infer_.new_attr("size", size);
            if ( bias.index() != 0) {
                infer_.new_attr("bias", std::get<1>(bias) );
            }
            if ( beta.index() != 0) {
                infer_.new_attr("beta", std::get<1>(beta) );
            }
            if ( alpha.index() != 0) {
                infer_.new_attr("alpha", std::get<1>(alpha) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("LRN");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_LRN(X, Y, alpha, beta, bias, size) != YNX_OK ) {
                yannx_panic("API: LRN  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto size = fetch_int(stack);
            auto bias = fetch_optional_float(stack);
            auto beta = fetch_optional_float(stack);
            auto alpha = fetch_optional_float(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_LRN(X, Y, alpha, beta, bias, size) != YNX_OK ) {
                yannx_panic("API: LRN  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LRN)
    };


    struct LpPool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto p = fetch_optional_int(stack);
            auto kernel_shape = fetch_ints(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( strides.index() != 0) {
                infer_.new_attr("strides", std::get<1>(strides) );
            }
            if ( pads.index() != 0) {
                infer_.new_attr("pads", std::get<1>(pads) );
            }
            if ( p.index() != 0) {
                infer_.new_attr("p", std::get<1>(p) );
            }
            infer_.new_attr("kernel_shape", kernel_shape);
            if ( auto_pad.index() != 0) {
                infer_.new_attr("auto_pad", std::get<1>(auto_pad) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("LpPool");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_LpPool(X, Y, auto_pad, kernel_shape, p, pads, strides) != YNX_OK ) {
                yannx_panic("API: LpPool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto p = fetch_optional_int(stack);
            auto kernel_shape = fetch_ints(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_LpPool(X, Y, auto_pad, kernel_shape, p, pads, strides) != YNX_OK ) {
                yannx_panic("API: LpPool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LpPool)
    };


    struct Dropout : NativeWord<TensorType> {
        tensor_t output;
        std::variant<void *, tensor_t> mask;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto seed = fetch_optional_int(stack);

            auto training_mode = fetch_optional_tensor(stack);
            auto ratio = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(2);
            if ( seed.index() != 0) {
                infer_.new_attr("seed", std::get<1>(seed) );
            }

            infer_.new_input(training_mode);
            infer_.new_input(ratio);
            infer_.new_input(data);

            auto f = query_inference_function("Dropout");
            infer_.do_inference(f);
            infer_.check_output(0, output);
            infer_.check_output(1, mask);

#endif

            if ( data->onnx_Dropout(data, ratio, training_mode, output, mask, seed) != YNX_OK ) {
                yannx_panic("API: Dropout  return error!");
            }

            put_tensor(stack, output);
            put_optional_tensor(stack, mask);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto seed = fetch_optional_int(stack);

            auto training_mode = fetch_optional_tensor(stack);
            auto ratio = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Dropout(data, ratio, training_mode, output, mask, seed) != YNX_OK ) {
                yannx_panic("API: Dropout  return error!");
            }

            put_tensor(stack, output);
            put_optional_tensor(stack, mask);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Dropout)
    };


    struct MaxPool : NativeWord<TensorType> {
        tensor_t Y;
        std::variant<void *, tensor_t> Indices;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto strides = fetch_optional_ints(stack);
            auto storage_order = fetch_optional_int(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_ints(stack);
            auto dilations = fetch_optional_ints(stack);
            auto ceil_mode = fetch_optional_int(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(2);
            if ( strides.index() != 0) {
                infer_.new_attr("strides", std::get<1>(strides) );
            }
            if ( storage_order.index() != 0) {
                infer_.new_attr("storage_order", std::get<1>(storage_order) );
            }
            if ( pads.index() != 0) {
                infer_.new_attr("pads", std::get<1>(pads) );
            }
            infer_.new_attr("kernel_shape", kernel_shape);
            if ( dilations.index() != 0) {
                infer_.new_attr("dilations", std::get<1>(dilations) );
            }
            if ( ceil_mode.index() != 0) {
                infer_.new_attr("ceil_mode", std::get<1>(ceil_mode) );
            }
            if ( auto_pad.index() != 0) {
                infer_.new_attr("auto_pad", std::get<1>(auto_pad) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("MaxPool");
            infer_.do_inference(f);
            infer_.check_output(0, Y);
            infer_.check_output(1, Indices);

#endif

            if ( X->onnx_MaxPool(X, Y, Indices, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides) != YNX_OK ) {
                yannx_panic("API: MaxPool  return error!");
            }

            put_tensor(stack, Y);
            put_optional_tensor(stack, Indices);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto strides = fetch_optional_ints(stack);
            auto storage_order = fetch_optional_int(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_ints(stack);
            auto dilations = fetch_optional_ints(stack);
            auto ceil_mode = fetch_optional_int(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_MaxPool(X, Y, Indices, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides) != YNX_OK ) {
                yannx_panic("API: MaxPool  return error!");
            }

            put_tensor(stack, Y);
            put_optional_tensor(stack, Indices);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MaxPool)
    };


    struct GlobalLpPool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto p = fetch_optional_int(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( p.index() != 0) {
                infer_.new_attr("p", std::get<1>(p) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("GlobalLpPool");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_GlobalLpPool(X, Y, p) != YNX_OK ) {
                yannx_panic("API: GlobalLpPool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto p = fetch_optional_int(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_GlobalLpPool(X, Y, p) != YNX_OK ) {
                yannx_panic("API: GlobalLpPool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GlobalLpPool)
    };


    struct LpNormalization : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto p = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( p.index() != 0) {
                infer_.new_attr("p", std::get<1>(p) );
            }
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("LpNormalization");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_LpNormalization(input, output, axis, p) != YNX_OK ) {
                yannx_panic("API: LpNormalization  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto p = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_LpNormalization(input, output, axis, p) != YNX_OK ) {
                yannx_panic("API: LpNormalization  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LpNormalization)
    };


    struct Conv : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_optional_ints(stack);
            auto group = fetch_optional_int(stack);
            auto dilations = fetch_optional_ints(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto B = fetch_optional_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( strides.index() != 0) {
                infer_.new_attr("strides", std::get<1>(strides) );
            }
            if ( pads.index() != 0) {
                infer_.new_attr("pads", std::get<1>(pads) );
            }
            if ( kernel_shape.index() != 0) {
                infer_.new_attr("kernel_shape", std::get<1>(kernel_shape) );
            }
            if ( group.index() != 0) {
                infer_.new_attr("group", std::get<1>(group) );
            }
            if ( dilations.index() != 0) {
                infer_.new_attr("dilations", std::get<1>(dilations) );
            }
            if ( auto_pad.index() != 0) {
                infer_.new_attr("auto_pad", std::get<1>(auto_pad) );
            }

            infer_.new_input(B);
            infer_.new_input(W);
            infer_.new_input(X);

            auto f = query_inference_function("Conv");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Conv(X, W, B, Y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: Conv  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_optional_ints(stack);
            auto group = fetch_optional_int(stack);
            auto dilations = fetch_optional_ints(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto B = fetch_optional_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_Conv(X, W, B, Y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: Conv  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Conv)
    };


    struct GlobalMaxPool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("GlobalMaxPool");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_GlobalMaxPool(X, Y) != YNX_OK ) {
                yannx_panic("API: GlobalMaxPool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_GlobalMaxPool(X, Y) != YNX_OK ) {
                yannx_panic("API: GlobalMaxPool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GlobalMaxPool)
    };


    struct MaxUnpool : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_ints(stack);

            auto output_shape = fetch_optional_tensor(stack);
            auto I = fetch_tensor(stack);
            auto X = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( strides.index() != 0) {
                infer_.new_attr("strides", std::get<1>(strides) );
            }
            if ( pads.index() != 0) {
                infer_.new_attr("pads", std::get<1>(pads) );
            }
            infer_.new_attr("kernel_shape", kernel_shape);

            infer_.new_input(output_shape);
            infer_.new_input(I);
            infer_.new_input(X);

            auto f = query_inference_function("MaxUnpool");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( X->onnx_MaxUnpool(X, I, output_shape, output, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: MaxUnpool  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_ints(stack);

            auto output_shape = fetch_optional_tensor(stack);
            auto I = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_MaxUnpool(X, I, output_shape, output, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: MaxUnpool  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MaxUnpool)
    };


    struct AveragePool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_ints(stack);
            auto count_include_pad = fetch_optional_int(stack);
            auto ceil_mode = fetch_optional_int(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( strides.index() != 0) {
                infer_.new_attr("strides", std::get<1>(strides) );
            }
            if ( pads.index() != 0) {
                infer_.new_attr("pads", std::get<1>(pads) );
            }
            infer_.new_attr("kernel_shape", kernel_shape);
            if ( count_include_pad.index() != 0) {
                infer_.new_attr("count_include_pad", std::get<1>(count_include_pad) );
            }
            if ( ceil_mode.index() != 0) {
                infer_.new_attr("ceil_mode", std::get<1>(ceil_mode) );
            }
            if ( auto_pad.index() != 0) {
                infer_.new_attr("auto_pad", std::get<1>(auto_pad) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("AveragePool");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_AveragePool(X, Y, auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: AveragePool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_ints(stack);
            auto count_include_pad = fetch_optional_int(stack);
            auto ceil_mode = fetch_optional_int(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_AveragePool(X, Y, auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: AveragePool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(AveragePool)
    };


    struct InstanceNormalization : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto epsilon = fetch_optional_float(stack);

            auto B = fetch_tensor(stack);
            auto scale = fetch_tensor(stack);
            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( epsilon.index() != 0) {
                infer_.new_attr("epsilon", std::get<1>(epsilon) );
            }

            infer_.new_input(B);
            infer_.new_input(scale);
            infer_.new_input(input);

            auto f = query_inference_function("InstanceNormalization");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_InstanceNormalization(input, scale, B, output, epsilon) != YNX_OK ) {
                yannx_panic("API: InstanceNormalization  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto epsilon = fetch_optional_float(stack);

            auto B = fetch_tensor(stack);
            auto scale = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_InstanceNormalization(input, scale, B, output, epsilon) != YNX_OK ) {
                yannx_panic("API: InstanceNormalization  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(InstanceNormalization)
    };


    struct Flatten : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("Flatten");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Flatten(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Flatten  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Flatten(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Flatten  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Flatten)
    };


    struct GlobalAveragePool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("GlobalAveragePool");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_GlobalAveragePool(X, Y) != YNX_OK ) {
                yannx_panic("API: GlobalAveragePool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_GlobalAveragePool(X, Y) != YNX_OK ) {
                yannx_panic("API: GlobalAveragePool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GlobalAveragePool)
    };


    struct MaxRoiPool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto spatial_scale = fetch_optional_float(stack);
            auto pooled_shape = fetch_ints(stack);

            auto rois = fetch_tensor(stack);
            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( spatial_scale.index() != 0) {
                infer_.new_attr("spatial_scale", std::get<1>(spatial_scale) );
            }
            infer_.new_attr("pooled_shape", pooled_shape);

            infer_.new_input(rois);
            infer_.new_input(X);

            auto f = query_inference_function("MaxRoiPool");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_MaxRoiPool(X, rois, Y, pooled_shape, spatial_scale) != YNX_OK ) {
                yannx_panic("API: MaxRoiPool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto spatial_scale = fetch_optional_float(stack);
            auto pooled_shape = fetch_ints(stack);

            auto rois = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_MaxRoiPool(X, rois, Y, pooled_shape, spatial_scale) != YNX_OK ) {
                yannx_panic("API: MaxRoiPool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MaxRoiPool)
    };


    struct BatchNormalization : NativeWord<TensorType> {
        tensor_t Y;
        std::variant<void *, tensor_t> running_mean;
        std::variant<void *, tensor_t> running_var;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto training_mode = fetch_optional_int(stack);
            auto momentum = fetch_optional_float(stack);
            auto epsilon = fetch_optional_float(stack);

            auto input_var = fetch_tensor(stack);
            auto input_mean = fetch_tensor(stack);
            auto B = fetch_tensor(stack);
            auto scale = fetch_tensor(stack);
            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(3);
            if ( training_mode.index() != 0) {
                infer_.new_attr("training_mode", std::get<1>(training_mode) );
            }
            if ( momentum.index() != 0) {
                infer_.new_attr("momentum", std::get<1>(momentum) );
            }
            if ( epsilon.index() != 0) {
                infer_.new_attr("epsilon", std::get<1>(epsilon) );
            }

            infer_.new_input(input_var);
            infer_.new_input(input_mean);
            infer_.new_input(B);
            infer_.new_input(scale);
            infer_.new_input(X);

            auto f = query_inference_function("BatchNormalization");
            infer_.do_inference(f);
            infer_.check_output(0, Y);
            infer_.check_output(1, running_mean);
            infer_.check_output(2, running_var);

#endif

            if ( X->onnx_BatchNormalization(X, scale, B, input_mean, input_var, Y, running_mean, running_var, epsilon, momentum, training_mode) != YNX_OK ) {
                yannx_panic("API: BatchNormalization  return error!");
            }

            put_tensor(stack, Y);
            put_optional_tensor(stack, running_mean);
            put_optional_tensor(stack, running_var);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto training_mode = fetch_optional_int(stack);
            auto momentum = fetch_optional_float(stack);
            auto epsilon = fetch_optional_float(stack);

            auto input_var = fetch_tensor(stack);
            auto input_mean = fetch_tensor(stack);
            auto B = fetch_tensor(stack);
            auto scale = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_BatchNormalization(X, scale, B, input_mean, input_var, Y, running_mean, running_var, epsilon, momentum, training_mode) != YNX_OK ) {
                yannx_panic("API: BatchNormalization  return error!");
            }

            put_tensor(stack, Y);
            put_optional_tensor(stack, running_mean);
            put_optional_tensor(stack, running_var);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(BatchNormalization)
    };


    struct StringNormalizer : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto stopwords = fetch_optional_strings(stack);
            auto locale = fetch_optional_string(stack);
            auto is_case_sensitive = fetch_optional_int(stack);
            auto case_change_action = fetch_optional_string(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( stopwords.index() != 0) {
                infer_.new_attr("stopwords", std::get<1>(stopwords) );
            }
            if ( locale.index() != 0) {
                infer_.new_attr("locale", std::get<1>(locale) );
            }
            if ( is_case_sensitive.index() != 0) {
                infer_.new_attr("is_case_sensitive", std::get<1>(is_case_sensitive) );
            }
            if ( case_change_action.index() != 0) {
                infer_.new_attr("case_change_action", std::get<1>(case_change_action) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("StringNormalizer");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_StringNormalizer(X, Y, case_change_action, is_case_sensitive, locale, stopwords) != YNX_OK ) {
                yannx_panic("API: StringNormalizer  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto stopwords = fetch_optional_strings(stack);
            auto locale = fetch_optional_string(stack);
            auto is_case_sensitive = fetch_optional_int(stack);
            auto case_change_action = fetch_optional_string(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_StringNormalizer(X, Y, case_change_action, is_case_sensitive, locale, stopwords) != YNX_OK ) {
                yannx_panic("API: StringNormalizer  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(StringNormalizer)
    };


    struct Shrink : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto lambd = fetch_optional_float(stack);
            auto bias = fetch_optional_float(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( lambd.index() != 0) {
                infer_.new_attr("lambd", std::get<1>(lambd) );
            }
            if ( bias.index() != 0) {
                infer_.new_attr("bias", std::get<1>(bias) );
            }

            infer_.new_input(input);

            auto f = query_inference_function("Shrink");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Shrink(input, output, bias, lambd) != YNX_OK ) {
                yannx_panic("API: Shrink  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto lambd = fetch_optional_float(stack);
            auto bias = fetch_optional_float(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Shrink(input, output, bias, lambd) != YNX_OK ) {
                yannx_panic("API: Shrink  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Shrink)
    };


    struct MeanVarianceNormalization : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axes = fetch_optional_ints(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axes.index() != 0) {
                infer_.new_attr("axes", std::get<1>(axes) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("MeanVarianceNormalization");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_MeanVarianceNormalization(X, Y, axes) != YNX_OK ) {
                yannx_panic("API: MeanVarianceNormalization  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axes = fetch_optional_ints(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_MeanVarianceNormalization(X, Y, axes) != YNX_OK ) {
                yannx_panic("API: MeanVarianceNormalization  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MeanVarianceNormalization)
    };


    struct ConvInteger : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_optional_ints(stack);
            auto group = fetch_optional_int(stack);
            auto dilations = fetch_optional_ints(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto w_zero_point = fetch_optional_tensor(stack);
            auto x_zero_point = fetch_optional_tensor(stack);
            auto w = fetch_tensor(stack);
            auto x = fetch_tensor(stack);

            y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( strides.index() != 0) {
                infer_.new_attr("strides", std::get<1>(strides) );
            }
            if ( pads.index() != 0) {
                infer_.new_attr("pads", std::get<1>(pads) );
            }
            if ( kernel_shape.index() != 0) {
                infer_.new_attr("kernel_shape", std::get<1>(kernel_shape) );
            }
            if ( group.index() != 0) {
                infer_.new_attr("group", std::get<1>(group) );
            }
            if ( dilations.index() != 0) {
                infer_.new_attr("dilations", std::get<1>(dilations) );
            }
            if ( auto_pad.index() != 0) {
                infer_.new_attr("auto_pad", std::get<1>(auto_pad) );
            }

            infer_.new_input(w_zero_point);
            infer_.new_input(x_zero_point);
            infer_.new_input(w);
            infer_.new_input(x);

            auto f = query_inference_function("ConvInteger");
            infer_.do_inference(f);
            infer_.check_output(0, y);

#endif

            if ( x->onnx_ConvInteger(x, w, x_zero_point, w_zero_point, y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: ConvInteger  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_optional_ints(stack);
            auto group = fetch_optional_int(stack);
            auto dilations = fetch_optional_ints(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto w_zero_point = fetch_optional_tensor(stack);
            auto x_zero_point = fetch_optional_tensor(stack);
            auto w = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_ConvInteger(x, w, x_zero_point, w_zero_point, y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: ConvInteger  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ConvInteger)
    };


    struct QLinearConv : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_optional_ints(stack);
            auto group = fetch_optional_int(stack);
            auto dilations = fetch_optional_ints(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto B = fetch_optional_tensor(stack);
            auto y_zero_point = fetch_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto w_zero_point = fetch_tensor(stack);
            auto w_scale = fetch_tensor(stack);
            auto w = fetch_tensor(stack);
            auto x_zero_point = fetch_tensor(stack);
            auto x_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);

            y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( strides.index() != 0) {
                infer_.new_attr("strides", std::get<1>(strides) );
            }
            if ( pads.index() != 0) {
                infer_.new_attr("pads", std::get<1>(pads) );
            }
            if ( kernel_shape.index() != 0) {
                infer_.new_attr("kernel_shape", std::get<1>(kernel_shape) );
            }
            if ( group.index() != 0) {
                infer_.new_attr("group", std::get<1>(group) );
            }
            if ( dilations.index() != 0) {
                infer_.new_attr("dilations", std::get<1>(dilations) );
            }
            if ( auto_pad.index() != 0) {
                infer_.new_attr("auto_pad", std::get<1>(auto_pad) );
            }

            infer_.new_input(B);
            infer_.new_input(y_zero_point);
            infer_.new_input(y_scale);
            infer_.new_input(w_zero_point);
            infer_.new_input(w_scale);
            infer_.new_input(w);
            infer_.new_input(x_zero_point);
            infer_.new_input(x_scale);
            infer_.new_input(x);

            auto f = query_inference_function("QLinearConv");
            infer_.do_inference(f);
            infer_.check_output(0, y);

#endif

            if ( x->onnx_QLinearConv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: QLinearConv  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto kernel_shape = fetch_optional_ints(stack);
            auto group = fetch_optional_int(stack);
            auto dilations = fetch_optional_ints(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto B = fetch_optional_tensor(stack);
            auto y_zero_point = fetch_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto w_zero_point = fetch_tensor(stack);
            auto w_scale = fetch_tensor(stack);
            auto w = fetch_tensor(stack);
            auto x_zero_point = fetch_tensor(stack);
            auto x_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_QLinearConv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: QLinearConv  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(QLinearConv)
    };


    struct ConvTranspose : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto output_shape = fetch_optional_ints(stack);
            auto output_padding = fetch_optional_ints(stack);
            auto kernel_shape = fetch_optional_ints(stack);
            auto group = fetch_optional_int(stack);
            auto dilations = fetch_optional_ints(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto B = fetch_optional_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( strides.index() != 0) {
                infer_.new_attr("strides", std::get<1>(strides) );
            }
            if ( pads.index() != 0) {
                infer_.new_attr("pads", std::get<1>(pads) );
            }
            if ( output_shape.index() != 0) {
                infer_.new_attr("output_shape", std::get<1>(output_shape) );
            }
            if ( output_padding.index() != 0) {
                infer_.new_attr("output_padding", std::get<1>(output_padding) );
            }
            if ( kernel_shape.index() != 0) {
                infer_.new_attr("kernel_shape", std::get<1>(kernel_shape) );
            }
            if ( group.index() != 0) {
                infer_.new_attr("group", std::get<1>(group) );
            }
            if ( dilations.index() != 0) {
                infer_.new_attr("dilations", std::get<1>(dilations) );
            }
            if ( auto_pad.index() != 0) {
                infer_.new_attr("auto_pad", std::get<1>(auto_pad) );
            }

            infer_.new_input(B);
            infer_.new_input(W);
            infer_.new_input(X);

            auto f = query_inference_function("ConvTranspose");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_ConvTranspose(X, W, B, Y, auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: ConvTranspose  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto strides = fetch_optional_ints(stack);
            auto pads = fetch_optional_ints(stack);
            auto output_shape = fetch_optional_ints(stack);
            auto output_padding = fetch_optional_ints(stack);
            auto kernel_shape = fetch_optional_ints(stack);
            auto group = fetch_optional_int(stack);
            auto dilations = fetch_optional_ints(stack);
            auto auto_pad = fetch_optional_string(stack);

            auto B = fetch_optional_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_ConvTranspose(X, W, B, Y, auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: ConvTranspose  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ConvTranspose)
    };


    struct TfIdfVectorizer : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto weights = fetch_optional_floats(stack);
            auto pool_strings = fetch_optional_strings(stack);
            auto pool_int64s = fetch_optional_ints(stack);
            auto ngram_indexes = fetch_ints(stack);
            auto ngram_counts = fetch_ints(stack);
            auto mode = fetch_string(stack);
            auto min_gram_length = fetch_int(stack);
            auto max_skip_count = fetch_int(stack);
            auto max_gram_length = fetch_int(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( weights.index() != 0) {
                infer_.new_attr("weights", std::get<1>(weights) );
            }
            if ( pool_strings.index() != 0) {
                infer_.new_attr("pool_strings", std::get<1>(pool_strings) );
            }
            if ( pool_int64s.index() != 0) {
                infer_.new_attr("pool_int64s", std::get<1>(pool_int64s) );
            }
            infer_.new_attr("ngram_indexes", ngram_indexes);
            infer_.new_attr("ngram_counts", ngram_counts);
            infer_.new_attr("mode", mode);
            infer_.new_attr("min_gram_length", min_gram_length);
            infer_.new_attr("max_skip_count", max_skip_count);
            infer_.new_attr("max_gram_length", max_gram_length);

            infer_.new_input(X);

            auto f = query_inference_function("TfIdfVectorizer");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_TfIdfVectorizer(X, Y, max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings, weights) != YNX_OK ) {
                yannx_panic("API: TfIdfVectorizer  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto weights = fetch_optional_floats(stack);
            auto pool_strings = fetch_optional_strings(stack);
            auto pool_int64s = fetch_optional_ints(stack);
            auto ngram_indexes = fetch_ints(stack);
            auto ngram_counts = fetch_ints(stack);
            auto mode = fetch_string(stack);
            auto min_gram_length = fetch_int(stack);
            auto max_skip_count = fetch_int(stack);
            auto max_gram_length = fetch_int(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_TfIdfVectorizer(X, Y, max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings, weights) != YNX_OK ) {
                yannx_panic("API: TfIdfVectorizer  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(TfIdfVectorizer)
    };

}
namespace object_detection {

    struct RoiAlign : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto spatial_scale = fetch_optional_float(stack);
            auto sampling_ratio = fetch_optional_int(stack);
            auto output_width = fetch_optional_int(stack);
            auto output_height = fetch_optional_int(stack);
            auto mode = fetch_optional_string(stack);

            auto batch_indices = fetch_tensor(stack);
            auto rois = fetch_tensor(stack);
            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( spatial_scale.index() != 0) {
                infer_.new_attr("spatial_scale", std::get<1>(spatial_scale) );
            }
            if ( sampling_ratio.index() != 0) {
                infer_.new_attr("sampling_ratio", std::get<1>(sampling_ratio) );
            }
            if ( output_width.index() != 0) {
                infer_.new_attr("output_width", std::get<1>(output_width) );
            }
            if ( output_height.index() != 0) {
                infer_.new_attr("output_height", std::get<1>(output_height) );
            }
            if ( mode.index() != 0) {
                infer_.new_attr("mode", std::get<1>(mode) );
            }

            infer_.new_input(batch_indices);
            infer_.new_input(rois);
            infer_.new_input(X);

            auto f = query_inference_function("RoiAlign");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_RoiAlign(X, rois, batch_indices, Y, mode, output_height, output_width, sampling_ratio, spatial_scale) != YNX_OK ) {
                yannx_panic("API: RoiAlign  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto spatial_scale = fetch_optional_float(stack);
            auto sampling_ratio = fetch_optional_int(stack);
            auto output_width = fetch_optional_int(stack);
            auto output_height = fetch_optional_int(stack);
            auto mode = fetch_optional_string(stack);

            auto batch_indices = fetch_tensor(stack);
            auto rois = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_RoiAlign(X, rois, batch_indices, Y, mode, output_height, output_width, sampling_ratio, spatial_scale) != YNX_OK ) {
                yannx_panic("API: RoiAlign  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RoiAlign)
    };


    struct NonMaxSuppression : NativeWord<TensorType> {
        tensor_t selected_indices;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto center_point_box = fetch_optional_int(stack);

            auto score_threshold = fetch_optional_tensor(stack);
            auto iou_threshold = fetch_optional_tensor(stack);
            auto max_output_boxes_per_class = fetch_optional_tensor(stack);
            auto scores = fetch_tensor(stack);
            auto boxes = fetch_tensor(stack);

            selected_indices = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( center_point_box.index() != 0) {
                infer_.new_attr("center_point_box", std::get<1>(center_point_box) );
            }

            infer_.new_input(score_threshold);
            infer_.new_input(iou_threshold);
            infer_.new_input(max_output_boxes_per_class);
            infer_.new_input(scores);
            infer_.new_input(boxes);

            auto f = query_inference_function("NonMaxSuppression");
            infer_.do_inference(f);
            infer_.check_output(0, selected_indices);

#endif

            if ( boxes->onnx_NonMaxSuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, selected_indices, center_point_box) != YNX_OK ) {
                yannx_panic("API: NonMaxSuppression  return error!");
            }

            put_tensor(stack, selected_indices);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto center_point_box = fetch_optional_int(stack);

            auto score_threshold = fetch_optional_tensor(stack);
            auto iou_threshold = fetch_optional_tensor(stack);
            auto max_output_boxes_per_class = fetch_optional_tensor(stack);
            auto scores = fetch_tensor(stack);
            auto boxes = fetch_tensor(stack);


            if ( boxes->onnx_NonMaxSuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, selected_indices, center_point_box) != YNX_OK ) {
                yannx_panic("API: NonMaxSuppression  return error!");
            }

            put_tensor(stack, selected_indices);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(NonMaxSuppression)
    };

}
namespace quantization {

    struct QuantizeLinear : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto y_zero_point = fetch_optional_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);

            y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(y_zero_point);
            infer_.new_input(y_scale);
            infer_.new_input(x);

            auto f = query_inference_function("QuantizeLinear");
            infer_.do_inference(f);
            infer_.check_output(0, y);

#endif

            if ( x->onnx_QuantizeLinear(x, y_scale, y_zero_point, y, axis) != YNX_OK ) {
                yannx_panic("API: QuantizeLinear  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto y_zero_point = fetch_optional_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_QuantizeLinear(x, y_scale, y_zero_point, y, axis) != YNX_OK ) {
                yannx_panic("API: QuantizeLinear  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(QuantizeLinear)
    };


    struct DynamicQuantizeLinear : NativeWord<TensorType> {
        tensor_t y;
        tensor_t y_scale;
        tensor_t y_zero_point;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto x = fetch_tensor(stack);

            y = TensorType::create_undefined_user_tensor();
            y_scale = TensorType::create_undefined_user_tensor();
            y_zero_point = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(3);

            infer_.new_input(x);

            auto f = query_inference_function("DynamicQuantizeLinear");
            infer_.do_inference(f);
            infer_.check_output(0, y);
            infer_.check_output(1, y_scale);
            infer_.check_output(2, y_zero_point);

#endif

            if ( x->onnx_DynamicQuantizeLinear(x, y, y_scale, y_zero_point) != YNX_OK ) {
                yannx_panic("API: DynamicQuantizeLinear  return error!");
            }

            put_tensor(stack, y);
            put_tensor(stack, y_scale);
            put_tensor(stack, y_zero_point);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto x = fetch_tensor(stack);


            if ( x->onnx_DynamicQuantizeLinear(x, y, y_scale, y_zero_point) != YNX_OK ) {
                yannx_panic("API: DynamicQuantizeLinear  return error!");
            }

            put_tensor(stack, y);
            put_tensor(stack, y_scale);
            put_tensor(stack, y_zero_point);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(DynamicQuantizeLinear)
    };


    struct DequantizeLinear : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto x_zero_point = fetch_optional_tensor(stack);
            auto x_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);

            y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(x_zero_point);
            infer_.new_input(x_scale);
            infer_.new_input(x);

            auto f = query_inference_function("DequantizeLinear");
            infer_.do_inference(f);
            infer_.check_output(0, y);

#endif

            if ( x->onnx_DequantizeLinear(x, x_scale, x_zero_point, y, axis) != YNX_OK ) {
                yannx_panic("API: DequantizeLinear  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto x_zero_point = fetch_optional_tensor(stack);
            auto x_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_DequantizeLinear(x, x_scale, x_zero_point, y, axis) != YNX_OK ) {
                yannx_panic("API: DequantizeLinear  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(DequantizeLinear)
    };

}
namespace reduction {

    struct ReduceProd : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axes.index() != 0) {
                infer_.new_attr("axes", std::get<1>(axes) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ReduceProd");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ReduceProd(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceProd  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceProd(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceProd  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceProd)
    };


    struct ReduceMin : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axes.index() != 0) {
                infer_.new_attr("axes", std::get<1>(axes) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ReduceMin");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ReduceMin(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMin  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceMin(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMin  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceMin)
    };


    struct ReduceSumSquare : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axes.index() != 0) {
                infer_.new_attr("axes", std::get<1>(axes) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ReduceSumSquare");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ReduceSumSquare(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceSumSquare  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceSumSquare(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceSumSquare  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceSumSquare)
    };


    struct ReduceSum : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto noop_with_empty_axes = fetch_optional_int(stack);
            auto keepdims = fetch_optional_int(stack);

            auto axes = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( noop_with_empty_axes.index() != 0) {
                infer_.new_attr("noop_with_empty_axes", std::get<1>(noop_with_empty_axes) );
            }
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }

            infer_.new_input(axes);
            infer_.new_input(data);

            auto f = query_inference_function("ReduceSum");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ReduceSum(data, axes, reduced, keepdims, noop_with_empty_axes) != YNX_OK ) {
                yannx_panic("API: ReduceSum  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto noop_with_empty_axes = fetch_optional_int(stack);
            auto keepdims = fetch_optional_int(stack);

            auto axes = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceSum(data, axes, reduced, keepdims, noop_with_empty_axes) != YNX_OK ) {
                yannx_panic("API: ReduceSum  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceSum)
    };


    struct ReduceLogSumExp : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axes.index() != 0) {
                infer_.new_attr("axes", std::get<1>(axes) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ReduceLogSumExp");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ReduceLogSumExp(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceLogSumExp  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceLogSumExp(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceLogSumExp  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceLogSumExp)
    };


    struct ReduceMax : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axes.index() != 0) {
                infer_.new_attr("axes", std::get<1>(axes) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ReduceMax");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ReduceMax(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMax  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceMax(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMax  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceMax)
    };


    struct ArgMax : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto select_last_index = fetch_optional_int(stack);
            auto keepdims = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( select_last_index.index() != 0) {
                infer_.new_attr("select_last_index", std::get<1>(select_last_index) );
            }
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ArgMax");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ArgMax(data, reduced, axis, keepdims, select_last_index) != YNX_OK ) {
                yannx_panic("API: ArgMax  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto select_last_index = fetch_optional_int(stack);
            auto keepdims = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ArgMax(data, reduced, axis, keepdims, select_last_index) != YNX_OK ) {
                yannx_panic("API: ArgMax  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ArgMax)
    };


    struct ArgMin : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto select_last_index = fetch_optional_int(stack);
            auto keepdims = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( select_last_index.index() != 0) {
                infer_.new_attr("select_last_index", std::get<1>(select_last_index) );
            }
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ArgMin");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ArgMin(data, reduced, axis, keepdims, select_last_index) != YNX_OK ) {
                yannx_panic("API: ArgMin  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto select_last_index = fetch_optional_int(stack);
            auto keepdims = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ArgMin(data, reduced, axis, keepdims, select_last_index) != YNX_OK ) {
                yannx_panic("API: ArgMin  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ArgMin)
    };


    struct ReduceLogSum : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axes.index() != 0) {
                infer_.new_attr("axes", std::get<1>(axes) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ReduceLogSum");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ReduceLogSum(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceLogSum  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceLogSum(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceLogSum  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceLogSum)
    };


    struct ReduceMean : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axes.index() != 0) {
                infer_.new_attr("axes", std::get<1>(axes) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ReduceMean");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ReduceMean(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMean  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceMean(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMean  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceMean)
    };


    struct ReduceL2 : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axes.index() != 0) {
                infer_.new_attr("axes", std::get<1>(axes) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ReduceL2");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ReduceL2(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceL2  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceL2(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceL2  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceL2)
    };


    struct ReduceL1 : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);

            reduced = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axes.index() != 0) {
                infer_.new_attr("axes", std::get<1>(axes) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("ReduceL1");
            infer_.do_inference(f);
            infer_.check_output(0, reduced);

#endif

            if ( data->onnx_ReduceL1(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceL1  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto keepdims = fetch_optional_int(stack);
            auto axes = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceL1(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceL1  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceL1)
    };

}
namespace rnn {

    struct LSTM : NativeWord<TensorType> {
        std::variant<void *, tensor_t> Y;
        std::variant<void *, tensor_t> Y_h;
        std::variant<void *, tensor_t> Y_c;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto layout = fetch_optional_int(stack);
            auto input_forget = fetch_optional_int(stack);
            auto hidden_size = fetch_optional_int(stack);
            auto direction = fetch_optional_string(stack);
            auto clip = fetch_optional_float(stack);
            auto activations = fetch_optional_strings(stack);
            auto activation_beta = fetch_optional_floats(stack);
            auto activation_alpha = fetch_optional_floats(stack);

            auto P = fetch_optional_tensor(stack);
            auto initial_c = fetch_optional_tensor(stack);
            auto initial_h = fetch_optional_tensor(stack);
            auto sequence_lens = fetch_optional_tensor(stack);
            auto B = fetch_optional_tensor(stack);
            auto R = fetch_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);



#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(3);
            if ( layout.index() != 0) {
                infer_.new_attr("layout", std::get<1>(layout) );
            }
            if ( input_forget.index() != 0) {
                infer_.new_attr("input_forget", std::get<1>(input_forget) );
            }
            if ( hidden_size.index() != 0) {
                infer_.new_attr("hidden_size", std::get<1>(hidden_size) );
            }
            if ( direction.index() != 0) {
                infer_.new_attr("direction", std::get<1>(direction) );
            }
            if ( clip.index() != 0) {
                infer_.new_attr("clip", std::get<1>(clip) );
            }
            if ( activations.index() != 0) {
                infer_.new_attr("activations", std::get<1>(activations) );
            }
            if ( activation_beta.index() != 0) {
                infer_.new_attr("activation_beta", std::get<1>(activation_beta) );
            }
            if ( activation_alpha.index() != 0) {
                infer_.new_attr("activation_alpha", std::get<1>(activation_alpha) );
            }

            infer_.new_input(P);
            infer_.new_input(initial_c);
            infer_.new_input(initial_h);
            infer_.new_input(sequence_lens);
            infer_.new_input(B);
            infer_.new_input(R);
            infer_.new_input(W);
            infer_.new_input(X);

            auto f = query_inference_function("LSTM");
            infer_.do_inference(f);
            infer_.check_output(0, Y);
            infer_.check_output(1, Y_h);
            infer_.check_output(2, Y_c);

#endif

            if ( X->onnx_LSTM(X, W, R, B, sequence_lens, initial_h, initial_c, P, Y, Y_h, Y_c, activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget, layout) != YNX_OK ) {
                yannx_panic("API: LSTM  return error!");
            }

            put_optional_tensor(stack, Y);
            put_optional_tensor(stack, Y_h);
            put_optional_tensor(stack, Y_c);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto layout = fetch_optional_int(stack);
            auto input_forget = fetch_optional_int(stack);
            auto hidden_size = fetch_optional_int(stack);
            auto direction = fetch_optional_string(stack);
            auto clip = fetch_optional_float(stack);
            auto activations = fetch_optional_strings(stack);
            auto activation_beta = fetch_optional_floats(stack);
            auto activation_alpha = fetch_optional_floats(stack);

            auto P = fetch_optional_tensor(stack);
            auto initial_c = fetch_optional_tensor(stack);
            auto initial_h = fetch_optional_tensor(stack);
            auto sequence_lens = fetch_optional_tensor(stack);
            auto B = fetch_optional_tensor(stack);
            auto R = fetch_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_LSTM(X, W, R, B, sequence_lens, initial_h, initial_c, P, Y, Y_h, Y_c, activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget, layout) != YNX_OK ) {
                yannx_panic("API: LSTM  return error!");
            }

            put_optional_tensor(stack, Y);
            put_optional_tensor(stack, Y_h);
            put_optional_tensor(stack, Y_c);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LSTM)
    };


    struct GRU : NativeWord<TensorType> {
        std::variant<void *, tensor_t> Y;
        std::variant<void *, tensor_t> Y_h;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto linear_before_reset = fetch_optional_int(stack);
            auto layout = fetch_optional_int(stack);
            auto hidden_size = fetch_optional_int(stack);
            auto direction = fetch_optional_string(stack);
            auto clip = fetch_optional_float(stack);
            auto activations = fetch_optional_strings(stack);
            auto activation_beta = fetch_optional_floats(stack);
            auto activation_alpha = fetch_optional_floats(stack);

            auto initial_h = fetch_optional_tensor(stack);
            auto sequence_lens = fetch_optional_tensor(stack);
            auto B = fetch_optional_tensor(stack);
            auto R = fetch_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);



#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(2);
            if ( linear_before_reset.index() != 0) {
                infer_.new_attr("linear_before_reset", std::get<1>(linear_before_reset) );
            }
            if ( layout.index() != 0) {
                infer_.new_attr("layout", std::get<1>(layout) );
            }
            if ( hidden_size.index() != 0) {
                infer_.new_attr("hidden_size", std::get<1>(hidden_size) );
            }
            if ( direction.index() != 0) {
                infer_.new_attr("direction", std::get<1>(direction) );
            }
            if ( clip.index() != 0) {
                infer_.new_attr("clip", std::get<1>(clip) );
            }
            if ( activations.index() != 0) {
                infer_.new_attr("activations", std::get<1>(activations) );
            }
            if ( activation_beta.index() != 0) {
                infer_.new_attr("activation_beta", std::get<1>(activation_beta) );
            }
            if ( activation_alpha.index() != 0) {
                infer_.new_attr("activation_alpha", std::get<1>(activation_alpha) );
            }

            infer_.new_input(initial_h);
            infer_.new_input(sequence_lens);
            infer_.new_input(B);
            infer_.new_input(R);
            infer_.new_input(W);
            infer_.new_input(X);

            auto f = query_inference_function("GRU");
            infer_.do_inference(f);
            infer_.check_output(0, Y);
            infer_.check_output(1, Y_h);

#endif

            if ( X->onnx_GRU(X, W, R, B, sequence_lens, initial_h, Y, Y_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout, linear_before_reset) != YNX_OK ) {
                yannx_panic("API: GRU  return error!");
            }

            put_optional_tensor(stack, Y);
            put_optional_tensor(stack, Y_h);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto linear_before_reset = fetch_optional_int(stack);
            auto layout = fetch_optional_int(stack);
            auto hidden_size = fetch_optional_int(stack);
            auto direction = fetch_optional_string(stack);
            auto clip = fetch_optional_float(stack);
            auto activations = fetch_optional_strings(stack);
            auto activation_beta = fetch_optional_floats(stack);
            auto activation_alpha = fetch_optional_floats(stack);

            auto initial_h = fetch_optional_tensor(stack);
            auto sequence_lens = fetch_optional_tensor(stack);
            auto B = fetch_optional_tensor(stack);
            auto R = fetch_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_GRU(X, W, R, B, sequence_lens, initial_h, Y, Y_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout, linear_before_reset) != YNX_OK ) {
                yannx_panic("API: GRU  return error!");
            }

            put_optional_tensor(stack, Y);
            put_optional_tensor(stack, Y_h);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GRU)
    };


    struct RNN : NativeWord<TensorType> {
        std::variant<void *, tensor_t> Y;
        std::variant<void *, tensor_t> Y_h;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto layout = fetch_optional_int(stack);
            auto hidden_size = fetch_optional_int(stack);
            auto direction = fetch_optional_string(stack);
            auto clip = fetch_optional_float(stack);
            auto activations = fetch_optional_strings(stack);
            auto activation_beta = fetch_optional_floats(stack);
            auto activation_alpha = fetch_optional_floats(stack);

            auto initial_h = fetch_optional_tensor(stack);
            auto sequence_lens = fetch_optional_tensor(stack);
            auto B = fetch_optional_tensor(stack);
            auto R = fetch_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);



#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(2);
            if ( layout.index() != 0) {
                infer_.new_attr("layout", std::get<1>(layout) );
            }
            if ( hidden_size.index() != 0) {
                infer_.new_attr("hidden_size", std::get<1>(hidden_size) );
            }
            if ( direction.index() != 0) {
                infer_.new_attr("direction", std::get<1>(direction) );
            }
            if ( clip.index() != 0) {
                infer_.new_attr("clip", std::get<1>(clip) );
            }
            if ( activations.index() != 0) {
                infer_.new_attr("activations", std::get<1>(activations) );
            }
            if ( activation_beta.index() != 0) {
                infer_.new_attr("activation_beta", std::get<1>(activation_beta) );
            }
            if ( activation_alpha.index() != 0) {
                infer_.new_attr("activation_alpha", std::get<1>(activation_alpha) );
            }

            infer_.new_input(initial_h);
            infer_.new_input(sequence_lens);
            infer_.new_input(B);
            infer_.new_input(R);
            infer_.new_input(W);
            infer_.new_input(X);

            auto f = query_inference_function("RNN");
            infer_.do_inference(f);
            infer_.check_output(0, Y);
            infer_.check_output(1, Y_h);

#endif

            if ( X->onnx_RNN(X, W, R, B, sequence_lens, initial_h, Y, Y_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout) != YNX_OK ) {
                yannx_panic("API: RNN  return error!");
            }

            put_optional_tensor(stack, Y);
            put_optional_tensor(stack, Y_h);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto layout = fetch_optional_int(stack);
            auto hidden_size = fetch_optional_int(stack);
            auto direction = fetch_optional_string(stack);
            auto clip = fetch_optional_float(stack);
            auto activations = fetch_optional_strings(stack);
            auto activation_beta = fetch_optional_floats(stack);
            auto activation_alpha = fetch_optional_floats(stack);

            auto initial_h = fetch_optional_tensor(stack);
            auto sequence_lens = fetch_optional_tensor(stack);
            auto B = fetch_optional_tensor(stack);
            auto R = fetch_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_RNN(X, W, R, B, sequence_lens, initial_h, Y, Y_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout) != YNX_OK ) {
                yannx_panic("API: RNN  return error!");
            }

            put_optional_tensor(stack, Y);
            put_optional_tensor(stack, Y_h);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RNN)
    };

}
namespace sequence {

    struct SequenceEmpty : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto dtype = fetch_optional_int(stack);


            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( dtype.index() != 0) {
                infer_.new_attr("dtype", std::get<1>(dtype) );
            }


            auto f = query_inference_function("SequenceEmpty");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( output->onnx_SequenceEmpty(output, dtype) != YNX_OK ) {
                yannx_panic("API: SequenceEmpty  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto dtype = fetch_optional_int(stack);



            if ( output->onnx_SequenceEmpty(output, dtype) != YNX_OK ) {
                yannx_panic("API: SequenceEmpty  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SequenceEmpty)
    };


    struct SplitToSequence : NativeWord<TensorType> {
        tensor_t output_sequence;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto keepdims = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto split = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);

            output_sequence = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( keepdims.index() != 0) {
                infer_.new_attr("keepdims", std::get<1>(keepdims) );
            }
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(split);
            infer_.new_input(input);

            auto f = query_inference_function("SplitToSequence");
            infer_.do_inference(f);
            infer_.check_output(0, output_sequence);

#endif

            if ( input->onnx_SplitToSequence(input, split, output_sequence, axis, keepdims) != YNX_OK ) {
                yannx_panic("API: SplitToSequence  return error!");
            }

            put_tensor(stack, output_sequence);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto keepdims = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto split = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_SplitToSequence(input, split, output_sequence, axis, keepdims) != YNX_OK ) {
                yannx_panic("API: SplitToSequence  return error!");
            }

            put_tensor(stack, output_sequence);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SplitToSequence)
    };


    struct SequenceAt : NativeWord<TensorType> {
        tensor_t tensor;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto position = fetch_tensor(stack);
            auto input_sequence = fetch_tensor(stack);

            tensor = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(position);
            infer_.new_input(input_sequence);

            auto f = query_inference_function("SequenceAt");
            infer_.do_inference(f);
            infer_.check_output(0, tensor);

#endif

            if ( input_sequence->onnx_SequenceAt(input_sequence, position, tensor) != YNX_OK ) {
                yannx_panic("API: SequenceAt  return error!");
            }

            put_tensor(stack, tensor);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto position = fetch_tensor(stack);
            auto input_sequence = fetch_tensor(stack);


            if ( input_sequence->onnx_SequenceAt(input_sequence, position, tensor) != YNX_OK ) {
                yannx_panic("API: SequenceAt  return error!");
            }

            put_tensor(stack, tensor);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SequenceAt)
    };


    struct SequenceLength : NativeWord<TensorType> {
        tensor_t length;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input_sequence = fetch_tensor(stack);

            length = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input_sequence);

            auto f = query_inference_function("SequenceLength");
            infer_.do_inference(f);
            infer_.check_output(0, length);

#endif

            if ( input_sequence->onnx_SequenceLength(input_sequence, length) != YNX_OK ) {
                yannx_panic("API: SequenceLength  return error!");
            }

            put_tensor(stack, length);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input_sequence = fetch_tensor(stack);


            if ( input_sequence->onnx_SequenceLength(input_sequence, length) != YNX_OK ) {
                yannx_panic("API: SequenceLength  return error!");
            }

            put_tensor(stack, length);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SequenceLength)
    };


    struct SequenceConstruct : NativeWord<TensorType> {
        tensor_t output_sequence;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto inputs = fetch_tensors(stack);

            output_sequence = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(inputs);

            auto f = query_inference_function("SequenceConstruct");
            infer_.do_inference(f);
            infer_.check_output(0, output_sequence);

#endif

            if ( output_sequence->onnx_SequenceConstruct(inputs, output_sequence) != YNX_OK ) {
                yannx_panic("API: SequenceConstruct  return error!");
            }

            put_tensor(stack, output_sequence);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto inputs = fetch_tensors(stack);


            if ( output_sequence->onnx_SequenceConstruct(inputs, output_sequence) != YNX_OK ) {
                yannx_panic("API: SequenceConstruct  return error!");
            }

            put_tensor(stack, output_sequence);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SequenceConstruct)
    };


    struct SequenceInsert : NativeWord<TensorType> {
        tensor_t output_sequence;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto position = fetch_optional_tensor(stack);
            auto tensor = fetch_tensor(stack);
            auto input_sequence = fetch_tensor(stack);

            output_sequence = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(position);
            infer_.new_input(tensor);
            infer_.new_input(input_sequence);

            auto f = query_inference_function("SequenceInsert");
            infer_.do_inference(f);
            infer_.check_output(0, output_sequence);

#endif

            if ( input_sequence->onnx_SequenceInsert(input_sequence, tensor, position, output_sequence) != YNX_OK ) {
                yannx_panic("API: SequenceInsert  return error!");
            }

            put_tensor(stack, output_sequence);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto position = fetch_optional_tensor(stack);
            auto tensor = fetch_tensor(stack);
            auto input_sequence = fetch_tensor(stack);


            if ( input_sequence->onnx_SequenceInsert(input_sequence, tensor, position, output_sequence) != YNX_OK ) {
                yannx_panic("API: SequenceInsert  return error!");
            }

            put_tensor(stack, output_sequence);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SequenceInsert)
    };


    struct SequenceErase : NativeWord<TensorType> {
        tensor_t output_sequence;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto position = fetch_optional_tensor(stack);
            auto input_sequence = fetch_tensor(stack);

            output_sequence = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(position);
            infer_.new_input(input_sequence);

            auto f = query_inference_function("SequenceErase");
            infer_.do_inference(f);
            infer_.check_output(0, output_sequence);

#endif

            if ( input_sequence->onnx_SequenceErase(input_sequence, position, output_sequence) != YNX_OK ) {
                yannx_panic("API: SequenceErase  return error!");
            }

            put_tensor(stack, output_sequence);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto position = fetch_optional_tensor(stack);
            auto input_sequence = fetch_tensor(stack);


            if ( input_sequence->onnx_SequenceErase(input_sequence, position, output_sequence) != YNX_OK ) {
                yannx_panic("API: SequenceErase  return error!");
            }

            put_tensor(stack, output_sequence);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SequenceErase)
    };


    struct ConcatFromSequence : NativeWord<TensorType> {
        tensor_t concat_result;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto new_axis = fetch_optional_int(stack);
            auto axis = fetch_int(stack);

            auto input_sequence = fetch_tensor(stack);

            concat_result = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( new_axis.index() != 0) {
                infer_.new_attr("new_axis", std::get<1>(new_axis) );
            }
            infer_.new_attr("axis", axis);

            infer_.new_input(input_sequence);

            auto f = query_inference_function("ConcatFromSequence");
            infer_.do_inference(f);
            infer_.check_output(0, concat_result);

#endif

            if ( input_sequence->onnx_ConcatFromSequence(input_sequence, concat_result, axis, new_axis) != YNX_OK ) {
                yannx_panic("API: ConcatFromSequence  return error!");
            }

            put_tensor(stack, concat_result);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto new_axis = fetch_optional_int(stack);
            auto axis = fetch_int(stack);

            auto input_sequence = fetch_tensor(stack);


            if ( input_sequence->onnx_ConcatFromSequence(input_sequence, concat_result, axis, new_axis) != YNX_OK ) {
                yannx_panic("API: ConcatFromSequence  return error!");
            }

            put_tensor(stack, concat_result);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ConcatFromSequence)
    };

}
namespace tensor {

    struct CastLike : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto target_type = fetch_tensor(stack);
            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(target_type);
            infer_.new_input(input);

            auto f = query_inference_function("CastLike");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_CastLike(input, target_type, output) != YNX_OK ) {
                yannx_panic("API: CastLike  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto target_type = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_CastLike(input, target_type, output) != YNX_OK ) {
                yannx_panic("API: CastLike  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(CastLike)
    };


    struct Shape : NativeWord<TensorType> {
        tensor_t shape;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto start = fetch_optional_int(stack);
            auto end = fetch_optional_int(stack);

            auto data = fetch_tensor(stack);

            shape = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( start.index() != 0) {
                infer_.new_attr("start", std::get<1>(start) );
            }
            if ( end.index() != 0) {
                infer_.new_attr("end", std::get<1>(end) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("Shape");
            infer_.do_inference(f);
            infer_.check_output(0, shape);

#endif

            if ( data->onnx_Shape(data, shape, end, start) != YNX_OK ) {
                yannx_panic("API: Shape  return error!");
            }

            put_tensor(stack, shape);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto start = fetch_optional_int(stack);
            auto end = fetch_optional_int(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_Shape(data, shape, end, start) != YNX_OK ) {
                yannx_panic("API: Shape  return error!");
            }

            put_tensor(stack, shape);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Shape)
    };


    struct Reshape : NativeWord<TensorType> {
        tensor_t reshaped;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto allowzero = fetch_optional_int(stack);

            auto shape = fetch_tensor(stack);
            auto data = fetch_tensor(stack);

            reshaped = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( allowzero.index() != 0) {
                infer_.new_attr("allowzero", std::get<1>(allowzero) );
            }

            infer_.new_input(shape);
            infer_.new_input(data);

            auto f = query_inference_function("Reshape");
            infer_.do_inference(f);
            infer_.check_output(0, reshaped);

#endif

            if ( data->onnx_Reshape(data, shape, reshaped, allowzero) != YNX_OK ) {
                yannx_panic("API: Reshape  return error!");
            }

            put_tensor(stack, reshaped);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto allowzero = fetch_optional_int(stack);

            auto shape = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Reshape(data, shape, reshaped, allowzero) != YNX_OK ) {
                yannx_panic("API: Reshape  return error!");
            }

            put_tensor(stack, reshaped);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Reshape)
    };


    struct DepthToSpace : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto mode = fetch_optional_string(stack);
            auto blocksize = fetch_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( mode.index() != 0) {
                infer_.new_attr("mode", std::get<1>(mode) );
            }
            infer_.new_attr("blocksize", blocksize);

            infer_.new_input(input);

            auto f = query_inference_function("DepthToSpace");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_DepthToSpace(input, output, blocksize, mode) != YNX_OK ) {
                yannx_panic("API: DepthToSpace  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto mode = fetch_optional_string(stack);
            auto blocksize = fetch_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_DepthToSpace(input, output, blocksize, mode) != YNX_OK ) {
                yannx_panic("API: DepthToSpace  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(DepthToSpace)
    };


    struct Concat : NativeWord<TensorType> {
        tensor_t concat_result;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_int(stack);

            auto inputs = fetch_tensors(stack);

            concat_result = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            infer_.new_attr("axis", axis);

            infer_.new_input(inputs);

            auto f = query_inference_function("Concat");
            infer_.do_inference(f);
            infer_.check_output(0, concat_result);

#endif

            if ( concat_result->onnx_Concat(inputs, concat_result, axis) != YNX_OK ) {
                yannx_panic("API: Concat  return error!");
            }

            put_tensor(stack, concat_result);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_int(stack);

            auto inputs = fetch_tensors(stack);


            if ( concat_result->onnx_Concat(inputs, concat_result, axis) != YNX_OK ) {
                yannx_panic("API: Concat  return error!");
            }

            put_tensor(stack, concat_result);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Concat)
    };


    struct Gather : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(indices);
            infer_.new_input(data);

            auto f = query_inference_function("Gather");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( data->onnx_Gather(data, indices, output, axis) != YNX_OK ) {
                yannx_panic("API: Gather  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Gather(data, indices, output, axis) != YNX_OK ) {
                yannx_panic("API: Gather  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Gather)
    };


    struct Size : NativeWord<TensorType> {
        tensor_t size;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto data = fetch_tensor(stack);

            size = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(data);

            auto f = query_inference_function("Size");
            infer_.do_inference(f);
            infer_.check_output(0, size);

#endif

            if ( data->onnx_Size(data, size) != YNX_OK ) {
                yannx_panic("API: Size  return error!");
            }

            put_tensor(stack, size);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto data = fetch_tensor(stack);


            if ( data->onnx_Size(data, size) != YNX_OK ) {
                yannx_panic("API: Size  return error!");
            }

            put_tensor(stack, size);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Size)
    };


    struct Cast : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto to = fetch_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            infer_.new_attr("to", to);

            infer_.new_input(input);

            auto f = query_inference_function("Cast");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Cast(input, output, to) != YNX_OK ) {
                yannx_panic("API: Cast  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto to = fetch_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Cast(input, output, to) != YNX_OK ) {
                yannx_panic("API: Cast  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Cast)
    };


    struct Split : NativeWord<TensorType> {
        std::vector<tensor_t>outputs;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto split = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);



#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(split);
            infer_.new_input(input);

            auto f = query_inference_function("Split");
            infer_.do_inference(f);
            infer_.check_output(0, outputs);

#endif

            if ( input->onnx_Split(input, split, outputs, axis) != YNX_OK ) {
                yannx_panic("API: Split  return error!");
            }

            put_tensors(stack, outputs);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto split = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Split(input, split, outputs, axis) != YNX_OK ) {
                yannx_panic("API: Split  return error!");
            }

            put_tensors(stack, outputs);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Split)
    };


    struct Identity : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(input);

            auto f = query_inference_function("Identity");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Identity(input, output) != YNX_OK ) {
                yannx_panic("API: Identity  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto input = fetch_tensor(stack);


            if ( input->onnx_Identity(input, output) != YNX_OK ) {
                yannx_panic("API: Identity  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Identity)
    };


    struct Slice : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto steps = fetch_optional_tensor(stack);
            auto axes = fetch_optional_tensor(stack);
            auto ends = fetch_tensor(stack);
            auto starts = fetch_tensor(stack);
            auto data = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(steps);
            infer_.new_input(axes);
            infer_.new_input(ends);
            infer_.new_input(starts);
            infer_.new_input(data);

            auto f = query_inference_function("Slice");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( data->onnx_Slice(data, starts, ends, axes, steps, output) != YNX_OK ) {
                yannx_panic("API: Slice  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto steps = fetch_optional_tensor(stack);
            auto axes = fetch_optional_tensor(stack);
            auto ends = fetch_tensor(stack);
            auto starts = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Slice(data, starts, ends, axes, steps, output) != YNX_OK ) {
                yannx_panic("API: Slice  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Slice)
    };


    struct GatherND : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto batch_dims = fetch_optional_int(stack);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( batch_dims.index() != 0) {
                infer_.new_attr("batch_dims", std::get<1>(batch_dims) );
            }

            infer_.new_input(indices);
            infer_.new_input(data);

            auto f = query_inference_function("GatherND");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( data->onnx_GatherND(data, indices, output, batch_dims) != YNX_OK ) {
                yannx_panic("API: GatherND  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto batch_dims = fetch_optional_int(stack);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_GatherND(data, indices, output, batch_dims) != YNX_OK ) {
                yannx_panic("API: GatherND  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GatherND)
    };


    struct SpaceToDepth : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto blocksize = fetch_int(stack);

            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            infer_.new_attr("blocksize", blocksize);

            infer_.new_input(input);

            auto f = query_inference_function("SpaceToDepth");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_SpaceToDepth(input, output, blocksize) != YNX_OK ) {
                yannx_panic("API: SpaceToDepth  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto blocksize = fetch_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_SpaceToDepth(input, output, blocksize) != YNX_OK ) {
                yannx_panic("API: SpaceToDepth  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SpaceToDepth)
    };


    struct Squeeze : NativeWord<TensorType> {
        tensor_t squeezed;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto axes = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);

            squeezed = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(axes);
            infer_.new_input(data);

            auto f = query_inference_function("Squeeze");
            infer_.do_inference(f);
            infer_.check_output(0, squeezed);

#endif

            if ( data->onnx_Squeeze(data, axes, squeezed) != YNX_OK ) {
                yannx_panic("API: Squeeze  return error!");
            }

            put_tensor(stack, squeezed);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto axes = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Squeeze(data, axes, squeezed) != YNX_OK ) {
                yannx_panic("API: Squeeze  return error!");
            }

            put_tensor(stack, squeezed);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Squeeze)
    };


    struct Unique : NativeWord<TensorType> {
        tensor_t Y;
        std::variant<void *, tensor_t> indices;
        std::variant<void *, tensor_t> inverse_indices;
        std::variant<void *, tensor_t> counts;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto sorted = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(4);
            if ( sorted.index() != 0) {
                infer_.new_attr("sorted", std::get<1>(sorted) );
            }
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("Unique");
            infer_.do_inference(f);
            infer_.check_output(0, Y);
            infer_.check_output(1, indices);
            infer_.check_output(2, inverse_indices);
            infer_.check_output(3, counts);

#endif

            if ( X->onnx_Unique(X, Y, indices, inverse_indices, counts, axis, sorted) != YNX_OK ) {
                yannx_panic("API: Unique  return error!");
            }

            put_tensor(stack, Y);
            put_optional_tensor(stack, indices);
            put_optional_tensor(stack, inverse_indices);
            put_optional_tensor(stack, counts);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto sorted = fetch_optional_int(stack);
            auto axis = fetch_optional_int(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Unique(X, Y, indices, inverse_indices, counts, axis, sorted) != YNX_OK ) {
                yannx_panic("API: Unique  return error!");
            }

            put_tensor(stack, Y);
            put_optional_tensor(stack, indices);
            put_optional_tensor(stack, inverse_indices);
            put_optional_tensor(stack, counts);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Unique)
    };


    struct IsNaN : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("IsNaN");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_IsNaN(X, Y) != YNX_OK ) {
                yannx_panic("API: IsNaN  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_IsNaN(X, Y) != YNX_OK ) {
                yannx_panic("API: IsNaN  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(IsNaN)
    };


    struct Tile : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto repeats = fetch_tensor(stack);
            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(repeats);
            infer_.new_input(input);

            auto f = query_inference_function("Tile");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Tile(input, repeats, output) != YNX_OK ) {
                yannx_panic("API: Tile  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto repeats = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Tile(input, repeats, output) != YNX_OK ) {
                yannx_panic("API: Tile  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Tile)
    };


    struct ReverseSequence : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto time_axis = fetch_optional_int(stack);
            auto batch_axis = fetch_optional_int(stack);

            auto sequence_lens = fetch_tensor(stack);
            auto input = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( time_axis.index() != 0) {
                infer_.new_attr("time_axis", std::get<1>(time_axis) );
            }
            if ( batch_axis.index() != 0) {
                infer_.new_attr("batch_axis", std::get<1>(batch_axis) );
            }

            infer_.new_input(sequence_lens);
            infer_.new_input(input);

            auto f = query_inference_function("ReverseSequence");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( input->onnx_ReverseSequence(input, sequence_lens, Y, batch_axis, time_axis) != YNX_OK ) {
                yannx_panic("API: ReverseSequence  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto time_axis = fetch_optional_int(stack);
            auto batch_axis = fetch_optional_int(stack);

            auto sequence_lens = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_ReverseSequence(input, sequence_lens, Y, batch_axis, time_axis) != YNX_OK ) {
                yannx_panic("API: ReverseSequence  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReverseSequence)
    };


    struct Transpose : NativeWord<TensorType> {
        tensor_t transposed;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto perm = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);

            transposed = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( perm.index() != 0) {
                infer_.new_attr("perm", std::get<1>(perm) );
            }

            infer_.new_input(data);

            auto f = query_inference_function("Transpose");
            infer_.do_inference(f);
            infer_.check_output(0, transposed);

#endif

            if ( data->onnx_Transpose(data, transposed, perm) != YNX_OK ) {
                yannx_panic("API: Transpose  return error!");
            }

            put_tensor(stack, transposed);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto perm = fetch_optional_ints(stack);

            auto data = fetch_tensor(stack);


            if ( data->onnx_Transpose(data, transposed, perm) != YNX_OK ) {
                yannx_panic("API: Transpose  return error!");
            }

            put_tensor(stack, transposed);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Transpose)
    };


    struct Trilu : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto upper = fetch_optional_int(stack);

            auto k = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( upper.index() != 0) {
                infer_.new_attr("upper", std::get<1>(upper) );
            }

            infer_.new_input(k);
            infer_.new_input(input);

            auto f = query_inference_function("Trilu");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Trilu(input, k, output, upper) != YNX_OK ) {
                yannx_panic("API: Trilu  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto upper = fetch_optional_int(stack);

            auto k = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Trilu(input, k, output, upper) != YNX_OK ) {
                yannx_panic("API: Trilu  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Trilu)
    };


    struct Where : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);
            auto condition = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(Y);
            infer_.new_input(X);
            infer_.new_input(condition);

            auto f = query_inference_function("Where");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( condition->onnx_Where(condition, X, Y, output) != YNX_OK ) {
                yannx_panic("API: Where  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);
            auto condition = fetch_tensor(stack);


            if ( condition->onnx_Where(condition, X, Y, output) != YNX_OK ) {
                yannx_panic("API: Where  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Where)
    };


    struct Compress : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto condition = fetch_tensor(stack);
            auto input = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(condition);
            infer_.new_input(input);

            auto f = query_inference_function("Compress");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( input->onnx_Compress(input, condition, output, axis) != YNX_OK ) {
                yannx_panic("API: Compress  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto condition = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Compress(input, condition, output, axis) != YNX_OK ) {
                yannx_panic("API: Compress  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Compress)
    };


    struct Unsqueeze : NativeWord<TensorType> {
        tensor_t expanded;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto axes = fetch_tensor(stack);
            auto data = fetch_tensor(stack);

            expanded = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(axes);
            infer_.new_input(data);

            auto f = query_inference_function("Unsqueeze");
            infer_.do_inference(f);
            infer_.check_output(0, expanded);

#endif

            if ( data->onnx_Unsqueeze(data, axes, expanded) != YNX_OK ) {
                yannx_panic("API: Unsqueeze  return error!");
            }

            put_tensor(stack, expanded);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto axes = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Unsqueeze(data, axes, expanded) != YNX_OK ) {
                yannx_panic("API: Unsqueeze  return error!");
            }

            put_tensor(stack, expanded);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Unsqueeze)
    };


    struct OneHot : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto values = fetch_tensor(stack);
            auto depth = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(values);
            infer_.new_input(depth);
            infer_.new_input(indices);

            auto f = query_inference_function("OneHot");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( indices->onnx_OneHot(indices, depth, values, output, axis) != YNX_OK ) {
                yannx_panic("API: OneHot  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto values = fetch_tensor(stack);
            auto depth = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);


            if ( indices->onnx_OneHot(indices, depth, values, output, axis) != YNX_OK ) {
                yannx_panic("API: OneHot  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(OneHot)
    };


    struct NonZero : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(X);

            auto f = query_inference_function("NonZero");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_NonZero(X, Y) != YNX_OK ) {
                yannx_panic("API: NonZero  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto X = fetch_tensor(stack);


            if ( X->onnx_NonZero(X, Y) != YNX_OK ) {
                yannx_panic("API: NonZero  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(NonZero)
    };


    struct ScatterND : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;


            auto updates = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);

            infer_.new_input(updates);
            infer_.new_input(indices);
            infer_.new_input(data);

            auto f = query_inference_function("ScatterND");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( data->onnx_ScatterND(data, indices, updates, output) != YNX_OK ) {
                yannx_panic("API: ScatterND  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {

            auto updates = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_ScatterND(data, indices, updates, output) != YNX_OK ) {
                yannx_panic("API: ScatterND  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ScatterND)
    };


    struct Resize : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto nearest_mode = fetch_optional_string(stack);
            auto mode = fetch_optional_string(stack);
            auto extrapolation_value = fetch_optional_float(stack);
            auto exclude_outside = fetch_optional_int(stack);
            auto cubic_coeff_a = fetch_optional_float(stack);
            auto coordinate_transformation_mode = fetch_optional_string(stack);

            auto sizes = fetch_optional_tensor(stack);
            auto scales = fetch_optional_tensor(stack);
            auto roi = fetch_optional_tensor(stack);
            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( nearest_mode.index() != 0) {
                infer_.new_attr("nearest_mode", std::get<1>(nearest_mode) );
            }
            if ( mode.index() != 0) {
                infer_.new_attr("mode", std::get<1>(mode) );
            }
            if ( extrapolation_value.index() != 0) {
                infer_.new_attr("extrapolation_value", std::get<1>(extrapolation_value) );
            }
            if ( exclude_outside.index() != 0) {
                infer_.new_attr("exclude_outside", std::get<1>(exclude_outside) );
            }
            if ( cubic_coeff_a.index() != 0) {
                infer_.new_attr("cubic_coeff_a", std::get<1>(cubic_coeff_a) );
            }
            if ( coordinate_transformation_mode.index() != 0) {
                infer_.new_attr("coordinate_transformation_mode", std::get<1>(coordinate_transformation_mode) );
            }

            infer_.new_input(sizes);
            infer_.new_input(scales);
            infer_.new_input(roi);
            infer_.new_input(X);

            auto f = query_inference_function("Resize");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_Resize(X, roi, scales, sizes, Y, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode) != YNX_OK ) {
                yannx_panic("API: Resize  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto nearest_mode = fetch_optional_string(stack);
            auto mode = fetch_optional_string(stack);
            auto extrapolation_value = fetch_optional_float(stack);
            auto exclude_outside = fetch_optional_int(stack);
            auto cubic_coeff_a = fetch_optional_float(stack);
            auto coordinate_transformation_mode = fetch_optional_string(stack);

            auto sizes = fetch_optional_tensor(stack);
            auto scales = fetch_optional_tensor(stack);
            auto roi = fetch_optional_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_Resize(X, roi, scales, sizes, Y, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode) != YNX_OK ) {
                yannx_panic("API: Resize  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Resize)
    };


    struct Pad : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto mode = fetch_optional_string(stack);

            auto constant_value = fetch_optional_tensor(stack);
            auto pads = fetch_tensor(stack);
            auto data = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( mode.index() != 0) {
                infer_.new_attr("mode", std::get<1>(mode) );
            }

            infer_.new_input(constant_value);
            infer_.new_input(pads);
            infer_.new_input(data);

            auto f = query_inference_function("Pad");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( data->onnx_Pad(data, pads, constant_value, output, mode) != YNX_OK ) {
                yannx_panic("API: Pad  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto mode = fetch_optional_string(stack);

            auto constant_value = fetch_optional_tensor(stack);
            auto pads = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Pad(data, pads, constant_value, output, mode) != YNX_OK ) {
                yannx_panic("API: Pad  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Pad)
    };


    struct IsInf : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto detect_positive = fetch_optional_int(stack);
            auto detect_negative = fetch_optional_int(stack);

            auto X = fetch_tensor(stack);

            Y = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( detect_positive.index() != 0) {
                infer_.new_attr("detect_positive", std::get<1>(detect_positive) );
            }
            if ( detect_negative.index() != 0) {
                infer_.new_attr("detect_negative", std::get<1>(detect_negative) );
            }

            infer_.new_input(X);

            auto f = query_inference_function("IsInf");
            infer_.do_inference(f);
            infer_.check_output(0, Y);

#endif

            if ( X->onnx_IsInf(X, Y, detect_negative, detect_positive) != YNX_OK ) {
                yannx_panic("API: IsInf  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto detect_positive = fetch_optional_int(stack);
            auto detect_negative = fetch_optional_int(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_IsInf(X, Y, detect_negative, detect_positive) != YNX_OK ) {
                yannx_panic("API: IsInf  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(IsInf)
    };


    struct GatherElements : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(indices);
            infer_.new_input(data);

            auto f = query_inference_function("GatherElements");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( data->onnx_GatherElements(data, indices, output, axis) != YNX_OK ) {
                yannx_panic("API: GatherElements  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_GatherElements(data, indices, output, axis) != YNX_OK ) {
                yannx_panic("API: GatherElements  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GatherElements)
    };


    struct ScatterElements : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto axis = fetch_optional_int(stack);

            auto updates = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);

            output = TensorType::create_undefined_user_tensor();


#ifdef USING_ONNX_IMPL
            YNXInferenceContextImpl infer_(1);
            if ( axis.index() != 0) {
                infer_.new_attr("axis", std::get<1>(axis) );
            }

            infer_.new_input(updates);
            infer_.new_input(indices);
            infer_.new_input(data);

            auto f = query_inference_function("ScatterElements");
            infer_.do_inference(f);
            infer_.check_output(0, output);

#endif

            if ( data->onnx_ScatterElements(data, indices, updates, output, axis) != YNX_OK ) {
                yannx_panic("API: ScatterElements  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {
            auto axis = fetch_optional_int(stack);

            auto updates = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_ScatterElements(data, indices, updates, output, axis) != YNX_OK ) {
                yannx_panic("API: ScatterElements  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ScatterElements)
    };

}


//
//  Registering all words
//
void register_all_onnx_defined_words( Runtime<TensorType>& runtime) {

    runtime.new_nword("NewConstant", common::Constant::creator);
    runtime.new_nword("NewVariable", common::Variable::creator);

    runtime.new_nword("onnx.RandomNormalLike", generator::RandomNormalLike::creator);
    runtime.new_nword("onnx.RandomNormal", generator::RandomNormal::creator);
    runtime.new_nword("onnx.RandomUniform", generator::RandomUniform::creator);
    runtime.new_nword("onnx.EyeLike", generator::EyeLike::creator);
    runtime.new_nword("onnx.Bernoulli", generator::Bernoulli::creator);
    runtime.new_nword("onnx.Multinomial", generator::Multinomial::creator);
    runtime.new_nword("onnx.RandomUniformLike", generator::RandomUniformLike::creator);
    runtime.new_nword("onnx.Range", generator::Range::creator);
    runtime.new_nword("onnx.GreaterOrEqual", logical::GreaterOrEqual::creator);
    runtime.new_nword("onnx.Or", logical::Or::creator);
    runtime.new_nword("onnx.BitShift", logical::BitShift::creator);
    runtime.new_nword("onnx.Greater", logical::Greater::creator);
    runtime.new_nword("onnx.Xor", logical::Xor::creator);
    runtime.new_nword("onnx.And", logical::And::creator);
    runtime.new_nword("onnx.LessOrEqual", logical::LessOrEqual::creator);
    runtime.new_nword("onnx.Not", logical::Not::creator);
    runtime.new_nword("onnx.Equal", logical::Equal::creator);
    runtime.new_nword("onnx.Less", logical::Less::creator);
    runtime.new_nword("onnx.Reciprocal", math::Reciprocal::creator);
    runtime.new_nword("onnx.LeakyRelu", math::LeakyRelu::creator);
    runtime.new_nword("onnx.HardSigmoid", math::HardSigmoid::creator);
    runtime.new_nword("onnx.Div", math::Div::creator);
    runtime.new_nword("onnx.Pow", math::Pow::creator);
    runtime.new_nword("onnx.Mul", math::Mul::creator);
    runtime.new_nword("onnx.Min", math::Min::creator);
    runtime.new_nword("onnx.Floor", math::Floor::creator);
    runtime.new_nword("onnx.Mean", math::Mean::creator);
    runtime.new_nword("onnx.Max", math::Max::creator);
    runtime.new_nword("onnx.Round", math::Round::creator);
    runtime.new_nword("onnx.Sigmoid", math::Sigmoid::creator);
    runtime.new_nword("onnx.Relu", math::Relu::creator);
    runtime.new_nword("onnx.LogSoftmax", math::LogSoftmax::creator);
    runtime.new_nword("onnx.Ceil", math::Ceil::creator);
    runtime.new_nword("onnx.Log", math::Log::creator);
    runtime.new_nword("onnx.Neg", math::Neg::creator);
    runtime.new_nword("onnx.Sub", math::Sub::creator);
    runtime.new_nword("onnx.PRelu", math::PRelu::creator);
    runtime.new_nword("onnx.Add", math::Add::creator);
    runtime.new_nword("onnx.Selu", math::Selu::creator);
    runtime.new_nword("onnx.Abs", math::Abs::creator);
    runtime.new_nword("onnx.QLinearMatMul", math::QLinearMatMul::creator);
    runtime.new_nword("onnx.Clip", math::Clip::creator);
    runtime.new_nword("onnx.Einsum", math::Einsum::creator);
    runtime.new_nword("onnx.Hardmax", math::Hardmax::creator);
    runtime.new_nword("onnx.Sqrt", math::Sqrt::creator);
    runtime.new_nword("onnx.Gemm", math::Gemm::creator);
    runtime.new_nword("onnx.Cos", math::Cos::creator);
    runtime.new_nword("onnx.Exp", math::Exp::creator);
    runtime.new_nword("onnx.Tan", math::Tan::creator);
    runtime.new_nword("onnx.Softmax", math::Softmax::creator);
    runtime.new_nword("onnx.SoftmaxCrossEntropyLoss", math::SoftmaxCrossEntropyLoss::creator);
    runtime.new_nword("onnx.Softsign", math::Softsign::creator);
    runtime.new_nword("onnx.Sum", math::Sum::creator);
    runtime.new_nword("onnx.Sinh", math::Sinh::creator);
    runtime.new_nword("onnx.Tanh", math::Tanh::creator);
    runtime.new_nword("onnx.TopK", math::TopK::creator);
    runtime.new_nword("onnx.Acos", math::Acos::creator);
    runtime.new_nword("onnx.Asin", math::Asin::creator);
    runtime.new_nword("onnx.Atan", math::Atan::creator);
    runtime.new_nword("onnx.Sign", math::Sign::creator);
    runtime.new_nword("onnx.Sin", math::Sin::creator);
    runtime.new_nword("onnx.MatMul", math::MatMul::creator);
    runtime.new_nword("onnx.Expand", math::Expand::creator);
    runtime.new_nword("onnx.Elu", math::Elu::creator);
    runtime.new_nword("onnx.Cosh", math::Cosh::creator);
    runtime.new_nword("onnx.Asinh", math::Asinh::creator);
    runtime.new_nword("onnx.Acosh", math::Acosh::creator);
    runtime.new_nword("onnx.Atanh", math::Atanh::creator);
    runtime.new_nword("onnx.Erf", math::Erf::creator);
    runtime.new_nword("onnx.Mod", math::Mod::creator);
    runtime.new_nword("onnx.ThresholdedRelu", math::ThresholdedRelu::creator);
    runtime.new_nword("onnx.MatMulInteger", math::MatMulInteger::creator);
    runtime.new_nword("onnx.Celu", math::Celu::creator);
    runtime.new_nword("onnx.CumSum", math::CumSum::creator);
    runtime.new_nword("onnx.Softplus", math::Softplus::creator);
    runtime.new_nword("onnx.NegativeLogLikelihoodLoss", math::NegativeLogLikelihoodLoss::creator);
    runtime.new_nword("onnx.Det", math::Det::creator);
    runtime.new_nword("onnx.HardSwish", math::HardSwish::creator);
    runtime.new_nword("onnx.LRN", nn::LRN::creator);
    runtime.new_nword("onnx.LpPool", nn::LpPool::creator);
    runtime.new_nword("onnx.Dropout", nn::Dropout::creator);
    runtime.new_nword("onnx.MaxPool", nn::MaxPool::creator);
    runtime.new_nword("onnx.GlobalLpPool", nn::GlobalLpPool::creator);
    runtime.new_nword("onnx.LpNormalization", nn::LpNormalization::creator);
    runtime.new_nword("onnx.Conv", nn::Conv::creator);
    runtime.new_nword("onnx.GlobalMaxPool", nn::GlobalMaxPool::creator);
    runtime.new_nword("onnx.MaxUnpool", nn::MaxUnpool::creator);
    runtime.new_nword("onnx.AveragePool", nn::AveragePool::creator);
    runtime.new_nword("onnx.InstanceNormalization", nn::InstanceNormalization::creator);
    runtime.new_nword("onnx.Flatten", nn::Flatten::creator);
    runtime.new_nword("onnx.GlobalAveragePool", nn::GlobalAveragePool::creator);
    runtime.new_nword("onnx.MaxRoiPool", nn::MaxRoiPool::creator);
    runtime.new_nword("onnx.BatchNormalization", nn::BatchNormalization::creator);
    runtime.new_nword("onnx.StringNormalizer", nn::StringNormalizer::creator);
    runtime.new_nword("onnx.Shrink", nn::Shrink::creator);
    runtime.new_nword("onnx.MeanVarianceNormalization", nn::MeanVarianceNormalization::creator);
    runtime.new_nword("onnx.ConvInteger", nn::ConvInteger::creator);
    runtime.new_nword("onnx.QLinearConv", nn::QLinearConv::creator);
    runtime.new_nword("onnx.ConvTranspose", nn::ConvTranspose::creator);
    runtime.new_nword("onnx.TfIdfVectorizer", nn::TfIdfVectorizer::creator);
    runtime.new_nword("onnx.RoiAlign", object_detection::RoiAlign::creator);
    runtime.new_nword("onnx.NonMaxSuppression", object_detection::NonMaxSuppression::creator);
    runtime.new_nword("onnx.QuantizeLinear", quantization::QuantizeLinear::creator);
    runtime.new_nword("onnx.DynamicQuantizeLinear", quantization::DynamicQuantizeLinear::creator);
    runtime.new_nword("onnx.DequantizeLinear", quantization::DequantizeLinear::creator);
    runtime.new_nword("onnx.ReduceProd", reduction::ReduceProd::creator);
    runtime.new_nword("onnx.ReduceMin", reduction::ReduceMin::creator);
    runtime.new_nword("onnx.ReduceSumSquare", reduction::ReduceSumSquare::creator);
    runtime.new_nword("onnx.ReduceSum", reduction::ReduceSum::creator);
    runtime.new_nword("onnx.ReduceLogSumExp", reduction::ReduceLogSumExp::creator);
    runtime.new_nword("onnx.ReduceMax", reduction::ReduceMax::creator);
    runtime.new_nword("onnx.ArgMax", reduction::ArgMax::creator);
    runtime.new_nword("onnx.ArgMin", reduction::ArgMin::creator);
    runtime.new_nword("onnx.ReduceLogSum", reduction::ReduceLogSum::creator);
    runtime.new_nword("onnx.ReduceMean", reduction::ReduceMean::creator);
    runtime.new_nword("onnx.ReduceL2", reduction::ReduceL2::creator);
    runtime.new_nword("onnx.ReduceL1", reduction::ReduceL1::creator);
    runtime.new_nword("onnx.LSTM", rnn::LSTM::creator);
    runtime.new_nword("onnx.GRU", rnn::GRU::creator);
    runtime.new_nword("onnx.RNN", rnn::RNN::creator);
    runtime.new_nword("onnx.SequenceEmpty", sequence::SequenceEmpty::creator);
    runtime.new_nword("onnx.SplitToSequence", sequence::SplitToSequence::creator);
    runtime.new_nword("onnx.SequenceAt", sequence::SequenceAt::creator);
    runtime.new_nword("onnx.SequenceLength", sequence::SequenceLength::creator);
    runtime.new_nword("onnx.SequenceConstruct", sequence::SequenceConstruct::creator);
    runtime.new_nword("onnx.SequenceInsert", sequence::SequenceInsert::creator);
    runtime.new_nword("onnx.SequenceErase", sequence::SequenceErase::creator);
    runtime.new_nword("onnx.ConcatFromSequence", sequence::ConcatFromSequence::creator);
    runtime.new_nword("onnx.CastLike", tensor::CastLike::creator);
    runtime.new_nword("onnx.Shape", tensor::Shape::creator);
    runtime.new_nword("onnx.Reshape", tensor::Reshape::creator);
    runtime.new_nword("onnx.DepthToSpace", tensor::DepthToSpace::creator);
    runtime.new_nword("onnx.Concat", tensor::Concat::creator);
    runtime.new_nword("onnx.Gather", tensor::Gather::creator);
    runtime.new_nword("onnx.Size", tensor::Size::creator);
    runtime.new_nword("onnx.Cast", tensor::Cast::creator);
    runtime.new_nword("onnx.Split", tensor::Split::creator);
    runtime.new_nword("onnx.Identity", tensor::Identity::creator);
    runtime.new_nword("onnx.Slice", tensor::Slice::creator);
    runtime.new_nword("onnx.GatherND", tensor::GatherND::creator);
    runtime.new_nword("onnx.SpaceToDepth", tensor::SpaceToDepth::creator);
    runtime.new_nword("onnx.Squeeze", tensor::Squeeze::creator);
    runtime.new_nword("onnx.Unique", tensor::Unique::creator);
    runtime.new_nword("onnx.IsNaN", tensor::IsNaN::creator);
    runtime.new_nword("onnx.Tile", tensor::Tile::creator);
    runtime.new_nword("onnx.ReverseSequence", tensor::ReverseSequence::creator);
    runtime.new_nword("onnx.Transpose", tensor::Transpose::creator);
    runtime.new_nword("onnx.Trilu", tensor::Trilu::creator);
    runtime.new_nword("onnx.Where", tensor::Where::creator);
    runtime.new_nword("onnx.Compress", tensor::Compress::creator);
    runtime.new_nword("onnx.Unsqueeze", tensor::Unsqueeze::creator);
    runtime.new_nword("onnx.OneHot", tensor::OneHot::creator);
    runtime.new_nword("onnx.NonZero", tensor::NonZero::creator);
    runtime.new_nword("onnx.ScatterND", tensor::ScatterND::creator);
    runtime.new_nword("onnx.Resize", tensor::Resize::creator);
    runtime.new_nword("onnx.Pad", tensor::Pad::creator);
    runtime.new_nword("onnx.IsInf", tensor::IsInf::creator);
    runtime.new_nword("onnx.GatherElements", tensor::GatherElements::creator);
    runtime.new_nword("onnx.ScatterElements", tensor::ScatterElements::creator);


}

}
#endif
