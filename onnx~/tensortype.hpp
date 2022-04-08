#ifndef _ONNX_IMPL_HPP_
#define _ONNX_IMPL_HPP_

#include <vector>
#include <string>
#include <algorithm>

#include <yannx.hpp>

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>

//
//  A simple Dummy Tensor followed Onnx's define
//  https://github.com/onnx/onnx/blob/main/docs/IR.md
//  https://github.com/onnx/onnx/blob/main/docs/Operators.md
//


namespace yannx {

enum TensorDataType {
    YNX_UNDEFINED = 0,
    YNX_FLOAT = 1,
    YNX_UINT8 = 2,
    YNX_INT8 = 3,
    YNX_UINT16 = 4,
    YNX_INT16 = 5,
    YNX_INT32 = 6,
    YNX_INT64 = 7,
    YNX_STRING = 8,
    YNX_BOOL = 9,
    YNX_FLOAT16 = 10,
    YNX_DOUBLE = 11,
    YNX_UINT32 = 12,
    YNX_UINT64 = 13,
    YNX_COMPLEX64 = 14,
    YNX_COMPLEX128 = 15,
    YNX_BFLOAT16 = 16
};

enum OperatorReturnType {
    YNX_SUCCESS = 0,
    YNX_TODO_ERROR = -1,
    YNX_INPUT_ERROR = -2,
    YNX_OUTPUT_ERROR = -3,
    YNX_ATTR_ERROR = -4,
};




/*
 *  null tensor:    an empty shape with a undefined data type
 *  scalar:         an empty shape with a defined data type
 *  tensor:         shape dimention > 0
 *
 *  https://github.com/onnx/onnx/blob/main/docs/IR.md#tensor-definition
 */

struct TensorType {
    TensorDataType dtype_;
    std::vector<size_t> shape_;

    TensorType(TensorElementType dtype, std::vector<size_t>& shape) {
        dtype_ = dtype;
        shape_ = shape;
    }
    using tensor_t = std::shared_ptr<TensorType>;

#include "onnx_def.hpp"

};

#include "onnx_impl.hpp"

}
#endif
