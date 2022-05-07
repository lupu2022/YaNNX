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
    virtual const char* device() = 0;
    virtual const void* value_(const std::vector<size_t>& pos) = 0;  // read a item via a position
    virtual const void* value_() = 0;                                // read a scalar

    // following is ONNX operator set
#ONNX_DEF#

};

}}

#endif
