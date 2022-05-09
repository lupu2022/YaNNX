//
//  this file is geneated by autogen
//

#ifndef _YANNX_TENSORAPI_HPP_
#define _YANNX_TENSORAPI_HPP_

#include <vector>
#include <variant>

namespace yannx { namespace tt {

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



struct TensorType;
using tensor_t = std::shared_ptr<TensorType>;

//
//  https://github.com/onnx/onnx/blob/main/docs/IR.md#tensor-definition
//  scalar:         an empty shape with a defined data type
//  tensor:         shape dimention > 0
//  undefined:      empty shape with a undefined data type, used for type_shape inference.
//
//  ONNX based tensor computing API
//  https://github.com/onnx/onnx/blob/main/docs/IR.md
//  https://github.com/onnx/onnx/blob/main/docs/Operators.md
//
struct TensorType {
public:
    // must be common operator
    virtual const char* device() = 0;
    virtual const std::vector<size_t>& shape()  = 0;
    virtual TensorDataType dtype() = 0;
    // io functions: single read & whole write
    virtual const void* item_(const std::vector<size_t>& position) = 0;
    virtual fill_(const void* pdata) = 0;

    // reset undefined to a defined
    virtual reset(TensorDataType dtype_, const std::vector<size_t>& shape_) = 0;
    virtual reset(TensorDataType dtype_, const void* pvalue) = 0;
    virtual reset(TensorDataType dtype_, const std::vector<size_t>& shape_, const void* pdata) = 0;

    // some fast access and help
    bool is_undefined() {
        if ( dtype_ == YNX_UNDEFINED ) {
            return true;
        }
        return false;
    }
    bool is_scalar() {
        if ( dtype_ != YNX_UNDEFINED && shape_.size() == 0) {
            return true;
        }
        return false;
    }
    size_t num_items() {
        auto s = shape();
        size_t n = 1;
        for (size_t i = 0; i < s.size(); i++) {
            n = n * s[i];
        }
        return n;
    }
    std::string to_string() {
        std::ostringstream ss;
        ss << TensorDataTypeString[ dtype() ];
        ss << ":[";
        for (size_t i = 0; i < shape().size(); i++) {
            ss << shape()[i];
            if (i != shape().size() - 1) {
                ss << " ";
            }
        }
        ss << "]";
        return ss.str();
    }

    template<typename T> T at(const std::vector<size_t>& position) {
        T r = *(const T *)pvalue(position);
        return T;
    }

    // following is ONNX operator set
#ONNX_DEF#
};

struct TensorFactory {
    virtual tensor_t create_undefined_tensor() = 0;
    virtual void register_user_tensor(tensor_t t, int64_t flag) = 0;
}

}}

#endif
