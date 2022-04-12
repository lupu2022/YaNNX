//
//  this file is geneated by onnx~/autogen
//

#ifndef _YANXX_TENSORTYPE_HPP_
#define _YANNX_TENSORTYPE_HPP_

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#ifdef USING_ONNX
#include <onnx/onnx_pb.h>
#include <onnx/defs/schema.h>
#include <onnx/defs/attr_proto_util.h>
#include <onnx/shape_inference/implementation.h>
#endif

#include <yannx.hpp>

//
//  A simple onnx based (type and shape inference only, or a pure dummy tensor ) Tensor.following ONNX IR
//  https://github.com/onnx/onnx/blob/main/docs/IR.md
//  https://github.com/onnx/onnx/blob/main/docs/Operators.md
//

namespace yannx {

enum OperatorReturnType {
    YNX_OK = 0,
    YNX_TODO_ERROR = -1,
    YNX_INPUT_ERROR = -2,
    YNX_OUTPUT_ERROR = -3,
    YNX_ATTR_ERROR = -4,
};

enum TensorDataType {
    YNX_UNDEFINED,
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
    YNX_float,
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
    "YNX_float",
    "YNX_UINT32",
    "YNX_UINT64",
    "YNX_COMPLEX64",
    "YNX_COMPLEX128",
    "YNX_BFLOAT16"
};


/*
 *  https://github.com/onnx/onnx/blob/main/docs/IR.md#tensor-definition
 *  scalar:         an empty shape with a defined data type
 *  tensor:         shape dimention > 0
 *  undefined:      empty shape with a undefined data type, used for type_shape inference.
 */

struct TensorType;
using tensor_t = std::shared_ptr<TensorType>;

//
//  User must be re-implement, return user side undefined tensor!
//
extern tensor_t create_undefined_tensor();

struct TensorType {
    TensorDataType dtype_;
    std::vector<size_t> shape_;

    TensorType() {
        dtype_ = TensorDataType::YNX_UNDEFINED;
    }
    TensorType(TensorDataType dtype, std::vector<size_t>& shape) {
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

#ONNX_DEF#

};

#ifdef USING_ONNX
struct YNXInferenceContextImpl : public InferenceContext {
    const size_t num_inputs_;
    const size_t num_outputs_;

    std::map<std::string, AttributeProto> attrs_;

    YNXInferenceContextImpl(size_t in, size_t out) : num_inputs_(in), num_outputs_(out) {
    }

    // setup interfaces
    void new_attr(const std::string& name, float v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, int64_t v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, std::string& v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }



    // InferenceContext apis
    size_t getNumInputs() const override {
        return num_inputs_;
    }
    size_t getNumOutputs() const override {
        return num_outputs_;
    }
    const TypeProto* getInputType(size_t index) const override {
        return nullptr;
    }
    const AttributeProto* getAttribute(const std::string& name) override {
        return nullptr;
    }
    const TensorProto* getInputData(size_t index) const override {
        return nullptr;
    }
    virtual TypeProto* getOutputType(size_t index) override {
        return nullptr;
    }

    // Skipping these impl
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

#define NWORD_CREATOR_DEFINE_TENSORTYPE(CLS)                                                \
    static std::shared_ptr<NativeWord<TensorType> >   creator(Runtime<TensorType>& rt ) {   \
        std::shared_ptr<NativeWord<TensorType> > wd(new CLS());                             \
        return wd;                                                                          \
    }

#ONNX_IMPL#



}
#endif
