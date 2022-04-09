#ifndef _ONNX_IMPL_HPP_
#define _ONNX_IMPL_HPP_

#include <vector>
#include <string>
#include <algorithm>

#include <yannx.hpp>

//
//  A simple Dummy Tensor followed Onnx's define
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

/*
 *  https://github.com/onnx/onnx/blob/main/docs/IR.md#tensor-definition
 *  scalar:         an empty shape with a defined data type
 *  tensor:         shape dimention > 0
 *  undefined:      an empty shape with a undefined data type, used for typing output.
 */

struct TensorType;
using tensor_t = std::shared_ptr<TensorType>;

struct TensorType {
    TensorDataType dtype_;
    std::vector<size_t> shape_;

    TensorType() {
        dtype_ = TensorElementType::YNX_UNDEFINED;
    }
    TensorType(TensorElementType dtype, std::vector<size_t>& shape) {
        dtype_ = dtype;
        shape_ = shape;
    }

    #include "onnx_defs.hpp"
};

//
//  User must be re-implement, return user side undefined tensor!
//
std::shared_ptr<TensorType> create_undefined_tensor() {
    return nullptr;
}

//
//  some common help functions, and none-auto operators
//
static double fetch_float(ValueStack<TensorType>& stack) {
    double v = stack.pop_number();
    return v;
}

static long fetch_int(ValueStack<TensorType>& stack) {
    long v = stack.pop_number();
    return v;
}

static std::string fetch_string(ValueStack<TensorType>& stack) {
    std::string v = stack.pop_string();
    return v;
}

static double fetch_tensor(ValueStack<TensorType>& stack) {
    double v = stack.pop_tensor();
    return v;
}

static std::vevtor<double> fetch_floats(ValueStack<TensorType>& stack) {
    auto v = stack.pop_number_tuple();
    std::vevtor<double> ret;
    for (size_t i = 0; i < v.size(); i++) {
        ret.push_back( v[i] );
    }
    return ret;
}

static std::vevtor<long> fetch_ints(ValueStack<TensorType>& stack) {
    auto v = stack.pop_number_tuple();
    std::vevtor<long> ret;
    for (size_t i = 0; i < v.size(); i++) {
        ret.push_back( v[i] );
    }
    return ret;
}

static std::vevtor<std::string> fetch_strings(ValueStack<TensorType>& stack) {
    auto v = stack.pop_string_tuple();
    return v;
}

static std::vevtor<std::string> fetch_tensors(ValueStack<TensorType>& stack) {
    auto v = stack.pop_tensor_tuple();
    return v;
}

static std::variant<void *, double> fetch_optional_float(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, double>(nullptr);
    }
    return std::variant<void *, double>( fetch_float(stack) );
}

static std::variant<void *, long> fetch_optional_int(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, long>(nullptr);
    }
    return std::variant<void *, long>( fetch_int(stack) );
}

static std::variant<void *, std::string> fetch_optional_string(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::string>(nullptr);
    }
    return std::variant<void *, std::string>( fetch_string(stack) );
}

static std::variant<void *, tensor_t> fetch_optional_tensor(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, tensor>(nullptr);
    }
    return std::variant<void *, tensor_t>( fetch_tensor(stack) );
}

static std::variant<void *, std::vector<double> > fetch_optional_floats(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::vector<double> >(nullptr);
    }
    return std::variant<void *, std::vector<double> >( fetch_floats(stack) );
}

static std::variant<void *, std::vector<long> > fetch_optional_int(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::vector<long> >(nullptr);
    }
    return std::variant<void *, std::vector<long> >( fetch_ints(stack) );
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
        stack.push_tensor(t);
    }
    stack.push_number(ts.size());
}

static void put_optional_tensors(ValueStack<TensorType>& stack, std::variant<void*, tensor_t>& ot) {
    if ( ot.index() == 0) {
        stack.push_none();
        return;
    }
    stack.push_tensor( std::get<1>(ot) );
}

#include "onnx_impl.hpp"

}
#endif
