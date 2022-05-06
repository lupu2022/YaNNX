#ifndef _YANNX_TENSORTYPE_HPP_
#define _YANNX_TENSORTYPE_HPP_

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#include <yannx.hpp>
#include <api.hpp>

#ifdef _USE_DNNL_CPU_
#include "dnnl/impl.hpp"
using device_float_t = yannx::dnnl::DNNLTensor<tt::TensorDataType::YNX_FLOAT>;
#endif

#ifdef _USE_CUDA_GPU_
#include "cuda/impl.hpp"
using device_float_t = yannx::dnnl::DNNLTensor<tt::TensorDataType::YNX_FLOAT>;
#endif

namespace yannx { namespace tt {

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

struct TensorType;
using tensor_t = std::shared_ptr<TensorType>;

template <class T, TensorDataType _DTYPE_>
struct ValueOnlyTensor {
public:
    ValueOnlyTensor(std::vector<size_t>& shape) {
        size_t n = 1;
        for (size_t i = 0; i < shape_.size(); i++) {
            n = n * shape[i];
        }
        value_.resize(n);
    }
    ValueOnlyTensor(std::vector<size_t>& shape, const void* pdata){
        size_t n = 1;
        for (size_t i = 0; i < shape_.size(); i++) {
            n = n * shape[i];
        }
        value_.resize(n);
        memcpy( value_.data(), pdata, sizeof(T) * n );
    }
    ValueOnlyTensor(const void* pvalue) {
        value_.resize(1);
        value_[0] = *(const T*)pvalue;
    }

    // value is only you need
    const void* value() {
        return (const void*)(value_.data());
    }
    TensorDataType dtype() {
        return _DTYPE_;
    }

private:
    std::vector<T> value_;
};
// for scalar or un-supported data type
using value_float_t = ValueOnlyTensor<float, TensorDataType::YNX_FLOAT>;
using value_int64_t = ValueOnlyTensor<int64_t, TensorDataType::YNX_INT64>;
using value_bool_t = ValueOnlyTensor<unsigned char, TensorDataType::YNX_BOOL>;

/*
 *  https://github.com/onnx/onnx/blob/main/docs/IR.md#tensor-definition
 *  scalar:         an empty shape with a defined data type
 *  tensor:         shape dimention > 0
 *  undefined:      empty shape with a undefined data type, used for type_shape inference.
 */
struct TensorType : public OnnxOperatorSet {
public:
    // default is a undefined tensor
    TensorType() : dtype_(YNX_UNDEFINED), impl_((void*)NULL) {
        yannx_panic("Can't support target device type!");
    }
    ~TensorType() {}

    // fast access
    TensorDataType dtype() {
        return dtype_;
    }
    const std::vector<size_t>& shape() {
        return shape_;
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
    size_t items() {
        if ( dtype() == YNX_UNDEFINED ) {
            return 0;
        }
        size_t n = 1;
        auto shape_ = shape();
        for ( size_t i = 0; i < shape_.size(); i++) {
            n = n * shape_[i];
        }
        return n;
    }
    bool is_scalar() {
        if ( dtype() == YNX_UNDEFINED ) {
            return false;
        }
        if ( shape().size() == 0) {
            return true;
        }
        return false;
    }
    const void* value() {
        if ( impl_.index() == VALUE_FLOAT ) {
            return std::get<VALUE_FLOAT>(impl_)->value();
        }
        if ( impl_.index() == VALUE_INT64 ) {
            return std::get<VALUE_INT64>(impl_)->value();
        }
        if ( impl_.index() == VALUE_BOOL ) {
            return std::get<VALUE_BOOL>(impl_)->value();
        }
        yannx_panic("Can't call value() from none value Tensor");
        return nullptr;
    }

    // reset to a normal tensor
    void reset(TensorDataType dtype, std::vector<size_t>& shape) {
        yannx_assert(dtype == YNX_UNDEFINED, "Can't reset a typed tensor!");
        yannx_assert(shape.size() > 0, "Can't reset a typed tensor with zero shape!");
        yannx_assert(impl_.index() == UNKNOW_ANY, "Can't reset a setted tensor!");

        dtype_ = dtype;
        shape_ = shape;

        if ( dtype == YNX_FLOAT ) {
            impl_ = std::make_unique<device_float_t>(shape);
            return;
        }

        if ( dtype == YNX_INT64 ) {
            impl_ = std::make_unique<value_int64_t>(shape);
            return;
        }

        yannx_panic("DeviceTensor::reset can't be here!");
    }

    // reset to a normal tensor with filled data
    void reset(TensorDataType dtype, std::vector<size_t>& shape, const void* pdata) {
        yannx_assert(dtype == YNX_UNDEFINED, "Can't reset a typed tensor!");
        yannx_assert(shape.size() > 0, "Can't reset a typed tensor with zero shape!");
        yannx_assert(impl_.index() == UNKNOW_ANY, "Can't reset a setted tensor!");

        dtype_ = dtype;
        shape_ = shape;

        if ( dtype == YNX_FLOAT ) {
            impl_ = std::make_unique<device_float_t>(shape, pdata);
            return;
        }

        if ( dtype == YNX_INT64) {
            impl_ = std::make_unique<value_int64_t>(shape, pdata);
            return;
        }

        yannx_panic("DeviceTensor::reset can't be here!");
    }

    // reset to a scalar tensor
    void reset(TensorDataType dtype, const void* pvalue) {
        yannx_assert(dtype == YNX_UNDEFINED, "Can't reset a typed tensor!");
        yannx_assert(impl_.index() == UNKNOW_ANY, "Can't reset a setted tensor!");

        dtype_ = dtype;
        shape_.clear();

        if ( dtype == YNX_FLOAT) {
            impl_ = std::make_unique<value_float_t>(pvalue);
            return;
        }
        if ( dtype == YNX_INT64) {
            impl_ = std::make_unique<value_int64_t>(pvalue);
            return;
        }

        yannx_panic("DeviceTensor::reset can't be here!");
    }

    //
    //  Realy tensor computing API following ONNX define
    //
    const char* device() override {
        if ( impl_.index() == DEVICE_FLOAT ) {
            return std::get<DEVICE_FLOAT>(impl_)->device();
        }
        return "ValueOnly";
    }

#include "api_impl.inc"

private:
    TensorType* impl() {
        if ( impl_.index() == DEVICE_FLOAT ) {
            return (TensorType *) std::get<DEVICE_FLOAT>(impl_).get();
        }
        yannx_panic("Can't get impl from a none device tensor");
        return nullptr;
    }

private:
    // basic info about tensor
    TensorDataType      dtype_;
    std::vector<size_t> shape_;

    // ImplType enum order is same as TensorImpl's variant
    enum ImplType {
        UNKNOW_ANY = 0,
        DEVICE_FLOAT,
        VALUE_FLOAT,
        VALUE_INT64,
        VALUE_BOOL,
    };
    using TensorImpl = std::variant< void *,
                                     std::unique_ptr<device_float_t>,
                                     std::unique_ptr<value_float_t>,
                                     std::unique_ptr<value_int64_t>,
                                     std::unique_ptr<value_bool_t> >;
    TensorImpl impl_;

public:
    //
    //  User must be re-implement, return user side undefined tensor!
    //
    static tensor_t create_undefined_user_tensor();
    static void register_user_tensor(tensor_t, int64_t flag);
};

}} // end of namespace
#endif
