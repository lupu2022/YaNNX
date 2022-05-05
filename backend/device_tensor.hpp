#ifndef _DEVICE_TENSOR_HPP_
#define _DEVICE_TENSOR_HPP_

#include <vector>
#include <sstream>
#include <memory>
#include <cmath>
#include <cstring>
#include <random>

#include <yannx.hpp>
#include <tensortype.hpp>

// Follow definement is setup by Makefile
//#define _USE_DNNL_CPU_
//#define _USE_CUDA_GPU_

namespace yannx {

#ifdef _USE_DNNL_CPU_
namespace dnnl {
    template <tt::TensorDataType _DTYPE_> struct DNNLTensor;
}
using device_float_t = yannx::dnnl::DNNLTensor<tt::TensorDataType::YNX_FLOAT>;
#endif

#ifdef _USE_CUDA_GPU_
namespace cuda {
    template <tt::TensorDataType _DTYPE_> struct CUDATensor;
}
using device_float_t = yannx::cuda::CUDATensor<tt::TensorDataType::YNX_FLOAT>;
#endif

namespace tt {

template <class T, tt::TensorDataType _DTYPE_>
struct ValueOnlyTensor: public TensorType {
public:
    std::vector<T> value_;
    std::vector<size_t> shape_;

    ValueOnlyTensor(std::vector<size_t>& shape) {
        shape_ = shape;
        size_t n = 1;
        for (size_t i = 0; i < shape_.size(); i++) {
            n = n * shape_[i];
        }
        value_.resize(n);
    }
    ValueOnlyTensor(std::vector<size_t>& shape, const void* pdata) {
        shape_ = shape;
        size_t n = 1;
        for (size_t i = 0; i < shape_.size(); i++) {
            n = n * shape_[i];
        }
        memcpy( value_.data(), pdata, sizeof(T) * n );
    }
    ValueOnlyTensor(const void* pvalue) {
        value_.resize(1);
        value_[0] = *(const T*)pvalue;
    }

    const char* device() override {
        return "ValueOnly";
    }

    // value is only you need
    const void* value() override {
        return (const void*)(value_.data());
    }

    // we don't need these functions, call these via DeviceTensor
    void reset(tt::TensorDataType dtype, std::vector<size_t>& shape) override {
        yannx_panic("Can't call this interface!");
    }
    void reset(tt::TensorDataType dtype, std::vector<size_t>& shape, const void* pdata) override {
        yannx_panic("Can't call this interface!");
    }
    void reset(tt::TensorDataType dtype, const void* pvalue) override {
        yannx_panic("Can't call this interface!");
    }
    tt::TensorDataType dtype() override { return _DTYPE_; }
    const std::vector<size_t>& shape() override { return shape_; }
};

// for scalar or un-supported data type
using value_float_t = ValueOnlyTensor<float, TensorDataType::YNX_FLOAT>;
using value_int64_t = ValueOnlyTensor<int64_t, TensorDataType::YNX_INT64>;
using value_bool_t = ValueOnlyTensor<unsigned char, TensorDataType::YNX_BOOL>;

struct DeviceTensor: public TensorType  {
public:
    // init functions, return an undefined tensor
    DeviceTensor() : dtype_(tt::TensorDataType::YNX_UNDEFINED), impl_((void*)NULL) {
#ifdef _USE_DNNL_CPU_

#endif

#ifdef _USE_CUDA_GPU_

#endif
        yannx_panic("Can't support target device type!");
    }

    // override functions
    const std::vector<size_t>& shape() override {
        return shape_;
    }
    tt::TensorDataType dtype() override {
        return dtype_;
    }
    const char* device() override {
        return impl()->device();
    }

    void reset(tt::TensorDataType dtype, std::vector<size_t>& shape) override;
    void reset(tt::TensorDataType dtype, std::vector<size_t>& shape, const void* pdata) override;
    void reset(tt::TensorDataType dtype, const void* pvalue) override;

    const void* value() override {
        return impl()->value();
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
};

}} // end of namespace

#ifdef _USE_DNNL_CPU_
#include "dnnl/impl.hpp"
#endif

#ifdef _USE_CUDA_GPU_
#include "cuda/impl.hpp"
#endif

namespace yannx { namespace tt {

std::shared_ptr<TensorType> TensorType::create_undefined_user_tensor() {
    auto* p = new DeviceTensor();
    std::shared_ptr<TensorType> ret(p);
    return ret;
}

void  TensorType::register_user_tensor(std::shared_ptr<TensorType> tensor, int64_t flag) {

}

// reset to a normal tensor
void DeviceTensor::reset(TensorDataType dtype, std::vector<size_t>& shape) {
    yannx_assert(dtype == YNX_UNDEFINED, "Can't reset a typed tensor!");
    yannx_assert(shape.size() > 0, "Can't reset a typed tensor with zero shape!");
    yannx_assert(impl_.index() == UNKNOW_ANY, "Can't reset a setted tensor!");

    if ( dtype == YNX_FLOAT ) {
        impl_ = std::make_unique<device_float_t>(shape);
        return;
    }

    if ( dtype == YNX_INT64 ) {
        impl_ = std::make_unique<value_int64_t>(shape);
        return;
    }

    yannx_panic("Can't be here: DeviceTensor::reset");
}

// reset to a normal tensor with filled data
void DeviceTensor::reset(TensorDataType dtype, std::vector<size_t>& shape, const void* pdata) {
    yannx_assert(dtype == YNX_UNDEFINED, "Can't reset a typed tensor!");
    yannx_assert(shape.size() > 0, "Can't reset a typed tensor with zero shape!");
    yannx_assert(impl_.index() == UNKNOW_ANY, "Can't reset a setted tensor!");

    if ( dtype == YNX_FLOAT ) {
        impl_ = std::make_unique<device_float_t>(shape, pdata);
        return;
    }

    if ( dtype == YNX_INT64) {
        impl_ = std::make_unique<value_int64_t>(shape, pdata);
        return;
    }

    yannx_panic("Can't be here: DeviceTensor::reset");
}

// reset to a scalar tensor
void DeviceTensor::reset(TensorDataType dtype, const void* pvalue) {
    yannx_assert(dtype == YNX_UNDEFINED, "Can't reset a typed tensor!");
    yannx_assert(impl_.index() == UNKNOW_ANY, "Can't reset a setted tensor!");

    if ( dtype == YNX_FLOAT) {
        impl_ = std::make_unique<value_float_t>(pvalue);
        return;
    }
    if ( dtype == YNX_INT64) {
        impl_ = std::make_unique<value_int64_t>(pvalue);
        return;
    }

    yannx_panic("Can't be here: DeviceTensor::reset");
}

}} // end of namespace
#endif
