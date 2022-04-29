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
using dnnl_float_t = yannx::dnnl::DNNLTensor<tt::TensorDataType::YNX_FLOAT>;
#else
struct DNNL_FloatTensor{};
using dnnl_float_t = DNNL_FloatTensor;
#endif

#ifdef _USE_CUDA_GPU_
namespace cuda {
    template <tt::TensorDataType _DTYPE_> struct CUDATensor;
}
using cuda_float_t = yannx::cuda::CUDATensor<tt::TensorDataType::YNX_FLOAT>;
#else
struct CUDA_FloatTensor{};
using cuda_float_t = CUDA_FloatTensor;
#endif

namespace tt {

template <class T, tt::TensorDataType _DTYPE_>
struct ValueOnlyTensor: public TensorType {
public:
    std::vector<T> value_;
    std::vector<size_t> shape_;

    const void* value() override {
        return (const void*)value_.data();
    }

    void reset(tt::TensorDataType dtype, std::vector<size_t>& shape) override {
        shape_ = shape;
        size_t n = 1;
        for (size_t i = 0; i < shape_.size(); i++) {
            n = n * shape_[i];
        }
        value_.resize(n);
    }
    void reset(tt::TensorDataType dtype, std::vector<size_t>& shape, const void* pdata) override {
        shape_ = shape;
        size_t n = 1;
        for (size_t i = 0; i < shape_.size(); i++) {
            n = n * shape_[i];
        }
        memcpy( value_.data(), pdata, sizeof(T) * n );
    }

    void reset(tt::TensorDataType dtype, const void* pvalue) override {
        value_.resize(1);
        value_[0] = *(const T*)pvalue;
    }

    tt::TensorDataType dtype() override { return _DTYPE_; }
    const std::vector<size_t>& shape() override { return shape_; }
};

using value_int64_t = ValueOnlyTensor<int64_t, TensorDataType::YNX_INT64>;

struct DeviceTensor: public TensorType  {
public:
    enum DeviceType {
        DEVICE_CPU = 0,
        DEVICE_CUDA_0 = 1,
        DEVICE_CUDA_1 = 2,
        DEVICE_CUDA_2 = 3,
        DEVICE_CUDA_3 = 4,
        DEVICE_CUDA_X = 100,
    };

public:
    // init functions, return an undefined tensor
    DeviceTensor(DeviceType device) : device_(device), dtype_(tt::TensorDataType::YNX_UNDEFINED), impl_((void*)NULL) {
#ifdef _USE_DNNL_CPU_
        if ( device_ == DEVICE_CPU ) {
            return;
        }
#endif
#ifdef _USE_CUDA_GPU_
        if ( device_ >= DEVICE_CUDA_0 && device_ <= DEVICE_CUDA_X ) {
            return;
        }
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

    void reset(tt::TensorDataType dtype, std::vector<size_t>& shape) override;
    void reset(tt::TensorDataType dtype, std::vector<size_t>& shape, const void* pdata) override;
    void reset(tt::TensorDataType dtype, const void* pvalue) override;

    const void* value() override {
        return impl()->value();
    }

#include "api_impl.inc"

private:
    TensorType* impl() {
#ifdef _USE_DNNL_CPU_
        if ( impl_.index() == DNNL_FLOAT ) {
            return (TensorType *) std::get<DNNL_FLOAT>(impl_).get();
        }
#endif

#ifdef _USE_CUDA_GPU_
        if ( impl_.index() == CUDA_FLOAT ) {
            return (TensorType *) std::get<CUDA_FLOAT>(impl_).get();
        }
#endif
        yannx_panic("Can't get impl from a UNKNOW_ANY tensor");
        return nullptr;
    }

private:
    // basic info about tensor
    const DeviceType    device_;
    TensorDataType      dtype_;
    std::vector<size_t> shape_;

    // ImplType enum order is same as TensorImpl's variant
    enum ImplType {
        UNKNOW_ANY = 0,
        DNNL_FLOAT,
        CUDA_FLOAT,
        VALUE_INT64
    };
    using TensorImpl = std::variant< void *,
                                     std::unique_ptr<dnnl_float_t>,
                                     std::unique_ptr<cuda_float_t>,
                                     std::unique_ptr<value_int64_t> >;
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
    auto* p = new DeviceTensor(DeviceTensor::DEVICE_CPU);
    std::shared_ptr<TensorType> ret(p);
    return ret;
}

void  TensorType::register_user_tensor(std::shared_ptr<TensorType> tensor, int64_t flag) {

}

// reset to a normal tensor
void DeviceTensor::reset(TensorDataType dtype, std::vector<size_t>& shape) {
    yannx_assert(dtype == YNX_UNDEFINED, "Can't reset a typed tensor!");
    if ( device_ == DEVICE_CPU ) {
        if ( dtype == YNX_FLOAT ) {
            impl_ = std::make_unique<yannx::dnnl::DNNLTensor<YNX_FLOAT>>(shape);
        }
        return;
    }
    if ( device_ >= DEVICE_CUDA_0 && device_ <= DEVICE_CUDA_X ) {
        return;
    }
}

// reset to a normal tensor with filled data
void DeviceTensor::reset(TensorDataType dtype, std::vector<size_t>& shape, const void* pdata) {
    yannx_assert(dtype == YNX_UNDEFINED, "Can't reset a typed tensor!");

}

// reset to a scalar tensor
void DeviceTensor::reset(TensorDataType dtype, const void* pvalue) {
    yannx_assert(dtype == YNX_UNDEFINED, "Can't reset a typed tensor!");
}

}} // end of namespace
#endif
