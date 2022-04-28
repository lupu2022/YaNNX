#ifndef _DEVICE_TENSOR_HPP_
#define _DEVICE_TENSOR_HPP_

#include <vector>
#include <sstream>
#include <memory>
#include <cmath>
#include <random>

#include <yannx.hpp>
#include <tensortype.hpp>

// Follow definement is setup by Makefile
//#define _USE_DNNL_CPU_
//#define _USE_CUDA_GPU_

namespace yannx {

#ifdef _USE_DNNL_CPU_
namespace dnnl {
    template <tt::TensorDataType _DTYPE_> struct CPUTensor;
}
using cpu_float_t = yannx::dnnl::CPUTensor<tt::TensorDataType::YNX_FLOAT>;
using cpu_int64_t = yannx::dnnl::CPUTensor<tt::TensorDataType::YNX_INT64>;
#else
struct CPU_FloatTensor{};
struct CPU_Int64Tensor{};
using cpu_float_t = CPU_FloatTensor;
using cpu_int64_t = CPU_Int64Tensor;
#endif

#ifdef _USE_CUDA_GPU_
namespace cuda {
    template <tt::TensorDataType _DTYPE_> struct CUDATensor;
}
using cuda_float_t = yannx::cuda::CUDATensor<tt::TensorDataType::YNX_FLOAT>;
using cuda_int64_t = yannx::cuda::CUDATensor<tt::TensorDataType::YNX_INT64>;
#else
struct CUDA_FloatTensor{};
struct CUDA_Int64Tensor{};
using cuda_float_t = CUDA_FloatTensor;
using cuda_int64_t = CUDA_Int64Tensor;
#endif

namespace tt {

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
        if ( impl_.index() == CPU_FLOAT ) {
            return (TensorType *) std::get<CPU_FLOAT>(impl_).get();
        }
        if ( impl_.index() == CPU_INT64 ) {
            return (TensorType *) std::get<CPU_INT64>(impl_).get();
        }
#endif

#ifdef _USE_CUDA_GPU_
        if ( impl_.index() == CUDA_FLOAT ) {
            return (TensorType *) std::get<CUDA_FLOAT>(impl_).get();
        }
        if ( impl_.index() == CUDA_INT64 ) {
            return (TensorType *) std::get<CUDA_INT64>(impl_).get();
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
        CPU_FLOAT,
        CPU_INT64,
        CUDA_FLOAT,
        CUDA_INT64
    };
    using TensorImpl = std::variant< void *,
                                     std::unique_ptr<cpu_float_t>,
                                     std::unique_ptr<cpu_int64_t>,
                                     std::unique_ptr<cuda_float_t>,
                                     std::unique_ptr<cuda_int64_t> >;
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

// reset to a normal tensor
void DeviceTensor::reset(TensorDataType dtype, std::vector<size_t>& shape) {
    yannx_assert(dtype == YNX_UNDEFINED, "Can't reset a typed tensor!");
    if ( device_ == DEVICE_CPU ) {
        if ( dtype == YNX_FLOAT ) {
            impl_ = std::make_unique<yannx::dnnl::CPUTensor<YNX_FLOAT>>(shape);
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
