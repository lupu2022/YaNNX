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

#ifndef _USE_DNNL_CPU_
#ifndef _USE_CUDA_GPU_
#include "tt_config.hpp"
#endif
#endif

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
using cuda_float_t = tt::cuda::CUDATensor<tt::TensorDataType::YNX_FLOAT>;
using cuda_int64_t = tt::cuda::CUDATensor<tt::TensorDataType::YNX_INT64>;
#else
struct CUDA_FloatTensor{};
struct CUDA_Int64Tensor{};
using cuda_float_t = CUDA_FloatTensor;
using cuda_int64_t = CUDA_Int64Tensor;
#endif

namespace tt {

struct DeviceTensor: public TensorType  {
public:
    // init functions
    TensorType() : shape_(), dtype_(tt::TensorDataType::YNX_UNDEFINED), impl_((void*)NULL) {};

    const std::vector<size_t>& shape() override {
        return shape_;
    }
    tt::TensorDataType& dtype() override {
        return dtype_;
    }

private:
    // basic info about tensor
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
    using TensorImpl = mpark::variant<  void *,
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

#endif
