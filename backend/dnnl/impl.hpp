#ifndef _DNNL_IMPL_HPP_
#define _DNNL_IMPL_HPP_

#include <iostream>
#include <cstring>
#include <cmath>

#include <yannx.hpp>
#include <tensortype.hpp>

#include "dnnl_help.hpp"

namespace yannx { namespace dnnl {

template <tt::TensorDataType _DTYPE_>
struct DNNLTensor : public tt::TensorType  {
    ~DNNLTensor() {
        release();
    }

    // buila a dense plain layout tensor
    DNNLTensor(const std::vector<size_t>& shape) : shape_(shape) {
        tt::TensorDataType tt_dtype = _DTYPE_;

        yannx_assert(shape_.size() != 0, "Can't build tensor with zero shape!");
        yannx_assert(tt_dtype != tt::YNX_UNDEFINED, "Can't implement a undefined CPU tensor");

        dtype_ = dnnl_help::tt_type_to_dnnl_type(tt_dtype);

        pd_ = nullptr;
        DNNL_CHECK(dnnl_memory_desc_init_by_tag(&plain_md_,
                                                shape_.size(),
                                                (const int64_t *)shape_.data(),
                                                dtype_,
                                                dnnl_help::ndim_to_mem_plain_tag(shape_.size())));
        DNNL_CHECK(dnnl_memory_create(&mem_, &plain_md_, dnnl_help::DNNL_ENGINE_DEFAULT, DNNL_MEMORY_ALLOCATE));
    }

    // buila plain tensor with filled value
    DNNLTensor(const std::vector<size_t>& shape, const void* pdata) : DNNLTensor(shape) {
        tt::TensorDataType tt_dtype = _DTYPE_;

        if ( tt_dtype == tt::YNX_FLOAT ) {
            float* dst = (float *)plain_ptr();
            const float* src = (const float *)pdata;
            memcpy(dst, src, num_items() * sizeof(float) );
            return;
        }

        yannx_panic("Can't build tensor with un-supported type!");
    }

    // buila arbitrary layout tensor with memory
    DNNLTensor(const std::vector<size_t>& shape, dnnl_memory_t mem, dnnl_primitive_desc_t pd) : shape_(shape) {
        yannx_assert(shape_.size() != 0, "Can't build tensor with zero shape!");

        tt::TensorDataType tt_dtype = _DTYPE_;
        dtype_ = dnnl_help::tt_type_to_dnnl_type(tt_dtype);

        pd_ = pd;
        mem_ = mem;
        DNNL_CHECK(dnnl_memory_desc_init_by_tag(&plain_md_,
                                                shape_.size(),
                                                (const int64_t *)shape_.data(),
                                                dtype_,
                                                dnnl_help::ndim_to_mem_plain_tag(shape_.size())));
    }

    // only one real overrided function
    const void* value() override {
        return plain_ptr();
    }

    // we don't need call these interface , it is via DeviceTensor
    void reset(tt::TensorDataType dtype, std::vector<size_t>& shape) override {}
    void reset(tt::TensorDataType dtype, std::vector<size_t>& shape, const void* pdata) override {}
    void reset(tt::TensorDataType dtype, const void* pvalue) override {}
    tt::TensorDataType dtype() override { return _DTYPE_; }
    const std::vector<size_t>& shape() override { return shape_; }

    // real tensor computing API
public:
    // two item binary operator : A op B = C
    tt::OperatorReturnType onnx_Add(tt::tensor_t A, tt::tensor_t B, tt::tensor_t C) override {
        return dnnl_binary_operator(A, B, C, dnnl_binary_add);
    }
    tt::OperatorReturnType onnx_Mul(tt::tensor_t A, tt::tensor_t B, tt::tensor_t C) override {
        return dnnl_binary_operator(A, B, C, dnnl_binary_mul);
    }
    tt::OperatorReturnType onnx_Div(tt::tensor_t A, tt::tensor_t B, tt::tensor_t C) override {
        return dnnl_binary_operator(A, B, C, dnnl_binary_div);
    }
    tt::OperatorReturnType onnx_Sub(tt::tensor_t A, tt::tensor_t B, tt::tensor_t C) override {
        return dnnl_binary_operator(A, B, C, dnnl_binary_sub);
    }

    // element wise operator : Y = op(X)
    tt::OperatorReturnType onnx_Abs(tt::tensor_t X, tt::tensor_t Y) override {
        return dnnl_eltwise_operator(X, Y, dnnl_eltwise_abs, 0.0, 0.0);
    }
    tt::OperatorReturnType onnx_HardSwish(tt::tensor_t X, tt::tensor_t Y) override {
        return dnnl_eltwise_operator(X, Y, dnnl_eltwise_hardswish, 0.0, 0.0);
    }
    tt::OperatorReturnType onnx_Relu(tt::tensor_t X, tt::tensor_t Y) override {
        return dnnl_eltwise_operator(X, Y, dnnl_eltwise_relu, 0.0, 0.0);
    }
    tt::OperatorReturnType onnx_Sigmoid(tt::tensor_t X, tt::tensor_t Y) override {
        return dnnl_eltwise_operator(X, Y, dnnl_eltwise_logistic, 0.0, 0.0);
    }
    tt::OperatorReturnType onnx_Sqrt(tt::tensor_t X, tt::tensor_t Y) override {
        return dnnl_eltwise_operator(X, Y, dnnl_eltwise_sqrt, 0.0, 0.0);
    }
    tt::OperatorReturnType onnx_Tanh(tt::tensor_t X, tt::tensor_t Y) override {
        return dnnl_eltwise_operator(X, Y, dnnl_eltwise_tanh, 0.0, 0.0);
    }


private:
    // help functions for computing API
    inline DNNLTensor<_DTYPE_>* dnnl(tt::tensor_t& t) {
        auto* p = (DNNLTensor<_DTYPE_> *)t.get();
        return p;
    }
    tt::OperatorReturnType dnnl_binary_operator(tt::tensor_t A, tt::tensor_t B, tt::tensor_t C, dnnl_alg_kind_t algo);
    tt::OperatorReturnType dnnl_eltwise_operator(tt::tensor_t X, tt::tensor_t Y, dnnl_alg_kind_t algo, float alpha, float beta);


private:
    // fast access
    dnnl_data_type_t dnnl_dtype() {
        return dtype_;
    }
    dnnl_memory_t dnnl_mem() {
        return mem_;
    }
    dnnl_memory_desc_t* dnnl_md() {
        dnnl_memory_desc_t *md;
        DNNL_CHECK(dnnl_memory_get_memory_desc(mem_, (const dnnl_memory_desc_t**)&md));
        return md;
    }
    const_dnnl_primitive_desc_t dnnl_primd() {
        return pd_;
    }
    void set_primd(dnnl_primitive_desc_t pd) {
        if ( pd_ != nullptr) {
            DNNL_CHECK(dnnl_primitive_desc_destroy(pd_));
        }
        pd_ = pd;
    }

    // some help functions
    size_t num_items() {
        size_t n = 1;
        for (size_t i = 0; i < shape_.size(); i++) {
            n = n * shape_[i];
        }
        return n;
    }

    // layout management
    void reorder(const dnnl_memory_desc_t* target_md) {
        dnnl_memory_desc_t *src_md = dnnl_md();
        int ret = dnnl_memory_desc_equal(src_md, target_md);
        if (ret == 1) {
            return;
        }
        do_reorder(target_md, target_md);
    }
    void reorder() {
        dnnl_memory_desc_t *src_md = dnnl_md();
        int ret = dnnl_memory_desc_equal(&plain_md_, src_md);
        if (ret == 1) {
            return;
        }
        do_reorder(src_md, &plain_md_);
    }

    void* plain_ptr() {
        reorder();
        void* data_ptr = nullptr;
        DNNL_CHECK(dnnl_memory_get_data_handle(mem_, &data_ptr));
        return data_ptr;
    }

    void release() {
        if ( mem_ != nullptr) {
            DNNL_CHECK(dnnl_memory_destroy(mem_));
        }
        mem_ = nullptr;
        if ( pd_ != nullptr) {
            DNNL_CHECK(dnnl_primitive_desc_destroy(pd_));
        }
        pd_ = nullptr;
    }

    void do_reorder(const dnnl_memory_desc_t* src_md, const dnnl_memory_desc_t* dst_md) {
        dnnl_memory_t dst_mem;
        DNNL_CHECK(dnnl_memory_create(&dst_mem, dst_md, dnnl_help::DNNL_ENGINE_DEFAULT, DNNL_MEMORY_ALLOCATE));

        dnnl_primitive_t      reorder_prim;
        dnnl_primitive_desc_t reorder_pd;
        DNNL_CHECK(dnnl_reorder_primitive_desc_create(&reorder_pd,
                                                      src_md, dnnl_help::DNNL_ENGINE_DEFAULT,
                                                      dst_md, dnnl_help::DNNL_ENGINE_DEFAULT,
                                                      nullptr));
        DNNL_CHECK(dnnl_primitive_create(&reorder_prim, reorder_pd));

        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, mem_);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(reorder_prim, dnnl_help::DNNL_STREAM_DEFAULT, 2, args));

        dnnl_help::release_prim(reorder_prim, reorder_pd);

        release();
        mem_ = dst_mem;
    }

private:
    dnnl_memory_desc_t              plain_md_;
    dnnl_memory_t                   mem_;
    dnnl_primitive_desc_t           pd_;
    dnnl_data_type_t                dtype_;

    const std::vector<size_t>     shape_;
};

#include "binary.hpp"
#include "eltwise.hpp"

}}

#endif
