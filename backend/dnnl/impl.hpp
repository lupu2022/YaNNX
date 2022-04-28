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
struct CPUTensor : public tt::TensorType  {
    ~CPUTensor() {
        release();
    }

    CPUTensor() {
        tt::TensorDataType tt_dtype = _DTYPE_;
    }

    // buila a dense plain layout tensor
    CPUTensor(const std::vector<size_t>& shape) : shape_(shape) {
        tt::TensorDataType tt_dtype = _DTYPE_;

        yannx_assert(shape_.size() != 0, "Can't build tensor with zero shape!");
        yannx_assert(tt_dtype != tt::YNX_UNDEFINED);

        dtype_ = dnnl_help::tt_type_to_dnnl_type(tt_dtype);

        pd_ = nullptr;
        DNNL_CHECK(dnnl_memory_desc_init_by_tag(&plain_md_,
                                                shape_.size(),
                                                (const int64_t *)shape_.data(),
                                                dtype_,
                                                dnnl_help::ndim_to_mem_plain_tag(shape_.size())));
        DNNL_CHECK(dnnl_memory_create(&mem_, &plain_md_, dnnl_help::DNNL_ENGINE_DEFAULT, DNNL_MEMORY_ALLOCATE));
    }

    // buila arbitrary layout tensor with memory
    CPUTensor(const std::vector<size_t>& shape, dnnl_memory_t mem, dnnl_primitive_desc_t pd) : shape_(shape) {
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
    const std::vector<size_t>& shape() {
        return shape_;
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

}}

#endif
