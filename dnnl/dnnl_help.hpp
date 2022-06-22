#ifndef _ONEDNN_DNNL_HELP_HPP_
#define _ONEDNN_DNNL_HELP_HPP_

#include <tuple>
#include <vector>
#include <dnnl.h>
#include <dnnl_debug.h>

#include <yannx.hpp>

#define COMPLAIN_DNNL_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns DNNL error: %s.\n", __FILE__, __LINE__, \
                what, dnnl_status2str(status)); \
        yannx_panic("DNNL API failed.\n"); \
    } while (0)

#define DNNL_CHECK(f) \
    do { \
        dnnl_status_t s_ = f; \
        if (s_ != dnnl_success) COMPLAIN_DNNL_ERROR_AND_EXIT(#f, s_); \
    } while (0)


namespace yannx { namespace dnnl { namespace dnnl_help {

    dnnl_engine_t DNNL_ENGINE_DEFAULT = nullptr;
    dnnl_stream_t DNNL_STREAM_DEFAULT = nullptr;

    inline void dnnl_begin(uint64_t seed = 0) {
        if ( DNNL_ENGINE_DEFAULT == nullptr) {
            DNNL_CHECK(dnnl_engine_create(&dnnl_help::DNNL_ENGINE_DEFAULT, dnnl_cpu, 0));
            DNNL_CHECK(dnnl_stream_create(&dnnl_help::DNNL_STREAM_DEFAULT, dnnl_help::DNNL_ENGINE_DEFAULT, 1));
        }
    }

    inline void dnnl_end() {
        if ( DNNL_ENGINE_DEFAULT != nullptr) {
            DNNL_CHECK(dnnl_stream_destroy(dnnl_help::DNNL_STREAM_DEFAULT));
            DNNL_CHECK(dnnl_engine_destroy(dnnl_help::DNNL_ENGINE_DEFAULT));
        }
        DNNL_ENGINE_DEFAULT = nullptr;
        DNNL_STREAM_DEFAULT = nullptr;
    }

    inline dnnl_data_type_t tt_type_to_dnnl_type(yannx::tt::TensorDataType dtype) {
        if ( dtype == yannx::tt::YNX_FLOAT) {
            return dnnl_f32;
        }
        yannx_panic("DNNL unsupport data type.");
        return dnnl_data_type_undef;
    }

    inline dnnl_format_tag_t ndim_to_mem_plain_tag(size_t ndim) {
        if (ndim == 1) {
            return dnnl_a;
        }
        if (ndim == 2) {
            return dnnl_ab;
        }
        if (ndim == 3) {
            return dnnl_abc;
        }
        if (ndim == 4) {
            return dnnl_abcd;
        }
        if (ndim == 5) {
            return dnnl_abcde;
        }
        if (ndim == 6) {
            return dnnl_abcdef;
        }
        return dnnl_format_tag_undef;
    }

    inline void set_arg(dnnl_exec_arg_t* args, int pos, int arg, dnnl_memory_t mem) {
        args[pos].arg = arg;
        args[pos].memory = mem;
    }

    inline void release_prim(dnnl_primitive_t prim, dnnl_primitive_desc_t pd) {
        DNNL_CHECK(dnnl_primitive_destroy(prim));
        if ( pd != nullptr ) {
            DNNL_CHECK(dnnl_primitive_desc_destroy(pd));
        }
    }

    inline std::vector<uint64_t> query_shape_from_md(const dnnl_memory_desc_t *md) {
        std::vector<uint64_t> shape;
        for (int i = 0; i < md->ndims; i++) {
            shape.push_back(  md->dims[i] );
        }
        return shape;
    }

}}}

#endif
