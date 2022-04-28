
template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::add(tensor_t& b, tensor_t& c) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        auto src0_md = dnnl_md();
        auto src0_mem = dnnl_mem();
        auto src1_md = b->cpu_float()->dnnl_md();
        auto src1_mem = b->cpu_float()->dnnl_mem();
        auto dst_md = c->cpu_float()->dnnl_md();

        dnnl_binary_desc_t desc;
        DNNL_CHECK(dnnl_binary_desc_init(&desc,
                                         dnnl_binary_add,
                                         src0_md, src1_md, dst_md));

        dnnl_primitive_desc_t binary_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&binary_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t binary;
        DNNL_CHECK(dnnl_primitive_create(&binary, binary_pd));

        // create destnation memory
        {
            auto dst_md = dnnl_primitive_desc_query_md(binary_pd, dnnl_query_dst_md, 0);
            c->cpu_float()->reorder(dst_md);
        }
        auto dst_mem = c->cpu_float()->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[3];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC_0, src0_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_SRC_1, src1_mem);
        dnnl_help::set_arg(args, 2, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(binary, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(binary, binary_pd);
    } else {
        return Err(TensorError("CPUTensor.add: type error"));
    }

    return Ok();
}


template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::mul(tensor_t& b, tensor_t& c) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        auto src0_md = dnnl_md();
        auto src0_mem = dnnl_mem();
        auto src1_md = b->cpu_float()->dnnl_md();
        auto src1_mem = b->cpu_float()->dnnl_mem();
        auto dst_md = c->cpu_float()->dnnl_md();

        dnnl_binary_desc_t desc;
        DNNL_CHECK(dnnl_binary_desc_init(&desc,
                                         dnnl_binary_mul,
                                         src0_md, src1_md, dst_md));

        dnnl_primitive_desc_t binary_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&binary_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t binary;
        DNNL_CHECK(dnnl_primitive_create(&binary, binary_pd));

        // create destnation memory
        {
            auto dst_md = dnnl_primitive_desc_query_md(binary_pd, dnnl_query_dst_md, 0);
            c->cpu_float()->reorder(dst_md);
        }
        auto dst_mem = c->cpu_float()->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[3];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC_0, src0_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_SRC_1, src1_mem);
        dnnl_help::set_arg(args, 2, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(binary, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(binary, binary_pd);
    } else {
        return Err(TensorError("CPUTensor.mul: type error"));
    }

    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::mm(tensor_t& b, tensor_t& c) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        float* ptr_a = (float *)plain_ptr();
        float* ptr_b = (float *)b->cpu_float()->plain_ptr();
        float* ptr_c = (float *)c->cpu_float()->plain_ptr();

        int m = shape()[0];
        int n = b->cpu_float()->shape()[1];
        int k = shape()[1];
        dnnl_sgemm('N', 'N',
                   m, n, k, 1.0, ptr_a, k, ptr_b, n, 0.0, ptr_c, n);
    } else {
        return Err(TensorError("CPUTensor.mm: type error"));
    }

    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::scale(float s, tensor_t& dst) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        // query memory and memory desc
        auto src_mem = dnnl_mem();
        auto src_md = dnnl_md();

        // create prim and pd
        dnnl_eltwise_desc_t desc;
        DNNL_CHECK(dnnl_eltwise_forward_desc_init(&desc,
                                                dnnl_forward_training,
                                                dnnl_eltwise_linear,
                                                src_md, s, 0.0));
        dnnl_primitive_desc_t eltwise_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&eltwise_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t eltwise;
        DNNL_CHECK(dnnl_primitive_create(&eltwise, eltwise_pd));

        // create destnation memory
        auto dst_md = dnnl_primitive_desc_query_md(eltwise_pd, dnnl_query_dst_md, 0);
        dst->cpu_float()->reorder(dst_md);
        auto dst_mem = dst->cpu_float()->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(eltwise, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(eltwise, eltwise_pd);

    } else {
        return Err(TensorError("CPUTensor.scale: type error"));
    }

    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::inv(tensor_t& dst) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        // query memory and memory desc
        auto src_mem = dnnl_mem();
        auto src_md = dnnl_md();

        // create prim and pd
        dnnl_eltwise_desc_t desc;
        DNNL_CHECK(dnnl_eltwise_forward_desc_init(&desc,
                                                dnnl_forward_training,
                                                dnnl_eltwise_pow,
                                                src_md, 1.0, -1.0));
        dnnl_primitive_desc_t eltwise_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&eltwise_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t eltwise;
        DNNL_CHECK(dnnl_primitive_create(&eltwise, eltwise_pd));

        // create destnation memory
        auto dst_md = dnnl_primitive_desc_query_md(eltwise_pd, dnnl_query_dst_md, 0);
        dst->cpu_float()->reorder(dst_md);
        auto dst_mem = dst->cpu_float()->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(eltwise, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(eltwise, eltwise_pd);
    } else {
        return Err(TensorError("CPUTensor.inv: type error"));
    }
    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::sign(tensor_t& dst) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        float* ptr_a = (float *)plain_ptr();
        float* ptr_b = (float *)dst->cpu_float()->plain_ptr();
        auto total = items();

        #pragma omp parallel for
        for(size_t i = 0; i < total; i++) {
            if (ptr_a[i] >= 0.0) {
                ptr_b[i] = 1.0;
            } else {
                ptr_b[i] = -1.0;
            }
        }
    } else {
        return Err(TensorError("CPUTensor.sigh type error"));
    }
    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::abs(tensor_t& dst) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        // query memory and memory desc
        auto src_mem = dnnl_mem();
        auto src_md = dnnl_md();

        // create prim and pd
        dnnl_eltwise_desc_t desc;
        DNNL_CHECK(dnnl_eltwise_forward_desc_init(&desc,
                                                dnnl_forward_training,
                                                dnnl_eltwise_abs,
                                                src_md, 0.0, 0.0));
        dnnl_primitive_desc_t eltwise_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&eltwise_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t eltwise;
        DNNL_CHECK(dnnl_primitive_create(&eltwise, eltwise_pd));

        // create destnation memory
        auto dst_md = dnnl_primitive_desc_query_md(eltwise_pd, dnnl_query_dst_md, 0);
        dst->cpu_float()->reorder(dst_md);
        auto dst_mem = dst->cpu_float()->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(eltwise, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(eltwise, eltwise_pd);
    } else {
        return Err(TensorError("CPUTensor.abs type error"));
    }
    return Ok();
}


template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::ln(tensor_t& dst) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        // query memory and memory desc
        auto src_mem = dnnl_mem();
        auto src_md = dnnl_md();

        // create prim and pd
        dnnl_eltwise_desc_t desc;
        DNNL_CHECK(dnnl_eltwise_forward_desc_init(&desc,
                                                dnnl_forward_training,
                                                dnnl_eltwise_log,
                                                src_md, 0.0, 0.0));
        dnnl_primitive_desc_t eltwise_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&eltwise_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t eltwise;
        DNNL_CHECK(dnnl_primitive_create(&eltwise, eltwise_pd));

        // create destnation memory
        auto dst_md = dnnl_primitive_desc_query_md(eltwise_pd, dnnl_query_dst_md, 0);
        dst->cpu_float()->reorder(dst_md);
        auto dst_mem = dst->cpu_float()->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(eltwise, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(eltwise, eltwise_pd);
    } else {
        return Err(TensorError("CPUTensor.ln type error"));
    }
    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::clamp(float min, float max, tensor_t& dst) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        // query memory and memory desc
        auto src_mem = dnnl_mem();
        auto src_md = dnnl_md();

        // create prim and pd
        dnnl_eltwise_desc_t desc;
        DNNL_CHECK(dnnl_eltwise_forward_desc_init(&desc,
                                                dnnl_forward_training,
                                                dnnl_eltwise_clip,
                                                src_md, min, max));
        dnnl_primitive_desc_t eltwise_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&eltwise_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t eltwise;
        DNNL_CHECK(dnnl_primitive_create(&eltwise, eltwise_pd));

        // create destnation memory
        auto dst_md = dnnl_primitive_desc_query_md(eltwise_pd, dnnl_query_dst_md, 0);
        dst->cpu_float()->reorder(dst_md);
        auto dst_mem = dst->cpu_float()->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(eltwise, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(eltwise, eltwise_pd);

    } else {
        return Err(TensorError("CPUTensor.clamp type error"));
    }
    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::sigmoid(tensor_t& dst) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        // query memory and memory desc
        auto src_mem = dnnl_mem();
        auto src_md = dnnl_md();

        // create prim and pd
        dnnl_eltwise_desc_t desc;
        DNNL_CHECK(dnnl_eltwise_forward_desc_init(&desc,
                                                dnnl_forward_training,
                                                dnnl_eltwise_logistic,
                                                src_md, 0.0, 0.0));
        dnnl_primitive_desc_t eltwise_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&eltwise_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t eltwise;
        DNNL_CHECK(dnnl_primitive_create(&eltwise, eltwise_pd));

        // create destnation memory
        auto dst_md = dnnl_primitive_desc_query_md(eltwise_pd, dnnl_query_dst_md, 0);
        dst->cpu_float()->reorder(dst_md);
        auto dst_mem = dst->cpu_float()->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(eltwise, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(eltwise, eltwise_pd);
    } else {
        return Err(TensorError("CPUTensor.sigmoid type error"));
    }
    return Ok();
}


template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::tanh(tensor_t& dst) {
    DataType dtype = _DTYPE_;

    if (dtype == DataType::Float) {
        // query memory and memory desc
        auto src_mem = dnnl_mem();
        auto src_md = dnnl_md();

        // create prim and pd
        dnnl_eltwise_desc_t desc;
        DNNL_CHECK(dnnl_eltwise_forward_desc_init(&desc,
                                                dnnl_forward_training,
                                                dnnl_eltwise_tanh,
                                                src_md, 0.0, 0.0));
        dnnl_primitive_desc_t eltwise_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&eltwise_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t eltwise;
        DNNL_CHECK(dnnl_primitive_create(&eltwise, eltwise_pd));

        // create destnation memory
        auto dst_md = dnnl_primitive_desc_query_md(eltwise_pd, dnnl_query_dst_md, 0);
        dst->cpu_float()->reorder(dst_md);
        auto dst_mem = dst->cpu_float()->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(eltwise, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(eltwise, eltwise_pd);
    } else {
        return Err(TensorError("CPUTensor.tanh type error"));
    }
    return Ok();
}


