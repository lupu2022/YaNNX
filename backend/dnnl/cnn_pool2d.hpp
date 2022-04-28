template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::average_pool2d(const ShapeType& kernel_shape, const ShapeType& pads, const ShapeType& stride, const ShapeType& dilations,
                                                     std::string& auto_pad, int ceil_mode, int count_include_pad, tensor_t& dst) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.average_pool2d: type error"));
    }

    auto src_mem = dnnl_mem();
    auto src_md = dnnl_md();
    auto dst_md = dst->cpu_float()->dnnl_md();

    // create prim and pd
    dnnl_pooling_desc_t desc;
    DNNL_CHECK(dnnl_pooling_forward_desc_init(&desc,
                                              dnnl_forward_training,
                                              dnnl_pooling_avg,
                                              src_md, dst_md,
                                              (const int64_t *)stride.dims(), (const int64_t *)kernel_shape.dims(),
                                              (const int64_t *)&pads[0], (const int64_t *)&pads[2]));

    dnnl_primitive_desc_t pool_pd;
    DNNL_CHECK(dnnl_primitive_desc_create(&pool_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
    dnnl_primitive_t pool;
    DNNL_CHECK(dnnl_primitive_create(&pool, pool_pd));

    // create destnation memory
    dst->cpu_float()->reorder(dnnl_primitive_desc_query_md(pool_pd, dnnl_query_dst_md, 0));
    auto dst_mem = dst->cpu_float()->dnnl_mem();

    // prepare arguments and execute
    dnnl_exec_arg_t args[2];
    dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
    dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);
    
    DNNL_CHECK(dnnl_primitive_execute(pool, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

    // destory prim and pd
    dnnl_help::release_prim(pool, pool_pd);

    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::max_pool2d(const ShapeType& kernel_shape, const ShapeType& pads, const ShapeType& stride, const ShapeType& dilations,
                                                     std::string& auto_pad, int ceil_mode, tensor_t& dst) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.max_pool2d: type error"));
    }

    auto src_mem = dnnl_mem();
    auto src_md = dnnl_md();
    auto dst_md = dst->cpu_float()->dnnl_md();

    // create prim and pd
    dnnl_pooling_desc_t desc;
    DNNL_CHECK(dnnl_pooling_forward_desc_init(&desc,
                                              dnnl_forward_inference,
                                              dnnl_pooling_max,
                                              src_md, dst_md,
                                              (const int64_t *)stride.dims(), (const int64_t *)kernel_shape.dims(),
                                              (const int64_t *)&pads[0], (const int64_t *)&pads[2]));

    dnnl_primitive_desc_t pool_pd;
    DNNL_CHECK(dnnl_primitive_desc_create(&pool_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
    dnnl_primitive_t pool;
    DNNL_CHECK(dnnl_primitive_create(&pool, pool_pd));

    // create destnation memory
    dst->cpu_float()->reorder(dnnl_primitive_desc_query_md(pool_pd, dnnl_query_dst_md, 0));
    auto dst_mem = dst->cpu_float()->dnnl_mem();

    // prepare arguments and execute
    dnnl_exec_arg_t args[2];
    dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
    dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);
    
    DNNL_CHECK(dnnl_primitive_execute(pool, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

    // destory prim and pd
    dnnl_help::release_prim(pool, pool_pd);

    return Ok();
}

