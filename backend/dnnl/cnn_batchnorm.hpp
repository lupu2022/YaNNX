dnnl_memory_t combo_scale_shift(cpu_float_t* scale, cpu_float_t* shift) {
    uint64_t combo_shape[2];
    combo_shape[0] = 2;
    combo_shape[1] = scale->shape()[0];

    dnnl_memory_desc_t md;
    dnnl_memory_t mem;
    DNNL_CHECK(dnnl_memory_desc_init_by_tag(&md,
                                        2,
                                        (const int64_t *)combo_shape,
                                        scale->dnnl_dtype(),
                                        dnnl_help::ndim_to_mem_plain_tag(2)));
    
    DNNL_CHECK(dnnl_memory_create(&mem, &md, dnnl_help::DNNL_ENGINE_DEFAULT, DNNL_MEMORY_ALLOCATE));

    float* data_ptr = nullptr;
    DNNL_CHECK(dnnl_memory_get_data_handle(mem, (void **)&data_ptr));
    memcpy(data_ptr, scale->plain_ptr(), sizeof(float) * combo_shape[1]);
    memcpy(data_ptr + combo_shape[1], shift->plain_ptr(), sizeof(float) * combo_shape[1]);

    return mem;
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::batch_norm2d(tensor_t& scale, tensor_t& shift, tensor_t& mean, tensor_t& var, 
                                                                   float epsilon, tensor_t& dst) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.batch_norm2d: type error"));
    }
    
    auto src_mem = dnnl_mem();
    auto src_md = dnnl_md();
    auto combo_mem = combo_scale_shift(scale->cpu_float(), shift->cpu_float());
    auto mean_mem = mean->cpu_float()->dnnl_mem();
    auto var_mem = var->cpu_float()->dnnl_mem();

    dnnl_batch_normalization_desc_t desc;
    DNNL_CHECK(dnnl_batch_normalization_forward_desc_init(&desc, dnnl_forward_inference, src_md, epsilon, 
                                                          dnnl_use_scaleshift | dnnl_use_global_stats  ));

    dnnl_primitive_desc_t bn_pd;
    DNNL_CHECK(dnnl_primitive_desc_create(&bn_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
    dnnl_primitive_t bn;
    DNNL_CHECK(dnnl_primitive_create(&bn, bn_pd));

    // create destnation memory
    dst->cpu_float()->reorder(dnnl_primitive_desc_query_md(bn_pd, dnnl_query_dst_md, 0));
    auto dst_mem = dst->cpu_float()->dnnl_mem();
    
    // prepare arguments and execute
    dnnl_exec_arg_t args[5];
    dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
    dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);
    dnnl_help::set_arg(args, 2, DNNL_ARG_WEIGHTS_0, combo_mem);
    dnnl_help::set_arg(args, 3, DNNL_ARG_MEAN, mean_mem);
    dnnl_help::set_arg(args, 4, DNNL_ARG_VARIANCE , var_mem);
    
    DNNL_CHECK(dnnl_primitive_execute(bn, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

    // destory prim and pd
    dnnl_help::release_prim(bn, bn_pd);
    DNNL_CHECK(dnnl_memory_destroy(combo_mem));
    
    return Ok();
}

