template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::layernorm(tensor_t& scale, tensor_t& shift, tensor_t& running_mean, tensor_t& running_var, float epsilon, tensor_t& dst) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.layernorm: type error"));
    }

    auto src_mem = dnnl_mem();
    auto src_md = dnnl_md();
    auto mean_mem = running_mean->cpu_float()->dnnl_mem();
    auto var_mem = running_var->cpu_float()->dnnl_mem();
    auto stat_md = running_mean->cpu_float()->dnnl_md();

    dnnl_layer_normalization_desc_t desc;
    DNNL_CHECK(dnnl_layer_normalization_forward_desc_init(&desc, dnnl_forward_training, src_md, stat_md, epsilon,
                                                          dnnl_use_scaleshift  ));

    dnnl_primitive_desc_t ln_pd;
    DNNL_CHECK(dnnl_primitive_desc_create(&ln_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
    dnnl_primitive_t ln;
    DNNL_CHECK(dnnl_primitive_create(&ln, ln_pd));

    // create destnation memory
    dst->cpu_float()->reorder(dnnl_primitive_desc_query_md(ln_pd, dnnl_query_dst_md, 0));
    auto dst_mem = dst->cpu_float()->dnnl_mem();

    auto combo_mem = combo_scale_shift(scale->cpu_float(), shift->cpu_float());

    // prepare arguments and execute
    dnnl_exec_arg_t args[5];
    dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
    dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);
    dnnl_help::set_arg(args, 2, DNNL_ARG_MEAN, mean_mem);
    dnnl_help::set_arg(args, 3, DNNL_ARG_VARIANCE, var_mem);
    dnnl_help::set_arg(args, 4, DNNL_ARG_WEIGHTS_0, combo_mem);

    DNNL_CHECK(dnnl_primitive_execute(ln, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

    // destory prim and pd
    dnnl_help::release_prim(ln, ln_pd);
    DNNL_CHECK(dnnl_memory_destroy(combo_mem));

    return Ok();
}




