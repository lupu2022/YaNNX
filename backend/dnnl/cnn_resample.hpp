
template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::resample_linear(tensor_t& dst) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.resample_linear: type error"));
    }

    float factors[2];
    factors[0] = 1.0 * dst->shape()[2] / (1.0 * shape()[2]);
    factors[1] = 1.0 * dst->shape()[3] / (1.0 * shape()[3]);

    // query memory and memory desc
    auto src_mem = dnnl_mem();
    auto src_md = dnnl_md();
    auto dst_md = dst->cpu_float()->dnnl_md();

    // create prim and pd
    dnnl_resampling_desc_t desc;
    DNNL_CHECK(dnnl_resampling_forward_desc_init(&desc,
                                                 dnnl_forward_training,
                                                 dnnl_resampling_linear,
                                                 factors,
                                                 src_md, dst_md));

    dnnl_primitive_desc_t resample_pd;
    DNNL_CHECK(dnnl_primitive_desc_create(&resample_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
    dnnl_primitive_t resample;
    DNNL_CHECK(dnnl_primitive_create(&resample, resample_pd));

    // create destnation memory
    dst->cpu_float()->reorder(dnnl_primitive_desc_query_md(resample_pd, dnnl_query_dst_md, 0));
    auto dst_mem = dst->cpu_float()->dnnl_mem();

    // prepare arguments and execute
    dnnl_exec_arg_t args[2];
    dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
    dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

    DNNL_CHECK(dnnl_primitive_execute(resample, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

    // destory prim and pd
    dnnl_help::release_prim(resample, resample_pd);

    return Ok();
}

