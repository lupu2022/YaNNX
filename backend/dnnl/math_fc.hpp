
template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::fc(tensor_t& w, option_tensor_t& b, tensor_t& dst) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.fc: type error"));
    }

    // query memory and memory desc
    auto src_mem = dnnl_mem();
    auto src_md = dnnl_md();
    auto weight_md = w->cpu_float()->dnnl_md();
    auto weight_mem = w->cpu_float()->dnnl_mem();
    auto dst_md = dst->cpu_float()->dnnl_md();
    dnnl_memory_desc_t* bias_md = nullptr;
    dnnl_memory_t bias_mem = nullptr;
    if ( b.isSome() ) {
        bias_md = b.unwrapSome()->cpu_float()->dnnl_md();
        bias_mem = b.unwrapSome()->cpu_float()->dnnl_mem();
    }

    // create prim and pd
    dnnl_inner_product_desc_t desc;
    DNNL_CHECK(dnnl_inner_product_forward_desc_init(&desc,
                                                    dnnl_forward_training,
                                                    src_md, weight_md, bias_md, dst_md));

    dnnl_primitive_desc_t ip_pd;
    DNNL_CHECK(dnnl_primitive_desc_create(&ip_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
    dnnl_primitive_t ip;
    DNNL_CHECK(dnnl_primitive_create(&ip, ip_pd));

    // create destnation memory
    dst->cpu_float()->reorder( dnnl_primitive_desc_query_md(ip_pd, dnnl_query_dst_md, 0) );
    auto dst_mem = dst->cpu_float()->dnnl_mem();

    // prepare arguments and execute
    dnnl_exec_arg_t args[4];
    dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
    dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);
    dnnl_help::set_arg(args, 2, DNNL_ARG_WEIGHTS_0, weight_mem);
    dnnl_help::set_arg(args, 3, DNNL_ARG_BIAS, bias_mem);

    DNNL_CHECK(dnnl_primitive_execute(ip, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

    // destory prim and pd
    dnnl_help::release_prim(ip, ip_pd);

    return Ok();
}

