
template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::softmax(tensor_t& dst) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.softmax: type error"));
    }
    
    // query memory and memory desc
    auto src_mem = dnnl_mem();
    auto src_md = dnnl_md();

    // create prim and pd
    dnnl_softmax_desc_t desc;
    DNNL_CHECK(dnnl_softmax_forward_desc_init(&desc,
                                              dnnl_forward_training,
                                              src_md, 1));      // axis = 1;
    dnnl_primitive_desc_t softmax_pd;
    DNNL_CHECK(dnnl_primitive_desc_create(&softmax_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
    dnnl_primitive_t softmax;
    DNNL_CHECK(dnnl_primitive_create(&softmax, softmax_pd));

    // create destnation memory
    auto dst_md = dnnl_primitive_desc_query_md(softmax_pd, dnnl_query_dst_md, 0);
    dst->cpu_float()->reorder(dst_md);
    auto dst_mem = dst->cpu_float()->dnnl_mem();

    // prepare arguments and execute
    dnnl_exec_arg_t args[2];
    dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
    dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

    DNNL_CHECK(dnnl_primitive_execute(softmax, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

    // destory prim and pd
    dnnl_help::release_prim(softmax, softmax_pd);

    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::logsoftmax(tensor_t& dst) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.logsoftmax: type error"));
    }

    // query memory and memory desc
    auto src_mem = dnnl_mem();
    auto src_md = dnnl_md();

    // create prim and pd
    dnnl_logsoftmax_desc_t desc;
    DNNL_CHECK(dnnl_logsoftmax_forward_desc_init(&desc,
                                              dnnl_forward_training,
                                              src_md, 1));      // axis = 1;
    dnnl_primitive_desc_t softmax_pd;
    DNNL_CHECK(dnnl_primitive_desc_create(&softmax_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
    dnnl_primitive_t softmax;
    DNNL_CHECK(dnnl_primitive_create(&softmax, softmax_pd));

    // create destnation memory
    auto dst_md = dnnl_primitive_desc_query_md(softmax_pd, dnnl_query_dst_md, 0);
    dst->cpu_float()->reorder(dst_md);
    auto dst_mem = dst->cpu_float()->dnnl_mem();

    // prepare arguments and execute
    dnnl_exec_arg_t args[2];
    dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
    dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

    DNNL_CHECK(dnnl_primitive_execute(softmax, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

    // destory prim and pd
    dnnl_help::release_prim(softmax, softmax_pd);

    return Ok();
}