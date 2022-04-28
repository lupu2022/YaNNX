
template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::concat(std::vector<tensor_t>& all_tensors, int axis, tensor_t& dst) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.concat: type error"));
    }

    auto dst_mem = dst->cpu_float()->dnnl_mem();
    auto dst_md = dst->cpu_float()->dnnl_md();

    std::vector<dnnl_memory_desc_t> src_mds(all_tensors.size());
    for(size_t i = 0; i < all_tensors.size(); i++) {
        src_mds[i] = *all_tensors[i]->cpu_float()->dnnl_md();
    }

    dnnl_primitive_desc_t concat_pd;
    DNNL_CHECK(dnnl_concat_primitive_desc_create(&concat_pd,
                                                 dst_md,
                                                 all_tensors.size(),
                                                 axis,
                                                 &src_mds[0],
                                                 nullptr,
                                                 dnnl_help::DNNL_ENGINE_DEFAULT));

    dnnl_primitive_t concat;
    DNNL_CHECK(dnnl_primitive_create(&concat, concat_pd));

    // prepare arguments and execute
    std::vector<dnnl_exec_arg_t> args(all_tensors.size() + 1);
    for(size_t i = 0; i < all_tensors.size(); i++) {
        auto mem = all_tensors[i]->cpu_float()->dnnl_mem();
        dnnl_help::set_arg(&args[0], i, DNNL_ARG_MULTIPLE_SRC + i, mem);
    }
    dnnl_help::set_arg(&args[0], all_tensors.size(), DNNL_ARG_DST, dst_mem);

    DNNL_CHECK(dnnl_primitive_execute(concat, dnnl_help::DNNL_STREAM_DEFAULT, all_tensors.size() + 1, &args[0]));

    // destory prim and pd
    dnnl_help::release_prim(concat, concat_pd);

    return Ok();
}

