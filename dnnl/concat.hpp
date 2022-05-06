template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_Concat(std::vector<tt::tensor_t>& all_tensors, tt::tensor_t dst, int64_t axis) {
    tt::TensorDataType dtype = _DTYPE_;

    if (dtype == tt::YNX_FLOAT) {
        auto dst_mem = dnnl(dst)->dnnl_mem();
        auto dst_md = dnnl(dst)->dnnl_md();

        std::vector<dnnl_memory_desc_t> src_mds(all_tensors.size());
        for(size_t i = 0; i < all_tensors.size(); i++) {
            src_mds[i] = *(dnnl(all_tensors[i])->dnnl_md());
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
            auto mem = dnnl(all_tensors[i])->dnnl_mem();
            dnnl_help::set_arg(&args[0], i, DNNL_ARG_MULTIPLE_SRC + i, mem);
        }
        dnnl_help::set_arg(&args[0], all_tensors.size(), DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(concat, dnnl_help::DNNL_STREAM_DEFAULT, all_tensors.size() + 1, &args[0]));

        // destory prim and pd
        dnnl_help::release_prim(concat, concat_pd);
        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}
