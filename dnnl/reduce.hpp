template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::dnnl_reduce_operator(std::vector<tt::tensor_t>& inputs, tt::tensor_t dst, dnnl_alg_kind_t algo, float p, float eps) {
    tt::TensorDataType dtype = _DTYPE_;

    if (dtype == tt::YNX_FLOAT) {
        auto dst_mem = dnnl(dst)->dnnl_mem();
        auto dst_md = dnnl(dst)->dnnl_md();

        dnnl_reduction_desc_t desc;
        DNNL_CHECK( dnnl_reduction_desc_init(&desc,
                                         algo,
                                         dnnl(inputs[0])->dnnl_md(),
                                         dst_md,
                                         0.0, 0.0));

        dnnl_primitive_desc_t reduce_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&reduce_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t reduce;
        DNNL_CHECK(dnnl_primitive_create(&reduce, reduce_pd));

        // prepare arguments and execute
        std::vector<dnnl_exec_arg_t> args(inputs.size() + 1);
        for(size_t i = 0; i < inputs.size(); i++) {
            auto mem = dnnl(inputs[i])->dnnl_mem();
            dnnl_help::set_arg(&args[0], i, DNNL_ARG_MULTIPLE_SRC + i, mem);
        }
        dnnl_help::set_arg(&args[0], inputs.size(), DNNL_ARG_DST, dst_mem);

        // destory prim and pd
        dnnl_help::release_prim(reduce, reduce_pd);
        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}
