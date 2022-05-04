template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::dnnl_binary_operator(tt::tensor_t A, tt::tensor_t B, tt::tensor_t C, dnnl_alg_kind_t algo) {
    tt::TensorDataType dtype = _DTYPE_;

    if (dtype == tt::YNX_FLOAT) {
        auto src0_md = dnnl(A)->dnnl_md();
        auto src0_mem = dnnl(A)->dnnl_mem();

        auto src1_md = dnnl(B)->dnnl_md();
        auto src1_mem = dnnl(B)->dnnl_mem();
        auto dst_md = dnnl(C)->dnnl_md();

        dnnl_binary_desc_t desc;
        DNNL_CHECK(dnnl_binary_desc_init(&desc,
                                         algo,
                                         src0_md, src1_md, dst_md));

        dnnl_primitive_desc_t binary_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&binary_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t binary;
        DNNL_CHECK(dnnl_primitive_create(&binary, binary_pd));

        // create destnation memory
        {
            auto dst_md = dnnl_primitive_desc_query_md(binary_pd, dnnl_query_dst_md, 0);
            dnnl(C)->reorder(dst_md);
        }
        auto dst_mem = dnnl(C)->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[3];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC_0, src0_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_SRC_1, src1_mem);
        dnnl_help::set_arg(args, 2, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(binary, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(binary, binary_pd);
        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}
