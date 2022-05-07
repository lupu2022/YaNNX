template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_MatMul(tt::tensor_t A, tt::tensor_t B, tt::tensor_t Y) {
    tt::TensorDataType dtype = _DTYPE_;

    if (dtype == tt::YNX_FLOAT) {
        auto src0_md = A->device_float()->dnnl_md();
        auto src0_mem = A->device_float()->dnnl_mem();

        auto src1_md = B->device_float()->dnnl_md();
        auto src1_mem = B->device_float()->dnnl_mem();
        auto dst_md = Y->device_float()->dnnl_md();

        dnnl_matmul_desc_t desc;
        DNNL_CHECK(dnnl_matmul_desc_init(&desc, src0_md, src1_md, nullptr, dst_md));

        dnnl_primitive_desc_t matmul_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&matmul_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t matmul;
        DNNL_CHECK(dnnl_primitive_create(&matmul, matmul_pd));

        // create destnation memory
        {
            auto dst_md = dnnl_primitive_desc_query_md(matmul_pd, dnnl_query_dst_md, 0);
            Y->device_float()->reorder(dst_md);
        }
        auto dst_mem = Y->device_float()->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[3];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC_0, src0_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_SRC_1, src1_mem);
        dnnl_help::set_arg(args, 2, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(matmul, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(matmul, matmul_pd);

        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}
