template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_MatMul(tt::tensor_t A, tt::tensor_t B, tt::tensor_t Y) {
    tt::TensorDataType dtype = _DTYPE_;

    if (dtype == tt::YNX_FLOAT) {
        auto src0_md = dnnl(A)->dnnl_md();
        auto src0_mem = dnnl(A)->dnnl_mem();

        auto src1_md = dnnl(B)->dnnl_md();
        auto src1_mem = dnnl(B)->dnnl_mem();
        auto dst_md = dnnl(Y)->dnnl_md();

        dnnl_matmul_desc_t desc;
        DNNL_CHECK(dnnl_matmul_desc_init(&desc, src0_md, src1_md, nullptr, dst_md));

        dnnl_primitive_desc_t matmul_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&matmul_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t matmul;
        DNNL_CHECK(dnnl_primitive_create(&matmul, matmul_pd));

        // create destnation memory
        {
            auto dst_md = dnnl_primitive_desc_query_md(matmul_pd, dnnl_query_dst_md, 0);
            dnnl(Y)->reorder(dst_md);
        }
        auto dst_mem = dnnl(Y)->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[3];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src0_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_WEIGHTS, src1_mem);
        dnnl_help::set_arg(args, 2, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(matmul, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(matmul, matmul_pd);

        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}

template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_Gemm(tt::tensor_t A, tt::tensor_t B, std::variant<void *, tt::tensor_t>& C, tensor_t Y, float alpha, float beta, int64_t transA, int64_t transB) {
    tt::TensorDataType dtype = _DTYPE_;

    yannx_assert( alpha == 1.0, "DNNL's gemm only support alpha = 1.0");
    yannx_assert( beta == 1.0, "DNNL's gemm only support beta = 1.0");
    yannx_assert( transA == 0, "DNNL's gemm only support transA = 0 ");
    yannx_assert( transB == 1, "DNNL's gemm only support transB = 1 ");

    if (dtype == tt::YNX_FLOAT) {
        auto src_md = dnnl(A)->dnnl_md();
        auto src_mem = dnnl(A)->dnnl_mem();

        auto weight_md = dnnl(B)->dnnl_md();
        auto weight_mem = dnnl(B)->dnnl_mem();
        auto dst_md = dnnl(Y)->dnnl_md();

        dnnl_memory_desc_t* bias_md = nullptr;
        dnnl_memory_t bias_mem = nullptr;
        if ( C.index() == 1 ) {
            auto c = std::get<1>(C);
            bias_md = dnnl(c)->dnnl_md();
            bias_mem = dnnl(c)->dnnl_mem();
        }

        // create prim and pd
        dnnl_inner_product_desc_t desc;
        DNNL_CHECK(dnnl_inner_product_forward_desc_init(&desc,
                                                        dnnl_forward_inference,
                                                        src_md, weight_md, bias_md, dst_md));

        dnnl_primitive_desc_t ip_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&ip_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t ip;
        DNNL_CHECK(dnnl_primitive_create(&ip, ip_pd));

        // create destnation memory
        dnnl(Y)->reorder( dnnl_primitive_desc_query_md(ip_pd, dnnl_query_dst_md, 0) );
        auto dst_mem = dnnl(Y)->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[4];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);
        dnnl_help::set_arg(args, 2, DNNL_ARG_WEIGHTS_0, weight_mem);
        dnnl_help::set_arg(args, 3, DNNL_ARG_BIAS, bias_mem);

        DNNL_CHECK(dnnl_primitive_execute(ip, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(ip, ip_pd);
        return tt::YNX_OK;
    }

    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}
