template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_MaxPool(tt::tensor_t X, tt::tensor_t Y, std::variant<void *, tt::tensor_t>& Indices, std::string auto_pad, int64_t ceil_mode, std::vector<int64_t> dilations, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, int64_t storage_order, std::vector<int64_t> strides) {
    tt::TensorDataType dtype = _DTYPE_;

    if ( dtype == tt::YNX_FLOAT ) {
        auto src_mem = dnnl(X)->dnnl_mem();
        auto src_md = dnnl(X)->dnnl_md();
        auto dst_md = dnnl(Y)->dnnl_md();

        yannx_assert( Indices.index() == 0, "We don't support output Indices");

        // create prim and pd
        dnnl_pooling_desc_t desc;
        DNNL_CHECK(dnnl_pooling_forward_desc_init(&desc,
                                                dnnl_forward_inference,
                                                dnnl_pooling_max,
                                                src_md, dst_md,
                                                (const int64_t *)strides.data(), (const int64_t *)kernel_shape.data(),
                                                (const int64_t *)&pads[0], (const int64_t *)&pads[2]));

        dnnl_primitive_desc_t pool_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&pool_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t pool;
        DNNL_CHECK(dnnl_primitive_create(&pool, pool_pd));

        // create destnation memory
        dnnl(Y)->reorder(dnnl_primitive_desc_query_md(pool_pd, dnnl_query_dst_md, 0));
        auto dst_mem = dnnl(Y)->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(pool, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(pool, pool_pd);
        return tt::YNX_OK;
    }

    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}

template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_AveragePool(tt::tensor_t X, tensor_t Y, std::string auto_pad, int64_t ceil_mode, int64_t count_include_pad, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) {
    tt::TensorDataType dtype = _DTYPE_;

    if ( dtype == tt::YNX_FLOAT ) {
        auto src_mem = dnnl(X)->dnnl_mem();
        auto src_md = dnnl(X)->dnnl_md();
        auto dst_md = dnnl(Y)->dnnl_md();

        // create prim and pd
        dnnl_pooling_desc_t desc;
        DNNL_CHECK(dnnl_pooling_forward_desc_init(&desc,
                                                dnnl_forward_training,
                                                dnnl_pooling_avg,
                                                src_md, dst_md,
                                                (const int64_t *)strides.data(), (const int64_t *)kernel_shape.data(),
                                                (const int64_t *)&pads[0], (const int64_t *)&pads[2]));

        dnnl_primitive_desc_t pool_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&pool_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t pool;
        DNNL_CHECK(dnnl_primitive_create(&pool, pool_pd));

        // create destnation memory
        dnnl(Y)->reorder(dnnl_primitive_desc_query_md(pool_pd, dnnl_query_dst_md, 0));
        auto dst_mem = dnnl(Y)->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(pool, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(pool, pool_pd);
        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}

