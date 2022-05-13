template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_Conv(tt::tensor_t X, tt::tensor_t W, std::variant<void *, tt::tensor_t>& B, tt::tensor_t Y, std::string auto_pad, std::vector<int64_t> dilations, int64_t group, std::vector<int64_t> kernel_shape, std::vector<int64_t> pads, std::vector<int64_t> strides) {
    tt::TensorDataType dtype = _DTYPE_;

    if ( dtype == tt::YNX_FLOAT ) {
        auto src_mem = dnnl(X)->dnnl_mem();
        auto src_md = dnnl(X)->dnnl_md();
        auto weight_mem = dnnl(W)->dnnl_mem();
        auto weight_md = dnnl(W)->dnnl_md();
        auto dst_md = dnnl(Y)->dnnl_md();

        dnnl_memory_desc_t* bias_md = nullptr;
        dnnl_memory_t bias_mem = nullptr;
        if ( B.index() == 1 ) {
            auto b = std::get<1>(B);
            bias_md = dnnl(b)->dnnl_md();
            bias_mem = dnnl(b)->dnnl_mem();
        }

        for (size_t i = 0; i < dilations.size(); i++) {
            dilations[i] = dilations[i] - 1;
        }

        // create prim and pd
        dnnl_convolution_desc_t desc;
        DNNL_CHECK(dnnl_dilated_convolution_forward_desc_init(&desc,
                                                              dnnl_forward_inference,
                                                              dnnl_convolution_auto,
                                                              src_md, weight_md, bias_md, dst_md,
                                                              strides.data(), dilations.data(),
                                                              (const int64_t *)&pads[0], (const int64_t *)&pads[2]));

        dnnl_primitive_desc_t conv_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&conv_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t conv;
        DNNL_CHECK(dnnl_primitive_create(&conv, conv_pd));

        // create destnation memory
        dnnl(Y)->reorder(dnnl_primitive_desc_query_md(conv_pd, dnnl_query_dst_md, 0));
        auto dst_mem = dnnl(Y)->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[4];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);
        dnnl_help::set_arg(args, 2, DNNL_ARG_WEIGHTS_0, weight_mem);
        dnnl_help::set_arg(args, 3, DNNL_ARG_BIAS, bias_mem);

        DNNL_CHECK(dnnl_primitive_execute(conv, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(conv, conv_pd);
        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}
