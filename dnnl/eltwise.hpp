template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::dnnl_eltwise_operator(tt::tensor_t X, tt::tensor_t Y, dnnl_alg_kind_t algo, float alpha, float beta) {
    tt::TensorDataType dtype = _DTYPE_;

    if (dtype == tt::YNX_FLOAT) {
        // query memory and memory desc
        auto src_mem = dnnl(X)->dnnl_mem();
        auto src_md = dnnl(X)->dnnl_md();

        // create prim and pd
        dnnl_eltwise_desc_t desc;
        DNNL_CHECK(dnnl_eltwise_forward_desc_init(&desc,
                                                dnnl_forward_inference,
                                                algo,
                                                src_md, alpha, beta));
        dnnl_primitive_desc_t eltwise_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&eltwise_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t eltwise;
        DNNL_CHECK(dnnl_primitive_create(&eltwise, eltwise_pd));

        // create destnation memory
        auto dst_md = dnnl_primitive_desc_query_md(eltwise_pd, dnnl_query_dst_md, 0);
        dnnl(Y)->reorder(dst_md);
        auto dst_mem = dnnl(Y)->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[2];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);

        DNNL_CHECK(dnnl_primitive_execute(eltwise, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(eltwise, eltwise_pd);
        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}


template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_Clip(tt::tensor_t input, std::variant<void *, tt::tensor_t>& min_, std::variant<void *, tt::tensor_t>& max_, tt::tensor_t output) {
    tt::TensorDataType dtype = _DTYPE_;

    yannx_assert( min_.index() == 1, "Clip's min/max can't be empty!");
    yannx_assert( max_.index() == 1, "Clip's min/max can't be empty!");
    auto min = std::get<1>(min_);
    auto max = std::get<1>(max_);
    yannx_assert( min->is_scalar(), "Clip's min/max must be scalar!");
    yannx_assert( max->is_scalar(), "Clip's min/max must be scalar!");

    if (dtype == tt::YNX_FLOAT) {
        float min_value = min->item<float>( {} );
        float max_value = max->item<float>( {} );
        return dnnl_eltwise_operator(input, output, dnnl_eltwise_clip, min_value, max_value);
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}

