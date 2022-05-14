template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_LayerNormalization(tt::tensor_t X, tt::tensor_t Scale, std::variant<void *, tt::tensor_t>& B, tt::tensor_t Y, std::variant<void *, tt::tensor_t>& Mean, std::variant<void *, tt::tensor_t>& InvStdDev, int64_t axis, float epsilon, int64_t stash_type) {
    tt::TensorDataType dtype = _DTYPE_;

    yannx_assert( (size_t)axis == (X->shape().size() - 1), "DNNL's layernormliation only support last dimention as channel");
    yannx_assert( B.index() == 1, "DNNL's layernormliation must include B");
    yannx_assert( (Mean.index() == 0) && (InvStdDev.index() == 0), "DNN's layernormliation don't support other output");

    if ( dtype == tt::YNX_FLOAT ) {
        auto src_mem = dnnl(X)->dnnl_mem();
        auto src_md = dnnl(X)->dnnl_md();
        auto scale_mem = dnnl(Scale)->dnnl_mem();
        auto shift_mem = dnnl( std::get<1>(B))->dnnl_mem();

        dnnl_layer_normalization_desc_t desc;
        DNNL_CHECK(dnnl_layer_normalization_forward_desc_init(&desc, dnnl_forward_training, src_md, nullptr, epsilon,
                                                            dnnl_use_scale | dnnl_use_shift  ));

        dnnl_primitive_desc_t ln_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&ln_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t ln;
        DNNL_CHECK(dnnl_primitive_create(&ln, ln_pd));

        // create destnation memory
        dnnl(Y)->reorder(dnnl_primitive_desc_query_md(ln_pd, dnnl_query_dst_md, 0));
        auto dst_mem = dnnl(Y)->dnnl_mem();


        // prepare arguments and execute
        dnnl_exec_arg_t args[4];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);
        dnnl_help::set_arg(args, 2, DNNL_ARG_SCALE, scale_mem);
        dnnl_help::set_arg(args, 3, DNNL_ARG_SHIFT, shift_mem);

        DNNL_CHECK(dnnl_primitive_execute(ln, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(ln, ln_pd);

        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}
