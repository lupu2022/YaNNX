template <tt::TensorDataType _DTYPE_>
tt::OperatorReturnType DNNLTensor<_DTYPE_>::onnx_BatchNormalization(tt::tensor_t X, tt::tensor_t scale, tt::tensor_t B, tt::tensor_t input_mean, tt::tensor_t input_var, tt::tensor_t Y, std::variant<void *, tensor_t>& running_mean, std::variant<void *, tensor_t>& running_var, float epsilon, float momentum, int64_t training_mode) {
    tt::TensorDataType dtype = _DTYPE_;

    if ( dtype == tt::YNX_FLOAT ) {
        auto src_mem = dnnl(X)->dnnl_mem();
        auto src_md = dnnl(X)->dnnl_md();
        auto mean_mem = dnnl(input_mean)->dnnl_mem();
        auto var_mem = dnnl(input_var)->dnnl_mem();

        auto scale_mem = dnnl(scale)->dnnl_mem();
        auto shift_mem = dnnl(B)->dnnl_mem();

        dnnl_batch_normalization_desc_t desc;
        DNNL_CHECK(dnnl_batch_normalization_forward_desc_init(&desc, dnnl_forward_inference, src_md, epsilon,
                                                              dnnl_use_scale | dnnl_use_shift | dnnl_use_global_stats ));

        dnnl_primitive_desc_t bn_pd;
        DNNL_CHECK(dnnl_primitive_desc_create(&bn_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
        dnnl_primitive_t bn;
        DNNL_CHECK(dnnl_primitive_create(&bn, bn_pd));

        // create destnation memory
        dnnl(Y)->reorder(dnnl_primitive_desc_query_md(bn_pd, dnnl_query_dst_md, 0));
        auto dst_mem = dnnl(Y)->dnnl_mem();

        // prepare arguments and execute
        dnnl_exec_arg_t args[6];
        dnnl_help::set_arg(args, 0, DNNL_ARG_SRC, src_mem);
        dnnl_help::set_arg(args, 1, DNNL_ARG_DST, dst_mem);
        dnnl_help::set_arg(args, 2, DNNL_ARG_SCALE, scale_mem);
        dnnl_help::set_arg(args, 3, DNNL_ARG_SHIFT, shift_mem);
        dnnl_help::set_arg(args, 4, DNNL_ARG_MEAN, mean_mem);
        dnnl_help::set_arg(args, 5, DNNL_ARG_VARIANCE , var_mem);

        DNNL_CHECK(dnnl_primitive_execute(bn, dnnl_help::DNNL_STREAM_DEFAULT, sizeof(args)/sizeof(dnnl_exec_arg_t), args));

        // destory prim and pd
        dnnl_help::release_prim(bn, bn_pd);

        return tt::YNX_OK;
    }
    yannx_panic("Don't support data type");
    return tt::YNX_TODO_ERROR;
}
