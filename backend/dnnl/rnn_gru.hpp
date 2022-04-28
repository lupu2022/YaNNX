
template <DataType _DTYPE_>
Result<option_tensor_t, TensorError> CPUTensor<_DTYPE_>::gru(tensor_t& weight_i, tensor_t& weight_h, tensor_t& bias_all,
                                                    option_tensor_t& hidden_input, option_tensor_t& hidden_output, tensor_t& dst) {
    DataType dtype = _DTYPE_;
    if ( dtype != DataType::Float) {
        return Err(TensorError("CPUTensor.onelayer_gru: type error"));
    }

    auto src_md = dnnl_md();
    auto weight_i_md = weight_i->cpu_float()->dnnl_md();
    auto weight_h_md = weight_h->cpu_float()->dnnl_md();
    auto bias_md = bias_all->cpu_float()->dnnl_md();
    auto hidden_i_md = hidden_input.isSome() ? hidden_input.unwrapSome()->cpu_float()->dnnl_md() : NULL;
    auto hidden_o_md = hidden_output.isSome() ? hidden_output.unwrapSome()->cpu_float()->dnnl_md() : NULL;
    auto dst_md = dst->cpu_float()->dnnl_md();

    dnnl_rnn_desc_t desc;
    DNNL_CHECK(dnnl_lbr_gru_forward_desc_init(&desc, dnnl_forward_inference , dnnl_unidirectional_left2right, 
                                          src_md,
                                          hidden_i_md,
                                          weight_i_md,
                                          weight_h_md,
                                          bias_md,
                                          dst_md,
                                          hidden_o_md, 0));

    dnnl_primitive_desc_t gru_pd;
    DNNL_CHECK(dnnl_primitive_desc_create(&gru_pd, &desc, nullptr, dnnl_help::DNNL_ENGINE_DEFAULT, nullptr));
    dnnl_primitive_t gru;
    DNNL_CHECK(dnnl_primitive_create(&gru, gru_pd));

    // create destnation memory
    dst->cpu_float()->reorder(dnnl_primitive_desc_query_md(gru_pd, dnnl_query_dst_md, 0));

    // create destnation memory
    auto ws_md = dnnl_primitive_desc_query_md(gru_pd, dnnl_query_workspace_md, 0);
    dnnl_memory_t ws_mem;
    DNNL_CHECK(dnnl_memory_create(&ws_mem, ws_md, dnnl_help::DNNL_ENGINE_DEFAULT, DNNL_MEMORY_ALLOCATE));
    
    // prepare arguments and execute
    dnnl_exec_arg_t args[16];
    dnnl_help::set_arg(args, 0, DNNL_ARG_SRC_LAYER, dnnl_mem());
    dnnl_help::set_arg(args, 1, DNNL_ARG_DST_LAYER, dst->cpu_float()->dnnl_mem());
    dnnl_help::set_arg(args, 2, DNNL_ARG_WEIGHTS_LAYER, weight_i->cpu_float()->dnnl_mem());
    dnnl_help::set_arg(args, 3, DNNL_ARG_WEIGHTS_ITER, weight_h->cpu_float()->dnnl_mem());
    dnnl_help::set_arg(args, 4, DNNL_ARG_BIAS, bias_all->cpu_float()->dnnl_mem());
    dnnl_help::set_arg(args, 5, DNNL_ARG_WORKSPACE, ws_mem);
    
    int nArg = 6;
    if (hidden_input.isSome()) {
        dnnl_help::set_arg(args, nArg, DNNL_ARG_SRC_ITER, hidden_input.unwrapSome()->cpu_float()->dnnl_mem());
        nArg++;
    }
    if (hidden_output.isSome()) {
        dnnl_help::set_arg(args, nArg, DNNL_ARG_DST_ITER, hidden_output.unwrapSome()->cpu_float()->dnnl_mem());
        nArg++;
    }
    DNNL_CHECK(dnnl_primitive_execute(gru, dnnl_help::DNNL_STREAM_DEFAULT, nArg, args));

    // destory prim and pd
    dnnl_help::release_prim(gru, gru_pd);

    // create worksapce
    auto shape = std::move(dnnl_help::query_shape_from_md(ws_md));
    auto shapet = ShapeType(shape);
    auto ws = new CPUTensor<DataType::Float>(shapet, ws_mem, nullptr);
    auto ws_tensor = TensorType::new_tensor(ws, shapet);

    option_tensor_t opt = Err(ws_tensor);
    return Ok ( opt );
}

