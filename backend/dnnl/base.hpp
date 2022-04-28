template <DataType _DTYPE_>
int  CPUTensor<_DTYPE_>::write(const unsigned char* data) {
    DataType dtype = _DTYPE_;

    auto total_size = DataType_size(dtype) * items();
    auto ptr = (unsigned char *)plain_ptr();
    memcpy(ptr, data, total_size);
    return 0;
}

template <DataType _DTYPE_>
int  CPUTensor<_DTYPE_>::read(unsigned char* data) {
    DataType dtype = _DTYPE_;

    auto total_size = DataType_size(dtype) * items();    
    auto ptr = (unsigned char*)plain_ptr();
    memcpy(data, ptr, total_size);
    return 0;
}

template <DataType _DTYPE_>
Result<ScalarValue, TensorError> CPUTensor<_DTYPE_>::get(const ShapeType& pos) {
    DataType dtype = _DTYPE_;

    uint64_t offset = 0;
    for (size_t i = 0; i < pos.size(); i++) {
        offset = offset + pos[i] * strides()[i];
    }

    reorder();
    if (dtype == DataType::Float) {
        auto ptr = (float *)plain_ptr();
        return Ok(ScalarValue( ptr[offset] ));
    }
    return Err(TensorError("CPUTensor.get: don't support DataType!"));
}

template <DataType _DTYPE_>
Result<void, TensorError>  CPUTensor<_DTYPE_>::set(const ShapeType& pos, ScalarValue& value) {
    DataType dtype = _DTYPE_;

    uint64_t offset = 0;
    for (size_t i = 0; i < pos.size(); i++) {
        offset = offset + pos[i] * strides()[i];
    }

    if (dtype == DataType::Float) {
        auto ptr = (float *)plain_ptr();
        ptr[offset] = (float)value.v();
        return Ok();
    }
    return Err(TensorError("CPUTensor.set: don't support DataType!"));
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::fill_(ScalarValue& value) {
    DataType dtype = _DTYPE_;

    auto total = items();
    reorder();
    if (dtype == DataType::Float) {
        auto ptr = (float *)plain_ptr();
        for(size_t i = 0; i < total; i++) {
            ptr[i] = (float)value.v();
        }
        return Ok();
    }

    return Err(TensorError("CPUTensor.fill_: don't support DataType!"));
}


template <DataType _DTYPE_>
Result<tensor_t, TensorError> CPUTensor<_DTYPE_>::copy() {
    DataType dtype = _DTYPE_;

    tensor_t ret = nullptr;
    void* dst_ptr = nullptr;
    if (dtype == DataType::Float) {
        ShapeType st(shape());
        auto cpu_tensor = build_dense_plain_float(st);
        ret = TensorType::new_tensor(cpu_tensor, st);
        dst_ptr = cpu_tensor->plain_ptr();
    }

    if (ret == nullptr) {
        return Err(TensorError("CPUTensor.copy: don't support DataType!"));
    }

    void* src_ptr = plain_ptr();
    memcpy(dst_ptr, src_ptr, DataType_size(dtype) * items());

    return Ok(ret);
}


template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::copy_(tensor_t& src) {
    DataType dtype = _DTYPE_;

    void* src_ptr = nullptr;
    if ( dtype == DataType::Float ) {
        src_ptr = src->cpu_float()->plain_ptr();
    } else {
        return Err(TensorError("CPUTensor.copy_: type error"));
    }

    void* dst_ptr = plain_ptr();
    memcpy(dst_ptr, src_ptr, DataType_size(dtype) * items());
    return Ok();
}

template <DataType _DTYPE_>
Result<tensor_t, TensorError> CPUTensor<_DTYPE_>::zero_like() {
    DataType dtype = _DTYPE_;

    tensor_t ret = nullptr;
    void* dst_ptr = nullptr;
    if (dtype == DataType::Float) {
        ShapeType st(shape());
        auto cpu_tensor = build_dense_plain_float(st);
        ret = TensorType::new_tensor(cpu_tensor, st);
        dst_ptr = cpu_tensor->plain_ptr();
    }
    if (ret == nullptr) {
        return Err(TensorError("CPUTensor.zero_like: type error"));
    }

    memset(dst_ptr, 0, DataType_size(dtype) * items());
    return Ok(ret);
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::zero_() {
    DataType dtype = _DTYPE_;

    void* dst_ptr = plain_ptr();
    memset(dst_ptr, 0, DataType_size(dtype) * items());
    return Ok();
}

template <DataType _DTYPE_>
Result<tensor_t, TensorError> CPUTensor<_DTYPE_>::new_like() {
    DataType dtype = _DTYPE_;

    tensor_t ret = nullptr;
    if (dtype == DataType::Float) {
        ShapeType st(shape());
        auto cpu_tensor = build_dense_plain_float(st);
        ret = TensorType::new_tensor(cpu_tensor, st);
    }
    if (ret == nullptr) {
        return Err(TensorError("CPUTensor..new_like: type error"));
    }

    return Ok(ret);
}

template <DataType _DTYPE_>
Result<tensor_t, TensorError> CPUTensor<_DTYPE_>::new_like(const ShapeType& new_shape) {
    DataType dtype = _DTYPE_;

    tensor_t ret = nullptr;
    if (dtype == DataType::Float) {
        ShapeType st(new_shape);
        auto cpu_tensor = build_dense_plain_float(st);
        ret = TensorType::new_tensor(cpu_tensor, st);
    }
    if (ret == nullptr) {
        return Err(TensorError("CPUTensor.new_like: type error"));
    }

    return Ok(ret);
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::reshape_to(tensor_t& dst) {
    DataType dtype = _DTYPE_;

    void* dst_ptr = nullptr;
    if ( dtype == DataType::Float ) {
        dst_ptr = dst->cpu_float()->plain_ptr();
    } else {
        return Err(TensorError("CPUTensor.reshape_to: type error"));
    }

    void* src_ptr = plain_ptr();
    memcpy(dst_ptr, src_ptr, DataType_size(dtype) * items());

    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::reshape_(const ShapeType& new_shape) {
    DataType dtype = _DTYPE_;

    // create new memory with new shape
    dnnl_memory_desc_t new_md;
    dnnl_memory_t new_mem;
    DNNL_CHECK(dnnl_memory_desc_init_by_tag(&new_md,
                                        shape_.size(),
                                        (const int64_t *)new_shape.dims(),
                                        dtype_,
                                        dnnl_help::ndim_to_mem_plain_tag(new_shape.size())));
    
    DNNL_CHECK(dnnl_memory_create(&new_mem, &new_md, dnnl_help::DNNL_ENGINE_DEFAULT, DNNL_MEMORY_ALLOCATE));
    void* new_ptr = nullptr;
    DNNL_CHECK(dnnl_memory_get_data_handle(new_mem, &new_ptr));

    // copy to new memory
    void* src_ptr = plain_ptr();
    memcpy(new_ptr, src_ptr, DataType_size(dtype) * items());

    // release old memory
    release();
    pd_ = nullptr;
    plain_md_ = new_md;
    mem_ = new_mem;

    // modify shape and strides
    std::vector<size_t>* ptr = const_cast<std::vector<size_t> *>(&shape_);
    *ptr = new_shape.vec();
    ptr = const_cast<std::vector<size_t> *>(&strides_);
    *ptr = new_shape.dense_strides();

    return Ok();
}

template <DataType _DTYPE_>
Result<void, TensorError> CPUTensor<_DTYPE_>::slice(uint64_t axis, uint64_t begin, uint64_t end, tensor_t& dst) {
    DataType dtype = _DTYPE_;

    unsigned char* dst_ptr = nullptr;
    if ( dtype == DataType::Float ) {
        dst_ptr = (unsigned char *)dst->cpu_float()->plain_ptr();
    } else {
        return Err(TensorError("CPUTensor.slice: type error"));
    }

    unsigned char* src_ptr = (unsigned char *)plain_ptr();

    uint64_t in_size = 1;
    for (size_t i = axis + 1; i < shape().size(); i++) {
        in_size = in_size * shape()[i];
    }

    uint64_t out_size = in_size * shape()[axis];
    uint64_t dst_size = in_size * (end - begin);
    uint64_t out_loop = 1;
    for (size_t i = 0; i < axis; i++) {
        out_loop = out_loop * shape()[i];
    }

    for (size_t i = 0; i < out_loop; i++) {
        uint64_t src_offset = i * out_size + begin * in_size;
        uint64_t dst_offset = i * dst_size;

        unsigned char* src = src_ptr + DataType_size(dtype) * src_offset;
        unsigned char* dst = dst_ptr + DataType_size(dtype) * dst_offset;

        memcpy(dst, src, DataType_size(dtype) * dst_size);
    }

    return Ok();
}