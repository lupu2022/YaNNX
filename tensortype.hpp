#ifndef _YANNX_TENSORTYPE_HPP_
#define _YANNX_TENSORTYPE_HPP_

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <variant>
#include <cstring>

#include "yannx.hpp"

namespace yannx { namespace tt {

enum OperatorReturnType {
    YNX_OK = 0,
    YNX_TODO_ERROR = -1,
    YNX_INPUT_ERROR = -2,
    YNX_OUTPUT_ERROR = -3,
    YNX_ATTR_ERROR = -4,
};

enum TensorDataType {
    YNX_UNDEFINED = 0,
    YNX_FLOAT,
    YNX_UINT8,
    YNX_INT8,
    YNX_UINT16,
    YNX_INT16,
    YNX_INT32,
    YNX_INT64,
    YNX_STRING,
    YNX_BOOL,
    YNX_FLOAT16,
    YNX_DOUBLE,
    YNX_UINT32,
    YNX_UINT64,
    YNX_COMPLEX64,
    YNX_COMPLEX128,
    YNX_BFLOAT16
};

static const char* TensorDataTypeString[] = {
    "YNX_UNDEFINED",
    "YNX_FLOAT",
    "YNX_UINT8",
    "YNX_INT8",
    "YNX_UINT16",
    "YNX_INT16",
    "YNX_INT32",
    "YNX_INT64",
    "YNX_STRING",
    "YNX_BOOL",
    "YNX_FLOAT16",
    "YNX_DOUBLE",
    "YNX_UINT32",
    "YNX_UINT64",
    "YNX_COMPLEX64",
    "YNX_COMPLEX128",
    "YNX_BFLOAT16"
};

struct TensorType;
using tensor_t = std::shared_ptr<TensorType>;

//
//  https://github.com/onnx/onnx/blob/main/docs/IR.md#tensor-definition
//  scalar:         an empty shape with a defined data type
//  tensor:         shape dimention > 0
//  undefined:      empty shape with a undefined data type, used for type_shape inference.
//
//  ONNX based tensor computing API
//  https://github.com/onnx/onnx/blob/main/docs/IR.md
//  https://github.com/onnx/onnx/blob/main/docs/Operators.md
//
struct TensorType {
public:
    virtual ~TensorType() {}
    // must be common operator
    virtual const char* device() = 0;
    virtual const std::vector<size_t>& shape()  = 0;
    virtual TensorDataType dtype() = 0;
    // io functions
    virtual const void* get_data() = 0;
    virtual void set_data(const void* pdata) = 0;

    // reset undefined to a defined
    virtual void reset(TensorDataType dtype_, const std::vector<size_t>& shape_) = 0;
    virtual void reset(TensorDataType dtype_, const void* pvalue) = 0;
    virtual void reset(TensorDataType dtype_, const std::vector<size_t>& shape_, const void* pdata) = 0;

    // some fast access and help
    bool is_undefined() {
        if ( dtype() == YNX_UNDEFINED ) {
            return true;
        }
        return false;
    }
    bool is_scalar() {
        if ( dtype() != YNX_UNDEFINED && shape().size() == 0) {
            return true;
        }
        return false;
    }
    size_t num_items() {
        auto s = shape();
        size_t n = 1;
        for (size_t i = 0; i < s.size(); i++) {
            n = n * s[i];
        }
        return n;
    }
    std::string to_string() {
        std::ostringstream ss;
        ss << TensorDataTypeString[ dtype() ];
        ss << ":(";
        for (size_t i = 0; i < shape().size(); i++) {
            ss << shape()[i];
            if (i != shape().size() - 1) {
                ss << " ";
            }
        }
        ss << "):[";

        if ( dtype() == YNX_FLOAT ) {
            const float* pdata = (const float*)get_data();
            ss << dump(pdata);
        }
        if ( dtype() == YNX_INT64) {
            const int64_t* pdata = (const int64_t*)get_data();
            ss << dump(pdata);
        }
        ss << "]";
        return ss.str();
    }

    template<typename T>
    std::string dump(const T* data) {
        size_t items = num_items();
        std::ostringstream ss;
        if ( items <= 6 ) {
            for (size_t i = 0; i < items; i++) {
                ss << data[i] << " ";
            }
            return ss.str();
        }

        for (size_t i = 0; i < 3; i ++) {
            ss << data[i] << " ";
        }
        ss << " ... ";

        for (size_t i = items - 3; i < items; i ++) {
            ss << data[i] << " ";
        }
        return ss.str();
    }

    // following is ONNX operator set
#include "autogen/api_def.inc"

};

struct TensorFactory {
    static tensor_t create_undefined_user_tensor();
    static void register_user_tensor(tensor_t t, int64_t flag);
};

//
//  Following is a simple multiple type wrapped tensor
//

template <class T, TensorDataType _DTYPE_>
struct ValueOnlyTensor {
public:
    ValueOnlyTensor(const std::vector<size_t>& shape) {
        size_t n = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            n = n * shape[i];
        }
        value_.resize(n);
    }
    ValueOnlyTensor(const std::vector<size_t>& shape, const void* pdata){
        size_t n = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            n = n * shape[i];
        }
        value_.resize(n);
        memcpy( value_.data(), pdata, sizeof(T) * n );
    }
    ValueOnlyTensor(const void* pvalue) {
        value_.resize(1);
        value_[0] = *(const T*)pvalue;
    }

    TensorDataType dtype() {
        return _DTYPE_;
    }

    // value is only you need
    const void* data() {
        return (const void *)value_.data();
    }

    void fill(const void* pdata) {
        const T* mem = (T *)pdata;
        memcpy( value_.data(), mem,  value_.size() * sizeof(T) );
    }

private:
    std::vector<T> value_;
};

using value_float_t = ValueOnlyTensor<float, TensorDataType::YNX_FLOAT>;
using value_int64_t = ValueOnlyTensor<int64_t, TensorDataType::YNX_INT64>;
using value_bool_t = ValueOnlyTensor<unsigned char, TensorDataType::YNX_BOOL>;

template <TensorDataType _DTYPE_, typename DeviceImpl>
struct DeviceTensor : public TensorType {
public:
    // default is a undefined tensor
    DeviceTensor() : dtype_(YNX_UNDEFINED), impl_((void*)NULL) {
        static_assert(std::is_base_of<TensorType, DeviceImpl >::value, "Can't Using this tensor type!");
        static_assert((_DTYPE_ != YNX_INT64 ) && (_DTYPE_ != YNX_BOOL), "Don't support this type!");
    }
    ~DeviceTensor() {
    }

    DeviceImpl* impl() {
        if ( impl_.index() == DEVICE_IMPL ) {
            return (DeviceImpl *) std::get<DEVICE_IMPL>(impl_).get();
        }
        yannx_panic("Can't get impl from a none device tensor");
        return nullptr;
    }

    bool is_value_only() {
        if ( impl_.index() == DEVICE_IMPL ) {
            return false;
        }
        return true;
    }

    static DeviceImpl* impl(tensor_t& t) {
        DeviceTensor* dt = dynamic_cast<DeviceTensor *>(t.get());
        if ( dt ) {
            return dt->impl();
        }
        yannx_panic("Can't get impl from a none device tensor");
        return nullptr;
    }

    // fast access
    const char* device() override {
        if ( is_value_only() ) {
            return "ValueOnly";
        }
        return impl()->device();
    }
    TensorDataType dtype() override {
        return dtype_;
    }
    const std::vector<size_t>& shape() override {
        return shape_;
    }

    // reset to a normal tensor
    void reset(TensorDataType dtype, const std::vector<size_t>& shape) override {
        yannx_assert(dtype_ == YNX_UNDEFINED, "Can't reset a typed tensor!");
        yannx_assert(shape.size() > 0, "Can't reset with zero shape!");

        dtype_ = dtype;
        shape_ = shape;

        if ( dtype == _DTYPE_ ) {
            impl_ = std::make_unique<DeviceImpl>(shape);
            return;
        }

        if ( dtype == YNX_INT64 ) {
            impl_ = std::make_unique<value_int64_t>(shape);
            return;
        }

        if ( dtype == YNX_INT64 ) {
            impl_ = std::make_unique<value_int64_t>(shape);
            return;
        }

        if ( dtype == YNX_BOOL ) {
            impl_ = std::make_unique<value_int64_t>(shape);
            return;
        }

        yannx_panic("DeviceTensor::reset can't be here!");
    }

    // reset to a normal tensor with filled data
    void reset(TensorDataType dtype, const std::vector<size_t>& shape, const void* pdata) override {
        yannx_assert(dtype_ == YNX_UNDEFINED, "Can't reset a typed tensor!");
        yannx_assert(shape.size() > 0, "Can't reset a typed tensor with zero shape!");
        yannx_assert(impl_.index() == UNKNOW_ANY, "Can't reset a setted tensor!");

        dtype_ = dtype;
        shape_ = shape;

        if ( dtype == _DTYPE_ ) {
            impl_ = std::make_unique<DeviceImpl>(shape, pdata);
            return;
        }

        if ( dtype == YNX_FLOAT) {
            impl_ = std::make_unique<value_float_t>(shape, pdata);
            return;
        }

        if ( dtype == YNX_INT64) {
            impl_ = std::make_unique<value_int64_t>(shape, pdata);
            return;
        }

        if ( dtype == YNX_BOOL) {
            impl_ = std::make_unique<value_bool_t>(shape, pdata);
            return;
        }
        yannx_panic("DeviceTensor::reset can't be here!");
    }

    // reset to a scalar tensor
    void reset(TensorDataType dtype, const void* pvalue) override {
        yannx_assert(dtype_ == YNX_UNDEFINED, "Can't reset a typed tensor!");

        dtype_ = dtype;
        shape_.clear();

        if ( dtype == YNX_FLOAT) {
            impl_ = std::make_unique<value_float_t>(pvalue);
            return;
        }
        if ( dtype == YNX_INT64) {
            impl_ = std::make_unique<value_int64_t>(pvalue);
            return;
        }
        if ( dtype == YNX_BOOL) {
            impl_ = std::make_unique<value_bool_t>(pvalue);
            return;
        }
        yannx_panic("DeviceTensor::reset can't be here!");
    }

    const void* get_data() override {
        if ( is_value_only() ) {
            if ( impl_.index() == VALUE_FLOAT ) {
                return std::get<VALUE_FLOAT>(impl_)->data();
            }
            if ( impl_.index() == VALUE_INT64 ) {
                return std::get<VALUE_INT64>(impl_)->data();
            }
            if ( impl_.index() == VALUE_BOOL ) {
                return std::get<VALUE_BOOL>(impl_)->data();
            }
            yannx_panic("Can't be here!");
        }
        return impl()->get_data();
    }

    void set_data(const void* pdata) override {
        if ( is_value_only() ) {
            // check position wheather out of shape
            if ( impl_.index() == VALUE_FLOAT ) {
                return std::get<VALUE_FLOAT>(impl_)->fill(pdata);
            }
            if ( impl_.index() == VALUE_INT64 ) {
                return std::get<VALUE_INT64>(impl_)->fill(pdata);
            }
            if ( impl_.index() == VALUE_BOOL ) {
                return std::get<VALUE_BOOL>(impl_)->fill(pdata);
            }
            yannx_panic("Can't be here!");
        }
        return impl()->set_data(pdata);
    }

#include "autogen/api_impl.inc"


private:
    // basic info about tensor
    TensorDataType      dtype_;
    std::vector<size_t> shape_;

    // ImplType enum order is same as TensorImpl's variant
    enum ImplType {
        UNKNOW_ANY = 0,
        DEVICE_IMPL,
        VALUE_FLOAT,
        VALUE_INT64,
        VALUE_BOOL,
    };
    using TensorImpl = std::variant< void *,
                                     std::unique_ptr<DeviceImpl>,
                                     std::unique_ptr<value_float_t>,
                                     std::unique_ptr<value_int64_t>,
                                     std::unique_ptr<value_bool_t> >;
    TensorImpl impl_;
};

}} // end of namespace


#endif
