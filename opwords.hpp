#ifndef _YANNX_OPWORDS_HPP_
#define _YANNX_OPWORDS_HPP_

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#include <yannx.hpp>
#include <tensortype.hpp>

#include <onnx/onnx_pb.h>
#include <onnx/defs/schema.h>
#include <onnx/defs/attr_proto_util.h>
#include <onnx/defs/tensor_proto_util.h>
#include <onnx/shape_inference/implementation.h>

using namespace onnx;
using namespace yannx::tt;

namespace yannx { namespace opw {

TensorDataType datatype_from_string(const std::string& dt_str ) {
    for (size_t i = 0; i < YNX_BFLOAT16; i++) {
        if ( dt_str == TensorDataTypeString[i] ) {
            return (TensorDataType)i;
        }
    }
    return YNX_UNDEFINED;
};

TensorDataType datatype_from_onnx( int dt ) {
    switch( dt ) {
        case TensorProto_DataType_UNDEFINED:
            return YNX_UNDEFINED;
        case TensorProto_DataType_FLOAT:
            return YNX_FLOAT;
        case TensorProto_DataType_UINT8:
            return YNX_UINT8;
        case TensorProto_DataType_UINT16:
            return YNX_UINT16;
        case TensorProto_DataType_INT16:
            return YNX_INT16;
        case TensorProto_DataType_INT32:
            return YNX_INT32;
        case TensorProto_DataType_INT64:
            return YNX_INT64;
        case TensorProto_DataType_STRING:
            return YNX_STRING;
        case TensorProto_DataType_BOOL:
            return YNX_BOOL;
        case TensorProto_DataType_FLOAT16:
            return YNX_FLOAT16;
        case TensorProto_DataType_DOUBLE:
            return YNX_DOUBLE;
        case TensorProto_DataType_UINT32:
            return YNX_UINT32;
        case TensorProto_DataType_UINT64:
            return YNX_UINT64;
        case TensorProto_DataType_COMPLEX64:
            return YNX_COMPLEX64;
        case TensorProto_DataType_COMPLEX128:
            return YNX_COMPLEX128;
        case TensorProto_DataType_BFLOAT16:
            return YNX_BFLOAT16;
    }
    return YNX_UNDEFINED;
}

InferenceFunction query_inference_function(const std::string& op_name) {
    static std::map<const std::string, InferenceFunction> allInferenceFunctions;

    if ( allInferenceFunctions.size() == 0) {
        const std::vector<OpSchema> schemas = OpSchemaRegistry::get_all_schemas();

        for(size_t i = 0; i < schemas.size(); i++) {
            std::string name = schemas[i].Name();
            auto f = schemas[i].GetTypeAndShapeInferenceFunction();

            allInferenceFunctions[name] = f;
        }
    }

    if ( allInferenceFunctions.find(op_name) == allInferenceFunctions.end() ) {
        yannx_panic("Can't find InferenceFunction!");
    }

    return allInferenceFunctions[op_name];
}

struct YNXInferenceContextImpl : public InferenceContext {
    std::map<std::string, AttributeProto> attrs_;
    std::map<size_t, TypeProto> input_types_;
    std::map<size_t, TensorProto> input_datas_;
    std::map<size_t, TypeProto> output_types_;

    size_t input_num_;
    YNXInferenceContextImpl(std::vector<size_t> outs_) {
        input_num_ = 0;
        for (size_t i = 0; i < outs_.size(); i++) {
            TypeProto t;
            output_types_[ outs_[i] ] = t;
        }
    }

    // setup interfaces
    void new_attr(const std::string& name, const float v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, const int64_t v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, const std::string& v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, const std::vector<float>& v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, const std::vector<int64_t>& v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }
    void new_attr(const std::string& name, const std::vector<std::string>& v) {
        auto attr = MakeAttribute(name, v);
        attrs_[name] = attr;
    }

    void new_input(tensor_t t) {
        TypeProto proto;

        TypeProto_Tensor* p_tensor = proto.mutable_tensor_type();
        p_tensor->set_elem_type( t->dtype() );
        auto* shape = p_tensor->mutable_shape();

        shape->clear_dim();
        for (size_t i = 0; i < t->shape().size(); i++) {
            shape->add_dim();
            auto dim = shape->mutable_dim(i);
            dim->set_dim_value( t->shape()[i] );
        }
        size_t index = input_num_;
        input_types_[index] = proto;

        // converting tensortype to onnx's tensorproto
        if ( t->dtype() == YNX_FLOAT && t->device() == std::string("ValueOnly") ) {
            const float *d = (const float *)t->get_data();
            if ( d != nullptr ) {
                auto n = t->num_items();
                TensorProto t;
                t.set_data_type( TensorProto_DataType_FLOAT );
                t.clear_float_data();
                for (size_t i = 0; i < n; i++) {
                    t.add_float_data( d[i] );
                }

                input_datas_[index] = t;
            }
        }
        if ( t->dtype() == YNX_INT64 && t->device() == std::string("ValueOnly") ) {
            const int64_t *d = (const int64_t *)t->get_data();
            if ( d != nullptr ) {
                auto n = t->num_items();
                TensorProto t;
                t.set_data_type( TensorProto_DataType_INT64 );
                t.clear_int64_data();
                for (size_t i = 0; i < n; i++) {
                    t.add_int64_data( d[i] );
                }
                input_datas_[index] = t;
            }
        }

        input_num_ ++;
    }
    void new_input(std::variant<void *, tensor_t> v) {
        if ( v.index() == 1 ) {
            new_input(std::get<1>(v) );
        } else {
            input_num_ ++;
        }
    }
    void new_input(std::vector<tensor_t> v) {
        for (size_t i = 0; i < v.size(); i++) {
            new_input(v[i]);
        }
    }

    // call InferenceFunction
    void do_inference(InferenceFunction f) {
        f( *this );
    }

    int check_output(size_t index, tensor_t t) {
        auto* proto = getOutputType(index);
        auto p_tensor = proto->tensor_type();

        if (! p_tensor.has_elem_type() ) {
            return YNX_OUTPUT_ERROR;
        }
        TensorDataType dtype =  datatype_from_onnx(p_tensor.elem_type());

        std::vector<size_t> shape;
        auto shape_proto = p_tensor.shape();

        for (int i = 0; i < shape_proto.dim_size(); i++ ) {
            if ( !shape_proto.dim(i).has_dim_value() ) {
                return YNX_OUTPUT_ERROR;
            }
            shape.push_back( shape_proto.dim(i).dim_value() );
        }

        t->reset(dtype, shape);
        return YNX_OK;
    }

    // InferenceContext apis
    size_t getNumInputs() const override {
        return input_num_;
    }
    size_t getNumOutputs() const override {
        return output_types_.size();
    }
    const AttributeProto* getAttribute(const std::string& name) const override {
        if ( attrs_.find(name) != attrs_.end() ) {
            return &( attrs_.find(name)->second );
        }
        return nullptr;
    }
    const TypeProto* getInputType(size_t index) const override {
        if ( input_types_.find(index) != input_types_.end() ) {
            return &( input_types_.find(index)->second );
        }
        return nullptr;
    }
    const TensorProto* getInputData(size_t index) const override {
        if ( input_datas_.find(index) != input_datas_.end() ) {
            return &( input_datas_.find(index)->second );
        }
        return nullptr;
    }
    TypeProto* getOutputType(size_t index) override {
        if ( output_types_.find(index) != output_types_.end() ) {
            return &( output_types_.find(index)->second );
        }
        return nullptr;
    }

    // Skipping these impl, FIXME TensorProto seems not be used by Type&Shape inference
    const TensorShapeProto* getSymbolicInput(size_t index) const override {
        return nullptr;
    }
    const SparseTensorProto* getInputSparseData(size_t index) const override {
        return nullptr;
    }
    GraphInferencer* getGraphAttributeInferencer( const std::string& attr_name) override {
        return nullptr;
    }
};

//
//  some common help functions, and none-auto operators
//
static bool fetch_bool(ValueStack<TensorType>& stack) {
    float v = stack.pop_number();
    if ( v == 1) {
        return true;
    }
    yannx_assert(v == 0, "boolean must be 1 or 0!");
    return false;
}

static float fetch_float(ValueStack<TensorType>& stack) {
    float v = stack.pop_number();
    return v;
}

static int64_t fetch_int(ValueStack<TensorType>& stack) {
    int64_t v = stack.pop_number();
    return v;
}

static std::string fetch_string(ValueStack<TensorType>& stack) {
    std::string v = stack.pop_string();
    return v;
}

static tensor_t fetch_tensor(ValueStack<TensorType>& stack) {
    auto v = stack.pop_tensor();
    return v;
}

static std::vector<float> fetch_floats(ValueStack<TensorType>& stack) {
    auto v = stack.pop_number_tuple();
    std::vector<float> ret;
    for (size_t i = 0; i < v.size(); i++) {
        ret.push_back( v[i] );
    }
    return ret;
}

static std::vector<int64_t> fetch_ints(ValueStack<TensorType>& stack) {
    auto v = stack.pop_number_tuple();
    std::vector<int64_t> ret;
    for (size_t i = 0; i < v.size(); i++) {
        ret.push_back( v[i] );
    }
    return ret;
}

static std::vector<std::string> fetch_strings(ValueStack<TensorType>& stack) {
    auto v = stack.pop_string_tuple();
    return v;
}

static std::vector<tensor_t> fetch_tensors(ValueStack<TensorType>& stack) {
    auto v = stack.pop_tensor_tuple();
    return v;
}

static float fetch_optional_float(ValueStack<TensorType>& stack, float default_value) {
    if ( stack.top().is_none() ) {
        stack.pop();
        return default_value;
    }
    return fetch_float(stack);
}

static int64_t fetch_optional_int(ValueStack<TensorType>& stack, int64_t default_value) {
    if ( stack.top().is_none() ) {
        stack.pop();
        return default_value;
    }
    return fetch_int(stack);
}

static std::string fetch_optional_string(ValueStack<TensorType>& stack, std::string default_value ) {
    if ( stack.top().is_none() ) {
        stack.pop();
        return default_value;
    }
    return fetch_string(stack);
}

static std::vector<float> fetch_optional_floats(ValueStack<TensorType>& stack, std::vector<float> default_value) {
    if ( stack.top().is_none() ) {
        stack.pop();
        return default_value;
    }
    return fetch_floats(stack);
}

static std::vector<int64_t> fetch_optional_ints(ValueStack<TensorType>& stack, std::vector<int64_t> default_value) {
    if ( stack.top().is_none() ) {
        stack.pop();
        return default_value;
    }
    return fetch_ints(stack);
}

static std::vector<std::string> fetch_optional_strings(ValueStack<TensorType>& stack, std::vector<std::string> default_value) {
    if ( stack.top().is_none() ) {
        stack.pop();
        return default_value;
    }
    return fetch_strings(stack);
}

static std::variant<void *, tensor_t> fetch_optional_tensor(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        stack.pop();
        return std::variant<void *, tensor_t>(nullptr);
    }
    return std::variant<void *, tensor_t>( fetch_tensor(stack) );
}

static void put_tensor(ValueStack<TensorType>& stack, tensor_t t) {
    stack.push_tensor(t);
}

static void put_tensors(ValueStack<TensorType>& stack, std::vector<tensor_t>& ts) {
    for (size_t i = 0; i < ts.size(); i++) {
        stack.push_tensor(ts[i]);
    }
    stack.push_number(ts.size());
}

static void put_optional_tensor(ValueStack<TensorType>& stack, std::variant<void*, tensor_t>& ot) {
    if ( ot.index() == 0) {
        stack.push_none();
        return;
    }
    stack.push_tensor( std::get<1>(ot) );
}

// some help words, like Constant Parameter Variable
#define NWORD_CREATOR_DEFINE_TENSORTYPE(CLS)                                                \
    static std::shared_ptr<NativeWord<TensorType> >   creator(Runtime<TensorType>& rt ) {   \
        std::shared_ptr<NativeWord<TensorType> > wd(new CLS());                             \
        return wd;                                                                          \
    }

namespace common {
    struct Tensor : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            std::string dtype_string = fetch_string(stack);
            auto dtype = datatype_from_string(dtype_string);

            auto shape_ = fetch_ints(stack);
            std::vector<size_t> shape;
            for(size_t i = 0; i < shape_.size(); i++) {
                shape.push_back( shape_[i] );
            }

            output = rt.create_undefined_user_tensor();
            output->reset(dtype, shape);

            put_tensor(stack, output);
        }
        virtual void run(ValueStack<TensorType>& stack) {
            fetch_string(stack);
            fetch_ints(stack);
            put_tensor(stack, output);
        }

        NWORD_CREATOR_DEFINE_TENSORTYPE(Tensor)
    };

    struct Constant : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            std::string dtype_string = fetch_string(stack);
            auto dtype = datatype_from_string(dtype_string);

            auto shape_ = fetch_ints(stack);
            std::vector<size_t> shape;
            size_t items_ = 1;
            for(size_t i = 0; i < shape_.size(); i++) {
                shape.push_back( shape_[i] );
                items_ = items_ * shape_[i];
            }
            yannx_assert(shape_.size() > 0, "Can't create a empty shape(it is scalar)!");

            output = rt.create_undefined_user_tensor();
            if ( dtype == YNX_FLOAT) {
                auto values = fetch_floats(stack);
                if ( values.size() == 1 ) {
                    values = std::vector<float>(items_, values[0]);
                } else if ( items_ != values.size() ) {
                    yannx_panic("Create constant Tensor error, data size not eq shape!");
                }

                output->reset(dtype, shape, (void *)values.data());
            }
            if ( dtype == YNX_INT64) {
                auto values = fetch_ints(stack);
                if ( values.size() == 1 ) {
                    values = std::vector<int64_t>(items_, values[0]);
                } else if ( items_ != values.size() ) {
                    yannx_panic("Create constant Tensor error, data size not eq shape!");
                }
                output->reset(dtype, shape, (void *)values.data());
            }
            put_tensor(stack, output);
        }
        virtual void run(ValueStack<TensorType>& stack) {
            fetch_string(stack);
            fetch_ints(stack);
            fetch_floats(stack);

            put_tensor(stack, output);
        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Constant)
    };

    struct Scalar : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            std::string dtype_string = fetch_string(stack);
            auto dtype = datatype_from_string(dtype_string);

            output = rt.create_undefined_user_tensor();

            if ( dtype == YNX_FLOAT) {
                auto value = fetch_float(stack);
                output->reset(dtype, (void *)&value);
            }
            if ( dtype == YNX_INT64) {
                auto value = fetch_int(stack);
                output->reset(dtype, (void *)&value);
            }
            put_tensor(stack, output);
        }
        virtual void run(ValueStack<TensorType>& stack) {
            fetch_string(stack);
            fetch_float(stack);
            put_tensor(stack, output);
        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Scalar)
    };
}

#include "autogen/words_impl.inc"

//
//  Registering all words
//
void register_all_onnx_defined_words( Runtime<TensorType>& runtime) {

    runtime.new_nword("ynx.NewTensor~", common::Tensor::creator);
    runtime.new_nword("ynx.NewScalar~", common::Scalar::creator);
    runtime.new_nword("ynx.NewTensorWith~", common::Constant::creator);

#include "autogen/words_def.inc"

}

}}
#endif
