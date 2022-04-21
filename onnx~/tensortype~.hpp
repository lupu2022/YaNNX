//
//  this file is geneated by onnx~/autogen
//

#ifndef _YANXX_TENSORTYPE_HPP_
#define _YANNX_TENSORTYPE_HPP_

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#include <yannx.hpp>

#ifdef USING_ONNX_IMPL
#include <onnx/onnx_pb.h>
#include <onnx/defs/schema.h>
#include <onnx/defs/attr_proto_util.h>
#include <onnx/shape_inference/implementation.h>

using namespace onnx;
#endif


//
//  A simple onnx based (type and shape inference only, or a pure dummy tensor ) Tensor.following ONNX IR
//  https://github.com/onnx/onnx/blob/main/docs/IR.md
//  https://github.com/onnx/onnx/blob/main/docs/Operators.md
//

using namespace yannx;

namespace yannx_tt {

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

TensorDataType datatype_from_string(const std::string& dt_str ) {
    for (size_t i = 0; i < YNX_BFLOAT16; i++) {
        if ( dt_str == TensorDataTypeString[i] ) {
            return (TensorDataType)i;
        }
    }
    return YNX_UNDEFINED;
};


#ifdef USING_ONNX_IMPL
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
#endif


/*
 *  https://github.com/onnx/onnx/blob/main/docs/IR.md#tensor-definition
 *  scalar:         an empty shape with a defined data type
 *  tensor:         shape dimention > 0
 *  undefined:      empty shape with a undefined data type, used for type_shape inference.
 */

struct TensorType;
using tensor_t = std::shared_ptr<TensorType>;

struct TensorType {
    TensorDataType dtype_;
    std::vector<size_t> shape_;

    TensorType() {
        dtype_ = TensorDataType::YNX_UNDEFINED;
    }
    TensorType(TensorDataType dtype, std::vector<size_t>& shape) {
        reset(dtype, shape);
    }
    virtual void reset(TensorDataType dtype, std::vector<size_t>& shape) {
        dtype_ = dtype;
        shape_ = shape;
    }
    virtual std::string to_string() {
        std::ostringstream ss;
        ss << TensorDataTypeString[dtype_];
        ss << ":[";
        for (size_t i = 0; i < shape_.size(); i++) {
            ss << shape_[i];
            if (i != shape_.size() - 1) {
                ss << " ";
            }
        }
        ss << "]";
        return ss.str();
    }

    //
    //  User must be re-implement, return user side undefined tensor!
    //
    enum TensorFlag {
        TT_Constant,
        TT_Variable,
        TT_Parameter,
    };
    static tensor_t create_undefined_user_tensor();
    static void register_user_tensor(tensor_t, TensorFlag flag = TT_Constant);

#ONNX_DEF#

};

#ifdef USING_ONNX_IMPL
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
    std::vector<TypeProto> output_types_;

    size_t input_num_;
    const size_t output_num_;

    YNXInferenceContextImpl(size_t output_num) : output_num_(output_num) {
        input_num_ = 0;
        output_types_.resize( output_num_);
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
        p_tensor->set_elem_type( t->dtype_ );
        auto* shape = p_tensor->mutable_shape();

        shape->clear_dim();
        for (size_t i = 0; i < t->shape_.size(); i++) {
            shape->add_dim();
            auto dim = shape->mutable_dim(i);
            dim->set_dim_value( t->shape_[i] );
        }
        size_t index = input_num_;
        input_types_[index] = proto;

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
            new_input(v);
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
    int check_output(size_t index, std::variant<void *, tensor_t>& v) {
        tensor_t output = TensorType::create_undefined_user_tensor();
        if ( check_output(index, output) == YNX_OK ) {
            v = output;
        }
        return YNX_OK;
    }
    int check_output(size_t index, std::vector<tensor_t>& v) {
        yannx_panic("FIXME: how to check Variadic type");
        return YNX_OUTPUT_ERROR;
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
    TypeProto* getOutputType(size_t index) override {
        if ( index < output_types_.size() ) {
            return  &output_types_[index];
        }
        return nullptr;
    }

    // Skipping these impl, FIXME TensorProto seems not be used by Type&Shape inference
    const TensorProto* getInputData(size_t index) const override {
        return nullptr;
    }
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
#endif

//
//  some common help functions, and none-auto operators
//
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

static std::variant<void *, float> fetch_optional_float(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, float>(nullptr);
    }
    return std::variant<void *, float>( fetch_float(stack) );
}

static std::variant<void *, int64_t> fetch_optional_int(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, int64_t>(nullptr);
    }
    return std::variant<void *, int64_t>( fetch_int(stack) );
}

static std::variant<void *, std::string> fetch_optional_string(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::string>(nullptr);
    }
    return std::variant<void *, std::string>( fetch_string(stack) );
}

static std::variant<void *, tensor_t> fetch_optional_tensor(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, tensor_t>(nullptr);
    }
    return std::variant<void *, tensor_t>( fetch_tensor(stack) );
}

static std::variant<void *, std::vector<float> > fetch_optional_floats(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::vector<float> >(nullptr);
    }
    return std::variant<void *, std::vector<float> >( fetch_floats(stack) );
}

static std::variant<void *, std::vector<int64_t> > fetch_optional_ints(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::vector<int64_t> >(nullptr);
    }
    return std::variant<void *, std::vector<int64_t> >( fetch_ints(stack) );
}

static std::variant<void *, std::vector<std::string> > fetch_optional_strings(ValueStack<TensorType>& stack) {
    if ( stack.top().is_none() ) {
        return std::variant<void *, std::vector<std::string>> (nullptr);
    }
    return std::variant<void *, std::vector<std::string> >( fetch_strings(stack) );
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
    struct Constant : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto shape_ = fetch_ints(stack);
            std::vector<size_t> shape;
            for(size_t i = 0; i < shape_.size(); i++) {
                shape.push_back( shape_[i] );
            }

            std::string dtype_string = fetch_string(stack);
            auto dtype = datatype_from_string(dtype_string);

            output = TensorType::create_undefined_user_tensor();
            TensorType::register_user_tensor(output, TensorType::TT_Constant);
            output->reset(dtype, shape);

            put_tensor(stack, output);
        }
        virtual void run(ValueStack<TensorType>& stack) {
            fetch_ints(stack);
            fetch_string(stack);

            put_tensor(stack, output);
        }

        NWORD_CREATOR_DEFINE_TENSORTYPE(Constant)
    };

    struct Variable : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            auto shape_ = fetch_ints(stack);
            std::vector<size_t> shape;
            for(size_t i = 0; i < shape_.size(); i++) {
                shape.push_back( shape_[i] );
            }

            std::string dtype_string = fetch_string(stack);
            auto dtype = datatype_from_string(dtype_string);

            output = TensorType::create_undefined_user_tensor();
            output->reset(dtype, shape);

            put_tensor(stack, output);
        }
        virtual void run(ValueStack<TensorType>& stack) {
            put_tensor(stack, output);
        }

        NWORD_CREATOR_DEFINE_TENSORTYPE(Variable)
    };

}

#ONNX_IMPL#

//
//  Registering all words
//
void register_all_onnx_defined_words( Runtime<TensorType>& runtime) {

    runtime.new_nword("NewConstant", common::Constant::creator);
    runtime.new_nword("NewVariable", common::Variable::creator);

#ONNX_REGISTER#

}

}
#endif
