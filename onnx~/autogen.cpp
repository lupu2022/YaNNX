#include <vector>
#include <iostream>
#include <sstream>

#include <onnx/onnx_pb.h>
#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>
#include <onnx/shape_inference/implementation.h>

using namespace onnx;

static const char* ParameterOption[] = {
    "Single",
    "Optional",
    "Variadic"
};

static const char* AttrTypeName[] = {
  "UNDEFINED",
  "FLOAT",
  "INT",
  "STRING",
  "TENSOR",
  "GRAPH",
  "FLOAT_LIST",
  "INT_LIST",
  "STRING_LIST",
  "TENSORS",
  "GRAPHS",
  "SPARSE_TENSOR",
  "SPARSE_TENSORS",
};

static const char* DifferentiationCategory[] = {
  "Unknown",
  "Differentiable",
  "NonDifferentiable",
};

/*
enum TensorProto_DataType : int {
    TensorProto_DataType_UNDEFINED = 0,
    TensorProto_DataType_FLOAT = 1,
    TensorProto_DataType_UINT8 = 2,
    TensorProto_DataType_INT8 = 3,
    TensorProto_DataType_UINT16 = 4,
    TensorProto_DataType_INT16 = 5,
    TensorProto_DataType_INT32 = 6,
    TensorProto_DataType_INT64 = 7,
    TensorProto_DataType_STRING = 8,
    TensorProto_DataType_BOOL = 9,
    TensorProto_DataType_FLOAT16 = 10,
    TensorProto_DataType_DOUBLE = 11,
    TensorProto_DataType_UINT32 = 12,
    TensorProto_DataType_UINT64 = 13,
    TensorProto_DataType_COMPLEX64 = 14,
    TensorProto_DataType_COMPLEX128 = 15,
    TensorProto_DataType_BFLOAT16 = 16
};

enum AttributeProto_AttributeType : int {
  AttributeProto_AttributeType_UNDEFINED = 0,
  AttributeProto_AttributeType_FLOAT = 1,
  AttributeProto_AttributeType_INT = 2,
  AttributeProto_AttributeType_STRING = 3,
  AttributeProto_AttributeType_TENSOR = 4,
  AttributeProto_AttributeType_GRAPH = 5,
  AttributeProto_AttributeType_FLOATS = 6,
  AttributeProto_AttributeType_INTS = 7,
  AttributeProto_AttributeType_STRINGS = 8,
  AttributeProto_AttributeType_TENSORS = 9,
  AttributeProto_AttributeType_GRAPHS = 10,
  AttributeProto_AttributeType_SPARSE_TENSOR = 11,
  AttributeProto_AttributeType_SPARSE_TENSORS = 12
};
*/

std::string attrToString(int type, AttributeProto attr) {
    std::stringstream ss;
    switch(type) {
        case AttributeProto_AttributeType_FLOAT:
            ss << attr.f();
            break;
        case AttributeProto_AttributeType_INT:
            ss << attr.i();
            break;
        case AttributeProto_AttributeType_STRING:
            ss << attr.s();
            break;
        case AttributeProto_AttributeType_FLOATS:
            {
                auto vs = attr.floats();
                ss << "[";
                for(int i = 0; i < vs.size(); i++) {
                    ss << vs[i] << ", ";
                }
                ss << "]";
            }
            break;
        case AttributeProto_AttributeType_INTS:
            {
                auto vs = attr.ints();
                ss << "[";
                for(int i = 0; i < vs.size(); i++) {
                    ss << vs[i] << ", ";
                }
                ss << "]";
            }
            break;
        case AttributeProto_AttributeType_STRINGS:
            {
                auto vs = attr.strings();
                ss << "[";
                for(int i = 0; i < vs.size(); i++) {
                    ss << vs[i] << ", ";
                }
                ss << "]";
            }
            break;
        case AttributeProto_AttributeType_TENSOR:
            {
                ss << "TODO";
            }
    }
    return ss.str();
}

std::vector<std::string> split(const std::string& strToSplit, char delimeter = '/') {
    std::stringstream ss(strToSplit);
    std::string item;
    std::vector<std::string> splittedStrings;
    while (std::getline(ss, item, delimeter))
    {
        splittedStrings.push_back(item);
    }
    return splittedStrings;
}

int main(int argc, char* argv[]) {
    const std::vector<OpSchema> schemas = OpSchemaRegistry::get_all_schemas();
    for (size_t i = 0; i < schemas.size(); i++) {
        // skip too old and new
        if ( schemas[i].Deprecated() ) {
            continue;
        }
        if ( schemas[i].support_level() == OpSchema::SupportType::EXPERIMENTAL ) {
            continue;
        }
        if ( schemas[i].HasFunction() ) {
            continue;
        }

        std::cout << schemas[i] << std::endl;
    }
}

