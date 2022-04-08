#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>

using namespace onnx;

static const char* TensorDataType[] = {
    "TD_UNDEFINED",
    "TD_FLOAT",
    "TD_UINT8",
    "TD_INT8",
    "TD_UINT16",
    "TD_INT16",
    "TD_INT32",
    "TD_INT64",
    "TD_STRING",
    "TD_BOOL",
    "TD_FLOAT16",
    "TD_DOUBLE",
    "TD_UINT32",
    "TD_UINT64",
    "TD_COMPLEX64",
    "TD_COMPLEX128",
    "TD_BFLOAT16"
};

static const char* ParameterType[] = {
    "tensor_t",                             // Signal
    "std::variant<void *, tensor_t>",       // Optional
    "std::vector<tensor_t>&"                // Variadic
};

/*
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
  AttributeProto_AttributeType_SPARSE_TENSORS = 12,
};
*/

std::string parse_tag(const std::string& filename) {
    const char delimeter = '/';
    std::stringstream ss(filename);
    std::string item;
    std::vector<std::string> splittedStrings;
    while (std::getline(ss, item, delimeter)) {
        splittedStrings.push_back(item);
    }
    return splittedStrings[ splittedStrings.size() - 2];
}

std::string api_generate(const OpSchema& op) {
    std::string op_name = op.Name();

    std::vector<std::string> tokens;

    auto allInputs = op.inputs();
    for(size_t i = 0; i < allInputs.size(); i++) {
        std::ostringstream oss;

        std::string iname = allInputs[i].GetName();
        std::string pt = ParameterType[ allInputs[i].GetOption() ];

        if ( i == 0 ) {
            oss << "/*inputs:*/ ";
        }
        oss << pt << " " << iname;
        tokens.push_back( oss.str() );
    }

    auto allOutputs = op.outputs();
    for(size_t i = 0; i < allOutputs.size(); i++) {
        std::ostringstream oss;

        std::string iname = allOutputs[i].GetName();
        std::string pt = ParameterType[ allOutputs[i].GetOption() ];

        if ( i == 0 ) {
            oss << "/*outputs:*/ ";
        }
        oss << pt << " " << iname;
        tokens.push_back( oss.str() );
    }

    auto allAttributes = op.attributes();
    for (auto i = allAttributes.begin(); i != allAttributes.end(); i++) {
        std::ostringstream oss;

        std::string attr_name = i->first;
        std::string type = "";
        switch ( i->second.type ) {
            case AttributeProto_AttributeType_FLOAT:
                type = "double";
                break;
            case AttributeProto_AttributeType_INT:
                type = "long";
                break;
            case AttributeProto_AttributeType_STRING:
                type = "std::string";
                break;
            case AttributeProto_AttributeType_FLOATS:
                type = "std::vector<double>";
                break;
            case AttributeProto_AttributeType_INTS:
                type = "std::vector<long>";
                break;
            case AttributeProto_AttributeType_STRINGS:
                type = "std::vector<std::string>";
                break;
            default:
                break;
        }
        if ( type == "" ) {
            std::cerr << "FIXME: can't be here !" << std::endl;
            std::cout << op << std::endl;
            exit(-1);
        }
        if ( i == allAttributes.begin() ) {
            oss << "/*attributes:*/ ";
        }

        if ( i->second.required ) {
            oss << type << " " << attr_name;
        } else {
            oss << "std::variant<void *," << type << " > " << attr_name;
        }

        tokens.push_back( oss.str() );
    }

    std::ostringstream oss;
    oss << "virtual OperatorReturnType onnx_" << op_name << "(";

    for (size_t i = 0; i < tokens.size(); i++) {
        oss << tokens[i] ;
        if ( i != tokens.size() - 1) {
            oss << ", ";
        }
    }

    oss << ") {" << std::endl;
    oss << "    return YNX_TODO_ERROR;" << std::endl;
    oss << "}" << std::endl;

    return oss.str();
}

std::string impl_generate(const OpSchema& op) {
    return "";
}

int main(int argc, char* argv[]) {
    const std::vector<OpSchema> schemas = OpSchemaRegistry::get_all_schemas();

    std::map<std::string, size_t> operators_by_name;
    std::map<std::string, std::vector<size_t> > operators_by_tag;

    // 0. fast scaning all the operators
    for (size_t i = 0; i < schemas.size(); i++) {
        // skip too old and new
        if ( schemas[i].Deprecated() ) {
            continue;
        }
        if ( schemas[i].support_level() == OpSchema::SupportType::EXPERIMENTAL ) {
            continue;
        }

        auto op = schemas[i];

        std::string tag = parse_tag( op.file() );
        if ( tag == "traditionalml" || tag == "controlflow" ) {
            continue;
        }

        std::string name = schemas[i].Name();
        if ( name.find("Constant") != std::string::npos ) {
            continue;
        }
        if ( name.find("Option") != std::string::npos ) {
            continue;
        }

        operators_by_name[name] = i;
        operators_by_tag[tag].push_back(i);
    }

    // 1. generating operator's API definement, sorted by abc
    std::ofstream ofs;
    ofs.open("onnx_def.hpp", std::ofstream::out);
    for (auto ii = operators_by_name.begin(); ii != operators_by_name.end(); ii++) {
        std::string api_code = api_generate( schemas[ ii->second ] );
        ofs << api_code << std::endl;
    }
    ofs.close();

    // 2. generating operator's implementation word, sorted by tags
    ofs.open("onnx_impl.hpp", std::ofstream::out);
    for (auto i = operators_by_tag.begin(); i != operators_by_tag.end(); i++) {
        ofs << "namespace " << i->first << " {" << std::endl;
        for (size_t ii = 0; ii < i->second.size(); ii++) {
            std::string api_code = impl_generate( schemas[ i->second[ii] ] );
        }
        ofs << "}" << std::endl;
    }
    ofs.close();
}

