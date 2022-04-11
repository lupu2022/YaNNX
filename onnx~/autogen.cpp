#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>

using namespace onnx;

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

void replace_all(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

std::string attribute_type_name(int t) {
    std::string type = "";
    switch ( t ) {
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
    assert(type != "");

    return type;
}


static const char* TensorParameterType[] = {
    "tensor_t",                          // Signal
    "std::variant<void *, tensor_t>&",   // Optional, shape = 0 & undefined
    "std::vector<tensor_t>&"             // Variadic
};

std::string api_generate(const OpSchema& op) {
    std::vector<std::string> tokens;

    std::string op_name = op.Name();
    auto allInputs = op.inputs();
    for(size_t i = 0; i < allInputs.size(); i++) {
        std::ostringstream oss;

        std::string iname = allInputs[i].GetName();
        std::string pt = TensorParameterType[ allInputs[i].GetOption() ];

        if ( i == 0 ) {
            oss << "/*inputs:*/ ";
        }
        oss << pt << " " << iname;
        tokens.push_back( oss.str() );
    }

    auto allOutputs = op.outputs();
    for(size_t i = 0; i < allOutputs.size(); i++) {
        std::ostringstream oss;

        std::string oname = allOutputs[i].GetName();
        std::string pt = TensorParameterType[ allOutputs[i].GetOption() ];

        if ( i == 0 ) {
            oss << "/*outputs:*/ ";
        }
        oss << pt << " " << oname;
        tokens.push_back( oss.str() );
    }

    auto allAttributes = op.attributes();
    for (auto i = allAttributes.begin(); i != allAttributes.end(); i++) {
        std::ostringstream oss;

        std::string attr_name = i->first;
        std::string type = attribute_type_name( i->second.type );
        if ( i == allAttributes.begin() ) {
            oss << "/*attributes:*/ ";
        }

        if ( i->second.required ) {
            oss << type << " " << attr_name;
        } else {
            oss << "std::variant<void *, " << type << " > " << attr_name;
        }

        tokens.push_back( oss.str() );
    }

    std::ostringstream oss;
    oss << "\t// https://github.com/onnx/onnx/blob/main/docs/Operators.md#" << op_name << std::endl;
    oss << "\tvirtual OperatorReturnType onnx_" << op_name << "(";

    for (size_t i = 0; i < tokens.size(); i++) {
        oss << tokens[i] ;
        if ( i != tokens.size() - 1) {
            oss << ", ";
        }
    }

    oss << ") {" << std::endl;
    oss << "\t    return YNX_TODO_ERROR;" << std::endl;
    oss << "\t}" << std::endl;
    return oss.str();
}

const char* word_template =  R"~~(
    struct #WORDNAME# : NativeWord<TensorType> {
#OUTPUT_DEF#
        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

#ATTR#
#INPUT#

#ifdef USING_NOONX

#else
#OUTPUT_INIT#
#endif


            if ( #CALL_API# != YNX_OK ) {
                yannx_panic("API: #WORDNAME#  return error!");
            }
#RETURN_OUTPUT#
        }
        virtual void run(ValueStack<TensorType>& stack) {
#ATTR#
#INPUT#
            if ( #CALL_API# != YNX_OK ) {
                yannx_panic("API: #WORDNAME#  return error!");
            }
#RETURN_OUTPUT#
        }
        NWORD_CREATOR_DEFINE(#WORDNAME#);
    }
)~~";

std::string impl_generate(const OpSchema& op) {
    std::string op_name = op.Name();

    std::string code = word_template;
    replace_all(code, "#WORDNAME#", op_name);

    // parsing attr
    {
        std::ostringstream oss;

        auto infos = op.attributes();
        for ( auto i = infos.rbegin(); i != infos.rend(); i++) {
            std::string aname = i->first;
            int t = i->second.type;
            int opt = (i->second.required == 0);

            switch ( t ) {
                case AttributeProto_AttributeType_FLOAT:
                    if (opt) {
                        oss << "\tauto " << aname << " = fetch_optional_float(stack);" << std::endl;
                    } else {
                        oss << "\tauto " << aname << " = fetch_float(stack);" << std::endl;
                    }
                    break;
                case AttributeProto_AttributeType_INT:
                    if (opt) {
                        oss << "\tauto " << aname << " = fetch_optional_int(stack);" << std::endl;
                    } else {
                        oss << "\tauto " << aname << " = fetch_int(stack);" << std::endl;
                    }
                    break;
                case AttributeProto_AttributeType_STRING:
                    if (opt) {
                        oss << "\tauto " << aname << " = fetch_optional_string(stack);" << std::endl;
                    } else {
                        oss << "\tauto " << aname << " = fetch_string(stack);" << std::endl;
                    }
                    break;
                case AttributeProto_AttributeType_FLOATS:
                    if (opt) {
                        oss << "\tauto " << aname << " = fetch_optional_floats(stack);" << std::endl;
                    } else {
                        oss << "\tauto " << aname << " = fetch_floats(stack);" << std::endl;
                    }
                    break;
                case AttributeProto_AttributeType_INTS:
                    if (opt) {
                        oss << "\tauto " << aname << " = fetch_optional_ints(stack);" << std::endl;
                    } else {
                        oss << "\tauto " << aname << " = fetch_ints(stack);" << std::endl;
                    }
                    break;
                case AttributeProto_AttributeType_STRINGS:
                    if (opt) {
                        oss << "\tauto " << aname << " = fetch_optional_strings(stack);" << std::endl;
                    } else {
                        oss << "\tauto " << aname << " = fetch_strings(stack);" << std::endl;
                    }
                    break;
                default:
                    assert(false);
                    break;
            }
        }
        auto attr_code = oss.str();
        replace_all(attr_code, "\t", "            ");
        replace_all(code, "#ATTR#", attr_code);
    }

    // processing input
    {
        std::ostringstream oss;

        auto allInputs = op.inputs();
        for(size_t i = 0; i < allInputs.size(); i++) {
            std::string iname = allInputs[i].GetName();
            int opt = allInputs[i].GetOption();
            if ( opt == 0 ) {
                oss << "\tauto " << iname << " = fetch_tensor(stack);" << std::endl;
            } else if ( opt == 1) {
                oss << "\tauto " << iname << " = fetch_optional_tensor(stack);" << std::endl;
            } else {
                oss << "\tauto " << iname << " = fetch_tensors(stack);" << std::endl;
            }
        }
        auto input_code = oss.str();
        replace_all(input_code, "\t", "            ");
        replace_all(code, "#INPUT#", input_code);
    }

    //processing output
    {
        auto allOutputs = op.outputs();
        {
            std::ostringstream oss;
            for(size_t i = 0; i < allOutputs.size(); i++) {
                int opt = allOutputs[i].GetOption();
                std::string oname = allOutputs[i].GetName();

                if ( opt == 0 ) {        // Single
                    oss << "\ttensor_t " << oname << ";" << std::endl;
                } else if ( opt == 1) {  // Optional
                    oss << "\tstd::variant<void *, tensor_t> " << oname << "(nullptr);" << std::endl;
                } else {                 // Variadic
                    oss << "\tstd::vector<tensor_t>" << oname << ";" << std::endl;
                }
            }
            std::string output_def = oss.str();
            replace_all(output_def, "\t", "        ");
            replace_all(code, "#OUTPUT_DEF#", output_def);
        }

        {
            std::ostringstream oss;
            for(size_t i = 0; i < allOutputs.size(); i++) {
                std::string oname = allOutputs[i].GetName();
                int opt = allOutputs[i].GetOption();
                if ( opt == 0 ) {        // Single
                    oss << "\t" << oname << " = create_undefined_tensor();" << std::endl;
                }
            }
            std::string output_init = oss.str();
            replace_all(output_init, "\t", "            ");
            replace_all(code, "#OUTPUT_INIT#", output_init);
        }
    }

    // call api
    {
        std::ostringstream oss;

        // find first tensor to executing API
        auto allInputs = op.inputs();
        auto allOutputs = op.outputs();
        auto allAttrs = op.attributes();
        if ( allInputs.size() > 0 && allInputs[0].GetOption() == 0) {
            oss << allInputs[0].GetName() << "->";
        } else if ( allOutputs.size() > 0 && allOutputs[0].GetOption() == 0) {
            oss << allOutputs[0].GetName() << "->";
        } else {
            std::cerr << op << std::endl;
            assert(false);
        }

        oss << "onnx_" << op.Name() << "(";

        std::vector<std::string> tokens;
        for(size_t i = 0; i < allInputs.size(); i++) {
            tokens.push_back( allInputs[i].GetName() );
        }
        for(size_t i = 0; i < allOutputs.size(); i++) {
            tokens.push_back( allOutputs[i].GetName() );
        }
        for (auto i = allAttrs.begin(); i != allAttrs.end(); i++) {
            tokens.push_back(i->first);
        }

        for (size_t i = 0; i < tokens.size(); i++) {
            oss << tokens[i] ;
            if ( i != tokens.size() - 1) {
                oss << ", ";
            }
        }
        oss << ")";

        std::string api_str = oss.str();
        //replace_all(api_str, "\t", "            ");
        replace_all(code, "#CALL_API#", api_str);
    }

    // return api
    {
        std::ostringstream oss;
        auto allOutputs = op.outputs();
        for(size_t i = 0; i < allOutputs.size(); i++) {
            int opt = allOutputs[i].GetOption();
            if ( opt == 0 ) {
                oss << "\tput_tensor(" << allOutputs[i].GetName() << ");" << std::endl;
            } else if ( opt == 1) {
                oss << "\tput_optional_tensor(" << allOutputs[i].GetName() << ");" << std::endl;
            } else {
                oss << "\tput_tensors(" << allOutputs[i].GetName() << ");" << std::endl;
            }
        }

        std::string ret_str = oss.str();
        replace_all(ret_str, "\t", "            ");
        replace_all(code, "#RETURN_OUTPUT#", ret_str);
    }
    return code;
}

std::string fileToString(const char* filename) {
    std::ifstream t(filename);
    std::string str;

    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    return str;
}


int main(int argc, char* argv[]) {
    const std::vector<OpSchema> schemas = OpSchemaRegistry::get_all_schemas();

    std::map<std::string, size_t> operators_by_name;
    std::map<std::string, std::vector<size_t> > operators_by_tag;

    auto result = fileToString("tensortype~.hpp");

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
        if ( tag == "traditionalml" || tag == "controlflow" || tag == "training" ) {
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
    std::ostringstream oss;
    for (auto ii = operators_by_name.begin(); ii != operators_by_name.end(); ii++) {
        std::string api_code = api_generate( schemas[ ii->second ] );
        oss << api_code << std::endl;
    }
    replace_all(result, "#ONNX_DEF#", oss.str());
    oss.clear();

    // 2. generating operator's implementation word, sorted by tags
    for (auto i = operators_by_tag.begin(); i != operators_by_tag.end(); i++) {
        oss << "namespace " << i->first << " {" << std::endl;
        for (size_t ii = 0; ii < i->second.size(); ii++) {
            std::string api_code = impl_generate( schemas[ i->second[ii] ] );
            oss << api_code << std::endl;
        }
        oss << "}" << std::endl;
    }
    std::string def_str = oss.str();
    replace_all(def_str, "\t", "    ");
    replace_all(result, "#ONNX_IMPL#", def_str);
    oss.clear();

    // 3. writing final result to file
    std::ofstream ofs;
    ofs.open("tensortype.hpp");
    ofs << result;
    ofs.close();
}

