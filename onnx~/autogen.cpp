#include <vector>
#include <iostream>
#include <sstream>

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>

using namespace onnx;

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

        auto op = schemas[i];

    }
}

