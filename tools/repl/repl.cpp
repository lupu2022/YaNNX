#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include "yannx.hpp"
#include "tensortype.hpp"
#include "opwords.hpp"
#include "dnnl/impl.hpp"

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

bool readline(const std::string& prop, std::string& code) {
    std::cout << prop << std::flush;
    if (std::getline(std::cin, code)) {
        return true;
    }
    return false;
}

//
// User's factory
//
using user_tensor_t = DeviceTensor<yannx::tt::YNX_FLOAT, yannx::dnnl::DNNLTensor<yannx::tt::YNX_FLOAT>>;
namespace yannx { namespace tt {
    tensor_t TensorFactory::create_undefined_user_tensor() {
        auto ret = std::make_shared< user_tensor_t>();
        return ret;
    }
    void TensorFactory::register_user_tensor(tensor_t t, int64_t flag) {

    }
}}

int main(const int argc, const char* argv[] ) {
    yannx::Runtime<yannx::tt::TensorType> runtime;
    yannx::opw::register_all_onnx_defined_words(runtime);
    yannx::dnnl::dnnl_help::dnnl_begin();

    // 0. load all code to one string
    std::string txt;
    for (int i = 1; i < argc; i++) {
        auto codes = fileToString(argv[i]);
        txt = txt + codes + "\n";
    }

    // 1. boostrap pre-loading code
    if ( txt != "" ) {
        runtime.boostrap(txt);
    }

    // 2. entering command loop
    std::string code;
    std::shared_ptr<yannx::UserWord<yannx::tt::TensorType>> executor;
    while (readline(">> ", code)) {
        if ( code.find("b ") == 0) {
            code = code.substr(2);
            std::cout << "boostrap: " << code << std::endl;
            executor = runtime.boostrap(code);
        } else if ( code == "f" ) {
            if ( executor != nullptr) {
                auto start = std::chrono::high_resolution_clock::now();
                runtime.forward( executor );
                auto stop = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout << "time: " << duration.count() << std::endl;
            }
        } else {
            std::cout << "Command error!" << std::endl;
            std::cout << "b [code string]: boostrap followed code" << std::endl;
            std::cout << "r: just run boostraped code" << std::endl;
        }
    }
}
