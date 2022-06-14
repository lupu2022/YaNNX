#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include <msgpack.hpp>
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

std::vector<unsigned char> fileToBuffer(const char* filename) {
    std::ifstream t(filename, std::ios::binary);
    std::vector<unsigned char> buf;

    t.seekg(0, std::ios::end);
    buf.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    buf.assign((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    return buf;
}


template<typename T>
void load_data(const char* weights_file, std::vector<T> &allPerform) {
    auto allWeights = std::move(fileToBuffer(weights_file));
    try {
        auto oh = msgpack::unpack((const char*)allWeights.data(), allWeights.size());
        allPerform = oh.get().as<std::vector<T>>();
    }
    catch (...) {
        std::cout << "Unpack weight error!" << std::endl;
        assert(false);
    }
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
using _ExpressBlob = std::tuple<
                                std::string,              //0 name
                                std::string,              //1 type
                                std::vector<size_t>,      //2 shape of tensor
                                std::vector<float> >;     //3 data of tensor
namespace yannx { namespace tt {
    static std::vector<tensor_t> allWeights;

    tensor_t TensorFactory::create_undefined_user_tensor() {
        auto ret = std::make_shared< user_tensor_t>();
        return ret;
    }
    void TensorFactory::register_user_tensor(tensor_t t, int64_t flag) {
        allWeights.push_back(t);
    }

    int write_registered_tensors(const char*weights_file) {
        std::vector<_ExpressBlob> allBlobs;
        load_data(weights_file, allBlobs);

        if ( allBlobs.size() != allWeights.size() ) {
            std::cout << "blob's number is not eq register tensors'" << std::endl;
            return -1;
        }

        for (size_t i = 0; i < allBlobs.size(); i++) {
            auto bshape = std::get<2>(allBlobs[i]);
            auto wshape = allWeights[i]->shape();

            if ( bshape != wshape ) {
                std::cout << std::get<0>(allBlobs[i]) << ": shape does not match" << std::endl;

                for (size_t j = 0; j < bshape.size(); j++) {
                    std::cout << bshape[j] << " " << wshape[j] << ", " ;
                }
                std::cout << std::endl;

                return -1;
            }

            allWeights[i]->set_data( std::get<3>(allBlobs[i]).data() );
        }
        return 0;
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
            yannx::tt::allWeights.clear();
            executor = runtime.boostrap(code);
            std::cout << "Received registered tensors: " << yannx::tt::allWeights.size() << std::endl;
        } else if ( code.find("l ") == 0) {
            std::vector<_ExpressBlob> blobs;
            auto weights_file = code.substr(2);
            std::cout << "Writing to registred tensors..." << std::endl;
            auto ret = write_registered_tensors( weights_file.c_str() );
            if ( ret == 0) {
                std::cout << "Done" << std::endl;
            } else {
                std::cout << "Failed" << std::endl;
            }

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
