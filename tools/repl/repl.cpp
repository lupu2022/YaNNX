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
using archived_tensor_t = std::vector< std::tuple<tensor_t, int>>;
using _ExpressBlob = std::tuple<
                                std::string,              //0 name
                                std::string,              //1 type
                                std::vector<size_t>,      //2 shape of tensor
                                std::vector<float> >;     //3 data of tensor
tensor_t create_undefined_user_tensor() {
    auto ret = std::make_shared< user_tensor_t>();
    return ret;
}

int write_registered_tensors(const char*weights_file, archived_tensor_t& allWeights ) {
    std::vector<_ExpressBlob> allBlobs;
    load_data(weights_file, allBlobs);

    if ( allBlobs.size() != allWeights.size() ) {
        std::cout << "blob's number is not eq register tensors'" << std::endl;
        return -1;
    }

    for (size_t i = 0; i < allBlobs.size(); i++) {
        auto bshape = std::get<2>(allBlobs[i]);
        auto wshape = std::get<0>(allWeights[i])->shape();

        if ( bshape != wshape ) {
            std::cout << std::get<0>(allBlobs[i]) << ": shape does not match" << std::endl;

            for (size_t j = 0; j < bshape.size(); j++) {
                std::cout << bshape[j] << " " << wshape[j] << ", " ;
            }
            std::cout << std::endl;

            return -1;
        }

        std::get<0>(allWeights[i])->set_data( std::get<3>(allBlobs[i]).data() );
    }
    return 0;
}

int main(const int argc, const char* argv[] ) {
    yannx::Runtime<yannx::tt::TensorType> runtime(create_undefined_user_tensor);
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
        } else if ( code.find("l ") == 0) {
            std::vector<_ExpressBlob> blobs;
            auto weights_file = code.substr(2);
            std::cout << "Writing to registred tensors..." << std::endl;
            auto ret = write_registered_tensors( weights_file.c_str(), runtime.archived() );
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
        } else if ( code == "cs" ) {
            runtime.clear();
            std::cout << "Cleaned stack" << std::endl;
        } else if ( code == "ls" ) {
            auto vec = runtime.vec();
            std::cout << "--- stack top ------" << std::endl;
            for (size_t i = 0; i < vec.size(); i++) {
                size_t ri = vec.size() - 1 - i;
                std::cout << i << ":\t" << vec[ri].to_string() << std::endl;
            }
            std::cout << "---- bottom --------" << std::endl;

        } else if ( code == "cr" ) {
            runtime.archived().clear();
            std::cout << "Cleaned registered tensors" << std::endl;
        } else if ( code == "lr" ) {
            std::cout << "Current registered tensor:" << runtime.archived().size() << std::endl;
        } else {
            std::cout << "Command error!" << std::endl;
            std::cout << "b [code string]: boostrap followed code" << std::endl;
            std::cout << "f: just run boostraped code again" << std::endl;
            std::cout << "l [msg file]: load msgpack file to registered tensors" << std::endl;
            std::cout << "cr: clean registered tensors" << std::endl;
            std::cout << "lr: list registered tensors" << std::endl;
            std::cout << "cs: clean stack" << std::endl;
        }
    }
}
