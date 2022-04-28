#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>


#define USING_ONNX_IMPL

#include "yannx.hpp"
#include "tensortype.hpp"

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

struct MyTensorType : public yannx::tt::TensorType {
    yannx::tt::TensorDataType dtype_;
    std::vector<size_t> shape_;
    std::vector<float> fvalue_;
    std::vector<int64_t> ivalue_;

    MyTensorType() {
        dtype_ = yannx::tt::YNX_UNDEFINED;
    }

    yannx::tt::TensorDataType dtype() override {
        return dtype_;
    }
    const std::vector<size_t>& shape() override {
        return shape_;
    }

    const void* value() override {
        if ( dtype_ == yannx::tt::YNX_UNDEFINED ) {
            yannx_panic("Can't access tensor value from undefined tensor");
        }
        if ( ivalue_.size() == 0 && fvalue_.size() == 0) {
            return nullptr;
        }
        if ( dtype_ == yannx::tt::YNX_FLOAT) {
            if ( fvalue_.size() == 0) {
                return nullptr;
            }
            return &fvalue_[0];
        }
        if ( dtype_ == yannx::tt::YNX_INT64) {
            if ( ivalue_.size() == 0) {
                return nullptr;
            }
            return &ivalue_[0];
        }
        return nullptr;
    }

    void reset(yannx::tt::TensorDataType dtype, std::vector<size_t>& shape) override {
        if ( dtype_ != yannx::tt::YNX_UNDEFINED ) {
            yannx_panic("Can't reset tensor more than once");
        }
        yannx_assert( dtype != yannx::tt::YNX_UNDEFINED, "Can't reset tensor with undefined");
        yannx_assert( shape.size() != 0, "Can't reset to scalar using this function");

        dtype_ = dtype;
        shape_ = shape;
    }
    void reset(yannx::tt::TensorDataType dtype, std::vector<size_t>& shape, const void* pdata) override {
        if ( dtype_ != yannx::tt::YNX_UNDEFINED ) {
            yannx_panic("Can't reset tensor more than once");
        }
        yannx_assert( dtype != yannx::tt::YNX_UNDEFINED, "Can't reset tensor with undefined");
        yannx_assert( shape.size() != 0, "Can't reset to scalar using this function");

        dtype_ = dtype;
        shape_ = shape;

        auto items_ = items();
        if ( dtype == yannx::tt::YNX_FLOAT ) {
            fvalue_.resize(items_, 0.0);
            const float* data = (const float*)pdata;
            for (size_t i = 0; i < items_; i++) {
                fvalue_[i] = data[i];
            }
            return;
        }
        if ( dtype == yannx::tt::YNX_INT64 ) {
            ivalue_.resize(items_, 0.0);
            const int64_t* data = (const int64_t *)pdata;
            for (size_t i = 0; i < items_; i++) {
                ivalue_[i] = data[i];
            }
            return;
        }
        yannx_panic("Reset tensor with un-support data type!");
    }
    void reset(yannx::tt::TensorDataType dtype, const void* pvalue) override {
        if ( dtype_ != yannx::tt::YNX_UNDEFINED ) {
            yannx_panic("Can't reset tensor more than once");
        }
        dtype_ = dtype;

        if ( dtype == yannx::tt::YNX_FLOAT ) {
            fvalue_.push_back( *(const float *)pvalue);
            return;
        }
        if ( dtype == yannx::tt::YNX_INT64 ) {
            ivalue_.push_back( *(const int64_t *)pvalue);
            return;
        }
        yannx_panic("Reset tensor(scalar) with un-support data type!");
    }
};

namespace yannx::tt {
    std::shared_ptr<TensorType> TensorType::create_undefined_user_tensor() {
        return std::make_shared<MyTensorType>();
    }

    void  TensorType::register_user_tensor(std::shared_ptr<TensorType> tensor, int64_t flag) {
    }
}

int main(const int argc, const char* argv[] ) {
    yannx::Runtime<yannx::tt::TensorType> runtime;
    yannx::tt::register_all_onnx_defined_words(runtime);

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
