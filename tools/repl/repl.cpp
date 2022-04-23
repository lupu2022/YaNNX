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

size_t shape_items(const std::vector<size_t> shape) {
    size_t items = 1;
    for(size_t i = 0; i < shape.size(); i++) {
        items = items * shape[i];
    }
    return items;
}

struct MyTensorType : public yannx_tt::TensorType {
    yannx_tt::TensorDataType dtype_;
    std::vector<size_t> shape_;
    std::vector<float> value_;

    MyTensorType() {
        dtype_ = yannx_tt::YNX_UNDEFINED;
    }

    yannx_tt::TensorDataType dtype() override {
        return dtype_;
    }
    const std::vector<size_t>& shape() override {
        return shape_;
    }
    float scalar_value() override {
        if ( dtype_ == yannx_tt::YNX_UNDEFINED ) {
            yannx_panic("Can't access scalar value from undefined tensor");
        }
        if ( shape_.size() != 0) {
            yannx_panic("Can't access scalar value from normal tensor");
        }
        if ( value_.size() != 1) {
            yannx_panic("Scalar internal error!");
        }
        return value_[0];
    }
    const std::vector<float>& value() override {
        if ( dtype_ == yannx_tt::YNX_UNDEFINED ) {
            yannx_panic("Can't access tensor value from undefined tensor");
        }
        if ( shape_.size() == 0) {
            yannx_panic("Can't access tensor value from scalar");
        }
        return value_;
    }

    void reset(yannx_tt::TensorDataType dtype, std::vector<size_t>& shape) override {
        if ( dtype_ != yannx_tt::YNX_UNDEFINED ) {
            yannx_panic("Can't reset tensor more than once");
        }
        yannx_assert( dtype != yannx_tt::YNX_UNDEFINED, "Can't reset tensor with undefined");
        yannx_assert( shape.size() != 0, "Can't reset to scalar using this function");

        dtype_ = dtype;
        shape_ = shape;

        auto items = shape_items(shape);
        value_.resize(items, 0);

    }
    void reset(yannx_tt::TensorDataType dtype, std::vector<size_t>& shape, std::vector<float> value) override {
        if ( dtype_ != yannx_tt::YNX_UNDEFINED ) {
            yannx_panic("Can't reset tensor more than once");
        }
        yannx_assert( dtype != yannx_tt::YNX_UNDEFINED, "Can't reset tensor with undefined");
        yannx_assert( shape.size() != 0, "Can't reset to scalar using this function");

        dtype_ = dtype;
        shape_ = shape;

        auto items = shape_items(shape);
        yannx_assert( value.size() == items, "Filled data size error");

        value_ = value;
    }
    void reset(yannx_tt::TensorDataType dtype, float value) override {
        if ( dtype_ != yannx_tt::YNX_UNDEFINED ) {
            yannx_panic("Can't reset tensor more than once");
        }
        dtype_ = dtype;
        value_.push_back(value);
    }

};

namespace yannx_tt {
    std::shared_ptr<TensorType> TensorType::create_undefined_user_tensor() {
        return std::make_shared<MyTensorType>();
    }

    void  TensorType::register_user_tensor(std::shared_ptr<TensorType> tensor, int64_t flag) {

    }
}

int main(const int argc, const char* argv[] ) {
    yannx::Runtime<yannx_tt::TensorType> runtime;
    yannx_tt::register_all_onnx_defined_words(runtime);

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
    std::shared_ptr<yannx::UserWord<yannx_tt::TensorType>> executor;
    while (readline(">> ", code)) {
        if ( code.find("b ") == 0) {
            code = code.substr(2);
            std::cout << "boostrap: " << code << std::endl;
            executor = runtime.boostrap(code);
        } else if ( code == "r" ) {
            if ( executor != nullptr) {
                auto start = std::chrono::high_resolution_clock::now();
                runtime.run( executor );
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
