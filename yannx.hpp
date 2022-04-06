#include <string>
#include <vector>

#include <opt/variant.hpp>
#include <opt/debug.hpp>

namespace yannx {
using YNumber = double;
using YTensor = int;


struct Value {
public:
    Value(const char *s) : t(T_String) {
        v._string = s;
    }
    Value(YNumber n) : t(T_Number) {
        v._number = n;
    }
    Value(YTensor* tensor) : t(T_Tensor) {
        v._tensor = tensor;
    }
    Value(size_t s) : t(T_Tuple) {
        v._size = s;
    }

private:
    enum ValueType {
        T_Number,
        T_String,
        T_Tensor,
        T_Tuple,
    };
    union {
        YNumber         _number;
        const char*     _string;
        YTensor*        _tensor;
        size_t          _size;
    } v;
    const ValueType t;
    friend struct Runtime;
};

struct ValueStack {
public:
    // basic operating
    virtual void drop() = 0;
    virtual void dup() = 0;
    virtual void dup2() = 0;
    virtual void swap() = 0;
    virtual void rot() = 0;

    // accessing with type
    virtual void push_number(YNumber v) = 0;
    virtual YNumber pop_number() = 0;
    virtual YNumber top_number() = 0;

    virtual void push_string(const char* v) = 0;
    virtual const char* pop_string() = 0;
    virtual const char* top_string() = 0;

    virtual void push_tuple(const std::vector<Value>& v) = 0;
    virtual const std::vector<Value> pop_tuple() = 0;
    virtual const std::vector<Value> top_tuple() = 0;

    virtual void push_tensor(YTensor* v) = 0;
    virtual YTensor* pop_tensor() = 0;
    virtual YTensor* top_tensor() = 0;
};

struct Runtime : public ValueStack  {
public:
    Runtime() {
    }
    ~Runtime() {
    }

public:
    virtual void drop() {
        pop();
    }
    virtual void dup() {
        auto v1 = pop();
        push(v1);
    }
    virtual void dup2() {
        auto v1 = pop();
        auto v2 = pop();
        push(v2);
        push(v1);
        push(v2);
        push(v1);
    }
    virtual void swap() {
        auto v1 = pop();
        auto v2 = pop();
        push(v1);
        push(v2);
    }
    virtual void rot() {
        auto v1 = pop();
        auto v2 = pop();
        auto v3 = pop();
        push(v2);
        push(v1);
        push(v3);
    }

    virtual void push_number(YNumber n) {
        Value v(n);
        stack_.push_back(v);
    }
    virtual YNumber pop_number() {
        auto v = top_number();
        stack_.pop_back();
        return v;
    }
    virtual YNumber top_number() {
        yannx_assert(stack_.size() >= 1, "top_number: stack out of size!");
        yannx_assert(stack_.back().t != Value::T_Number, "top_number: value's type error!");

        Value c = stack_.back();
        return c.v._number;
    }

    virtual void push_string(const char* s) {
        Value v(s);
        stack_.push_back(v);
    }
    virtual const char* pop_string() {
        auto v = top_string();
        stack_.pop_back();
        return v;
    }
    virtual const char* top_string() {
        yannx_assert(stack_.size() >= 1, "top_string: stack out of size!");
        yannx_assert(stack_.back().t != Value::T_String, "top_string: value's type error!");

        Value c = stack_.back();
        return c.v._string;
    }

    virtual void push_tensor(YTensor* t) {
        Value v(t);
        stack_.push_back(v);
    }
    virtual YTensor* pop_tensor() {
        auto v = top_tensor();
        stack_.pop_back();
        return v;
    }
    virtual YTensor* top_tensor() {
        yannx_assert(stack_.size() >= 1, "top_tensor: stack out of size!");
        yannx_assert(stack_.back().t != Value::T_Tensor, "top_tensor: value's type error!");

        Value c = stack_.back();
        return c.v._tensor;
    }

    virtual void push_tuple(const std::vector<Value>& v) = 0;
    virtual const std::vector<Value> pop_tuple() {
        auto ret = top_tuple();
        for (size_t i = 0; i < ret.size(); i++) {
            stack_.pop_back();
        }
        return ret;
    }
    virtual const std::vector<Value> top_tuple() {
        yannx_assert(stack_.size() >= 1, "top_tuple: stack out of size!");
        yannx_assert(stack_.back().t == Value::T_Tuple, "top_tuple: value's type error!");

        size_t len = stack_.back().v._size;
        yannx_assert(stack_.size() >= (len + 1), "top_tuple: stack out of size!");

        std::vector<Value> ret;
        ret.push_back( stack_.back() );
        for (size_t i = stack_.size() - len - 1; i < stack_.size() - 1; i++) {
            yannx_assert(stack_[i].t != Value::T_Tuple, "top_tuple: tuple can't including tuple!");
            ret.push_back( stack_[i] );
        }
        return ret;
    }

private:
    void push(std::vector<Value>& v) {
        if ( v.size() == 1) {
            stack_.push_back(v[0]);
        } else {
            push_tuple(v);
        }
    }
    std::vector<Value> pop() {
        yannx_assert(stack_.size() >= 1, "pop: stack out of size!");

        std::vector<Value> v;
        v.push_back(stack_.back()); stack_.pop_back();
        if ( v.back().t != Value::T_Tuple) {
            return v;
        }
        return pop_tuple();
    }

private:
    std::vector<Value> stack_;
};

}
