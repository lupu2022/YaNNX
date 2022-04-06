#include <string>
#include <vector>
#include <map>

#include <opt/variant.hpp>
#include <opt/debug.hpp>

namespace yannx {

using YNumber = double;

template<class YT>
struct Value {
using YTensor = std::shared_ptr<YT>;
public:
    Value() : t(T_Number) {
        v._number = 0.0;
    }
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

public:
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
    ValueType t;
};

template <class YT>
struct ValueStack {
using YTensor = std::shared_ptr<YT>;
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

    virtual void push_tuple(const std::vector<Value<YT> >& v) = 0;
    virtual const std::vector<Value<YT> > pop_tuple() = 0;
    virtual const std::vector<Value<YT> > top_tuple() = 0;

    virtual void push_tensor(YTensor* v) = 0;
    virtual YTensor* pop_tensor() = 0;
    virtual YTensor* top_tensor() = 0;
};


template<class YT> struct NativeWord;
template<class YT> struct UserWord;

// txt -> tokens ->  SyntaxElement (parsed) -> SyntaxElement (linked)
template<class YT>
struct SyntaxElement {
using YTensor = std::shared_ptr<YT>;
    enum SyntaxType {
        // parsed value,
        T_Number,
        T_String,
        T_Tuple,
        T_NativeSymbol,
        T_UserSymbol,

        // linked value
        T_NativeWord,
        T_UserWord,
    };
    SyntaxType type_;

    YNumber v_number;
    std::string v_string;
    std::shared_ptr< NativeWord<YT> > v_nword;
    std::shared_ptr< UserWord<YT> > v_uword;

    std::vector< mpark::variant<YNumber, std::string> > v_tuple;
};

#if 0
template <class YT>
struct UserHash {
using YTensor = std::shared_ptr<YT>;
    struct HashItem {
        enum HashType {
            T_Number,
            T_Tensor,
        }
    };

public:
    UserHash(vhash_t* g, vhash_t* l) {
        global_ = g;
        local_ = l;
    }
    void set(const char* name, Value<YTensor> item) {
        yannx_assert(item.t != Value<YTensor>::T_Tuple, "hash's item don't support tuple!");
        if ( local_->find(name) != local_->end() ) {
            yannx_panic("Hash only support write once!");
        }
        (*local_)[name] = item;
    }
    Value<YTensor> get(const char* name) {
        if ( local_->find(name) == local_->end() ) {
            if (global_ != NULL) {
                if ( global_->find(name) == global_->end()) {
                    yannx_panic("Can't find hash item in local and global");
                }
                return (*global_)[name];
            } else {
                yannx_panic("Can't find hash item in global");
            }
        }
        return (*local_)[name];
    }

private:
    vhash_t* global_;
    vhash_t* local_;
};

// buit-in word implement in c++ code
struct NativeWord {
    virtual void boot(ValueStack& rt, UserHash& hash) {
        run(rt, hash);
    }
    virtual void run(ValueStack& rt, UserHash& hash) = 0;
};

// defined with user code
struct UserWord {
    UserWord(){}
    void boot(ValueStack& stack, vhash_t* global_hash_) {
        if ( global_hash_ == nullptr) {
            global_hash_ = &local_hash_;
        }
        local_hash_.clear();
        UserHash uhash(global_hash_, &local_hash_);

        for (size_t i = 0; i < binary_.size(); i++) {
            if ( binary_[i].type_ == SyntaxElement::T_Number ) {
                stack.push_number( binary_[i].v_number);
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement::T_String ) {
                stack.push_string( binary_[i].v_string.c_str() );
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement::T_Tuple ) {
                //stack.push_tuple( binary_[i].v_tuple );
                //TODO
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement::T_NativeWord ) {
                binary_[i].v_nword->boot(stack, uhash);
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement::T_UserWord ) {
                binary_[i].v_uword->boot(stack, global_hash_);
                continue;
            }
            yannx_panic("FIXME: Can't be here");
        }

    }

    void run(ValueStack& stack, vhash_t* global_hash_) {
        if ( global_hash_ == nullptr) {
            global_hash_ = &local_hash_;
        }
        UserHash uhash(global_hash_, &local_hash_);

        for (size_t i = 0; i < binary_.size(); i++) {
            if ( binary_[i].type_ == SyntaxElement::T_Number ) {
                stack.push_number( binary_[i].v_number);
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement::T_String ) {
                stack.push_string( binary_[i].v_string.c_str() );
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement::T_Tuple ) {
                //stack.push_tuple( binary_[i].v_tuple );
                // TODO
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement::T_NativeWord ) {
                binary_[i].v_nword->run(stack, uhash);
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement::T_UserWord ) {
                binary_[i].v_uword->run(stack, global_hash_);
                continue;
            }
            yannx_panic("FIXME: Can't be here");
        }
    }

private:
    UserBinary& bin(){
        return binary_;
    }
    vhash_t* hash() {
        return &local_hash_;
    }

    UserBinary binary_;
    vhash_t local_hash_;
    friend struct Runtime;
};

using nword_creator_t = std::shared_ptr<NativeWord> (*) (Runtime&);
#endif

template<class YT>
struct Runtime : public ValueStack<YT>  {
using YTensor = std::shared_ptr<YT>;
public:
    Runtime() {
    }
    ~Runtime() {
    }
    /*
    UserHash global() {
        if ( executor_ == nullptr) {
            yannx_panic("Can't run in uncompiled mode");
        }
        UserHash ret(nullptr, executor_->hash());
        return ret;
    }
    void boot(const std::string& txt);
    void run();
    */
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
        Value<YT> v(n);
        stack_.push_back(v);
    }
    virtual YNumber pop_number() {
        auto v = top_number();
        stack_.pop_back();
        return v;
    }
    virtual YNumber top_number() {
        yannx_assert(stack_.size() >= 1, "top_number: stack out of size!");
        yannx_assert(stack_.back().t != Value<YT>::T_Number, "top_number: value's type error!");

        Value<YT> c = stack_.back();
        return c.v._number;
    }

    virtual void push_string(const char* s) {
        Value<YT> v(s);
        stack_.push_back(v);
    }
    virtual const char* pop_string() {
        auto v = top_string();
        stack_.pop_back();
        return v;
    }
    virtual const char* top_string() {
        yannx_assert(stack_.size() >= 1, "top_string: stack out of size!");
        yannx_assert(stack_.back().t != Value<YT>::T_String, "top_string: value's type error!");

        Value<YT> c = stack_.back();
        return c.v._string;
    }

    virtual void push_tensor(YTensor* t) {
        Value<YT> v(t);
        stack_.push_back(v);
    }
    virtual YTensor* pop_tensor() {
        auto v = top_tensor();
        stack_.pop_back();
        return v;
    }
    virtual YTensor* top_tensor() {
        yannx_assert(stack_.size() >= 1, "top_tensor: stack out of size!");
        yannx_assert(stack_.back().t != Value<YTensor>::T_Tensor, "top_tensor: value's type error!");

        Value<YTensor> c = stack_.back();
        return c.v._tensor;
    }

    virtual void push_tuple(const std::vector<Value<YT> >& v) = 0;
    virtual const std::vector<Value<YT> > pop_tuple() {
        auto ret = top_tuple();
        for (size_t i = 0; i < ret.size(); i++) {
            stack_.pop_back();
        }
        return ret;
    }
    virtual const std::vector<Value<YT> > top_tuple() {
        yannx_assert(stack_.size() >= 1, "top_tuple: stack out of size!");
        yannx_assert(stack_.back().t == Value<YTensor>::T_Tuple, "top_tuple: value's type error!");

        size_t len = stack_.back().v._size;
        yannx_assert(stack_.size() >= (len + 1), "top_tuple: stack out of size!");

        std::vector<Value<YT> > ret;
        ret.push_back( stack_.back() );
        for (size_t i = stack_.size() - len - 1; i < stack_.size() - 1; i++) {
            yannx_assert(stack_[i].t != Value<YTensor>::T_Tuple, "top_tuple: tuple can't including tuple!");
            ret.push_back( stack_[i] );
        }
        return ret;
    }

private:
    void push(std::vector<Value<YT> >& v) {
        if ( v.size() == 1) {
            stack_.push_back(v[0]);
        } else {
            push_tuple(v);
        }
    }
    std::vector<Value<YT> > pop() {
        yannx_assert(stack_.size() >= 1, "pop: stack out of size!");

        std::vector<Value<YT> > v;
        v.push_back(stack_.back()); stack_.pop_back();
        if ( v.back().t != Value<YTensor>::T_Tuple) {
            return v;
        }
        return pop_tuple();
    }

private:
#if 0
    // Native and User dictionary
    std::map<std::string, nword_creator_t> ndict_;
    std::map<std::string, UserCode> udict_;

    // runtime stuff
    std::unique_ptr<UserWord> executor_;
#endif

    std::vector<Value<YT> > stack_;
};

}
