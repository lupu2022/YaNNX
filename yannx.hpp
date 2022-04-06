#include <string>
#include <vector>
#include <map>
#include <math.h>

#include <opt/variant.hpp>
#include <opt/debug.hpp>

namespace yannx {

using YNumber = float;

template<class YT>
struct Value {
    using YTensor = std::shared_ptr<YT>;
    using _Value = mpark::variant<const YNumber, const std::string, YTensor>;

public:

    Value() : cell_((YNumber)0.0) {}
    Value(YNumber n): cell_(n) {}
    Value(const std::string& s): cell_(s) {}
    Value(YTensor t): cell_(t) {}

    YNumber number() {
        if ( cell_.index() != ValueType::T_Number ) {
            yannx_panic("Value type error!");
        }

        return mpark::get<YNumber>(cell_);
    }

    const std::string& string() {
        if ( cell_.index() != ValueType::T_String ) {
            yannx_panic("Value type error!");
        }

        return mpark::get<const std::string>(cell_);
    }

    YTensor tensor() {
        if ( cell_.index() != ValueType::T_Tensor ) {
            yannx_panic("Value type error!");
        }

        return mpark::get<YTensor>(cell_);
    }

private:
    enum ValueType{
        T_Number = 0,
        T_String,
        T_Tensor,
    };

    _Value cell_;
};

template <class YT>
struct ValueStack {
    using YTensor = std::shared_ptr<YT>;

protected:
    virtual Value<YT> top() = 0;
    virtual Value<YT> pop() = 0;
    virtual void push( Value<YT> v) = 0;

public:
    // main control
    void drop() {
        pop();
    }
    void dup() {
        auto v1 = pop();
        push(v1);
    }
    void dup2() {
        auto v1 = pop();
        auto v2 = pop();
        push(v2);
        push(v1);
        push(v2);
        push(v1);
    }
    void swap() {
        auto v1 = pop();
        auto v2 = pop();
        push(v1);
        push(v2);
    }
    void rot() {
        auto v1 = pop();
        auto v2 = pop();
        auto v3 = pop();
        push(v2);
        push(v1);
        push(v3);
    }

    // accessing with type
    void push_number(YNumber n) {
        Value<YT> v(n);
        push(v);
    }
    YNumber pop_number() {
        auto v = pop();
        return v.number();
    }
    YNumber top_number() {
        auto c = top();
        return c.number();
    }

    void push_string(const char* s) {
        Value<YT> v(s);
        push(v);
    }
    std::string& pop_string() {
        auto v = pop();
        return v.string();
    }
    const char* top_string() {
        auto v = top();
        return v.string();
    }

    void push_tensor(YTensor* t) {
        Value<YT> v(t);
        push(v);
    }
    YTensor pop_tensor() {
        auto v = pop();
        return v.tensor();

    }
    YTensor* top_tensor() {
        auto v = top();
        return v.tensor();
    }

    // fast acess psedo list
    const std::vector<YNumber> pop_number_tuple() {
        YNumber sn = pop_number();
        yannx_assert( roundf(sn) == sn, "pop_number_tuple: size must be integer!");

        std::vector<YNumber> ret;
        ret.resize( (size_t)sn);
        for (size_t i = 0; i < ret.size(); i++) {
            ret[ ret.size() - 1 - i ] = pop_number();
        }
        return ret;
    }
    const std::vector<YTensor*> pop_tensor_tuple() {
        YNumber sn = pop_number();
        yannx_assert( roundf(sn) == sn, "pop_tensor_tuple: size must be integer!");

        std::vector<YTensor*> ret;
        ret.resize( (size_t)sn);
        for (size_t i = 0; i < ret.size(); i++) {
            ret[ ret.size() - 1 - i ] = pop_tensor();
        }
        return ret;
    }
};


template<class YT> struct NativeWord;
template<class YT> struct UserWord;

// txt -> tokens ->  SyntaxElement<YT> (parsed) -> SyntaxElement<YT> (linked)
template<class YT>
struct SyntaxElement {
    enum SyntaxType {
        // parsed value,
        T_Number,
        T_String,
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
};

template<class YT>
struct WordHash {
    using vmap_t = std::map<std::string, Value<YT> >;

    WordHash(vmap_t* g, vmap_t* l) {
        global_ = g;
        local_ = l;
    }
    void set(const char* name, Value<YT> item) {
        if ( local_->find(name) != local_->end() ) {
            yannx_panic("Hash only support write once!");
        }
        (*local_)[name] = item;
    }
    Value<YT> get(const char* name) {
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
    vmap_t* global_;
    vmap_t* local_;
};

// buit-in word implement in c++ code
template<class YT>
struct NativeWord {
    virtual void boot(ValueStack<YT>& stack, WordHash<YT>& hash) {
        run(stack);
    }
    virtual void run(ValueStack<YT>& stack) = 0;
};

// defined with user code
template<class YT>
struct UserWord {
    using vmap_t = std::map<std::string, Value<YT> >;
    using UserBinary = std::vector<SyntaxElement<YT> >;      // compiled

    UserWord(){}
    void boot(ValueStack<YT>& stack, vmap_t* global_hash_) {
        if ( global_hash_ == nullptr) {
            global_hash_ = &local_hash_;
        }
        local_hash_.clear();
        WordHash<YT> whash(global_hash_, &local_hash_);

        for (size_t i = 0; i < binary_.size(); i++) {
            if ( binary_[i].type_ == SyntaxElement<YT>::T_Number ) {
                stack.push_number( binary_[i].v_number);
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement<YT>::T_String ) {
                stack.push_string( binary_[i].v_string );
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement<YT>::T_NativeWord ) {
                binary_[i].v_nword->boot(stack, whash);
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement<YT>::T_UserWord ) {
                binary_[i].v_uword->boot(stack, global_hash_);
                continue;
            }
            yannx_panic("FIXME: Can't be here");
        }
    }
    void run(ValueStack<YT>& stack) {
        for (size_t i = 0; i < binary_.size(); i++) {
            if ( binary_[i].type_ == SyntaxElement<YT>::T_Number ) {
                stack.push_number( binary_[i].v_number);
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement<YT>::T_String ) {
                stack.push_string( binary_[i].v_string );
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement<YT>::T_NativeWord ) {
                binary_[i].v_nword->run(stack);
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement<YT>::T_UserWord ) {
                binary_[i].v_uword->run(stack);
                continue;
            }
            yannx_panic("FIXME: Can't be here");
        }
    }

private:
    UserBinary& bin(){
        return binary_;
    }
    vmap_t* hash() {
        return &local_hash_;
    }

    UserBinary binary_;
    vmap_t local_hash_;
    friend struct Runtime;
};

template<class YT>
struct Runtime : public ValueStack<YT>  {
    using YTensor = std::shared_ptr<YT>;

public:
    Runtime() {
    }
    ~Runtime() {
    }
protected:
    virtual void push(Value<YT>  v) {
        stack_.push(v);
    }
    virtual Value<YT> pop() {
        yannx_assert(stack_.size() >= 1, "pop: stack out of size!");
        auto c = stack_.back();
        stack_.pop_back();
        return c;
    }
    virtual Value<YT> top() {
        yannx_assert(stack_.size() >= 1, "pop: stack out of size!");
        auto c = stack_.back();
        return c;
    }

private:
    std::vector<Value<YT> > stack_;
};

}
