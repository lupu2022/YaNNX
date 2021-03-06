#ifndef _YANNX_HPP_
#define _YANNX_HPP_

#include <iostream>
#include <map>
#include <memory>
#include <variant>
#include <vector>
#include <sstream>
#include <string>

#include <math.h>

#define yannx_assert(Expr, Msg) \
    yannx::__M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)

#define yannx_panic(Msg) \
    yannx::__M_Panic(__FILE__, __LINE__, Msg)


namespace yannx {

inline void __M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg) {
    if (!expr) {
        std::cerr << "Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

inline void __M_Panic(const char* file, int line, const char* msg) {
    std::cerr << "Assert failed:\t" << msg << "\n"
        << "Source:\t\t" << file << ", line " << line << "\n";
    abort();
}
using YNumber = float;

template<class YT>
struct Value {
    using YTensor = std::shared_ptr<YT>;
    using _Value = std::variant<const YNumber,const std::string, YTensor>;

public:
    Value() : cell_((YNumber)0.0) {}
    Value(YNumber n): cell_(n) {}
    Value(const std::string& s): cell_(s) {}
    Value(YTensor t): cell_(t) {}

    YNumber number() {
        if ( cell_.index() != ValueType::T_Number ) {
            yannx_panic("Value type error!");
        }

        return std::get<0>(cell_);
    }

    const std::string& string() {
        if ( cell_.index() != ValueType::T_String ) {
            yannx_panic("Value type error!");
        }

        return std::get<const std::string>(cell_);
    }
    // none is a shadowed value, used for Optional parameters
    bool is_none() {
        if ( cell_.index() != ValueType::T_String ) {
            return false;
        }
        std::string s = std::get<1>(cell_);
        if ( s == "") {
            return true;
        }
        return false;
    }

    YTensor tensor() {
        if ( cell_.index() != ValueType::T_Tensor ) {
            yannx_panic("Value type error!");
        }

        return std::get<YTensor>(cell_);
    }

    std::string to_string() {
        std::ostringstream ss;
        if ( cell_.index() == ValueType::T_Number ) {
            ss << std::get<0>(cell_) ;
        } else if ( cell_.index() == ValueType::T_String ) {
            if ( std::get<1>(cell_) == "" ) {
                ss << "<none>";
            } else {
                ss << "\"" << std::get<1>(cell_) << "\"";
            }
        } else {
            ss << "<tensor>:" << std::get<2>(cell_)->to_string();
        }
        return ss.str();
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

    virtual Value<YT> top() = 0;
    virtual Value<YT> pop() = 0;
    virtual void push(Value<YT> v) = 0;
    virtual void clear() = 0;
    virtual std::vector<Value<YT> >& vec() = 0;

    virtual void archive(YTensor tensor, int flags) = 0;
    virtual std::vector<std::tuple<YTensor, int>>& archived() = 0;

    // main control
    void drop() {
        pop();
    }
    void dup() {
        auto v1 = top();
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
    void push_string(const std::string& s) {
        Value<YT> v(s);
        push(v);
    }
    std::string pop_string() {
        auto v = pop();
        return v.string();
    }
    const char* top_string() {
        auto v = top();
        return v.string();
    }
    void push_none() {
        Value<YT> v("");
        push(v);
    }

    void push_tensor(YTensor t) {
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

    // fast access psedo data struct : list
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
    const std::vector<YTensor> pop_tensor_tuple() {
        YNumber sn = pop_number();
        yannx_assert( roundf(sn) == sn, "pop_tensor_tuple: size must be integer!");

        std::vector<YTensor> ret;
        ret.resize( (size_t)sn);
        for (size_t i = 0; i < ret.size(); i++) {
            ret[ ret.size() - 1 - i ] = pop_tensor();
        }
        return ret;
    }
    const std::vector<std::string> pop_string_tuple() {
        YNumber sn = pop_number();
        yannx_assert( roundf(sn) == sn, "pop_string_tuple: size must be integer!");

        std::vector<std::string> ret;
        ret.resize( (size_t)sn);
        for (size_t i = 0; i < ret.size(); i++) {
            ret[ ret.size() - 1 - i ] = pop_string();
        }
        return ret;
    }

};

template<class YT>
struct WordHash {
    using vmap_t = std::map<std::string, Value<YT> >;

    WordHash(vmap_t* g, vmap_t* l) {
        global_ = g;
        local_ = l;
    }
    void set(const std::string& name, Value<YT> item) {
        if ( local_->find(name) != local_->end() ) {
            yannx_panic("Hash only support write once!");
        }
        //(*local_)[name] = item;
        local_->emplace(name, item);
    }
    Value<YT> get(const std::string& name) {
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


template<class YT> struct Runtime;
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


// buit-in word implement in c++ code
template<class YT>
struct NativeWord {
    virtual void boot(Runtime<YT>& rt, WordHash<YT>& hash) {
        run(rt);
    }
    virtual void run(ValueStack<YT>& stack) = 0;
};

// defined with user code
template<class YT>
struct UserWord {
    using vmap_t = std::map<std::string, Value<YT> >;
    using UserBinary = std::vector<SyntaxElement<YT> >;      // compiled

    UserWord(){}

    std::map<std::string, Value<YT>>& hash() {
        return local_hash_;
    }

private:
    void boot(Runtime<YT>& rt, vmap_t* global_hash_) {
        if ( global_hash_ == nullptr) {
            global_hash_ = &local_hash_;
        }
        local_hash_.clear();
        WordHash<YT> whash(global_hash_, &local_hash_);

        for (size_t i = 0; i < binary_.size(); i++) {
            if ( binary_[i].type_ == SyntaxElement<YT>::T_Number ) {
                rt.push_number( binary_[i].v_number);
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement<YT>::T_String ) {
                rt.push_string( binary_[i].v_string );
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement<YT>::T_NativeWord ) {
                binary_[i].v_nword->boot(rt, whash);
                continue;
            }
            if ( binary_[i].type_ == SyntaxElement<YT>::T_UserWord ) {
                binary_[i].v_uword->boot(rt, global_hash_);
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

    UserBinary& bin(){
        return binary_;
    }

    UserBinary binary_;
    vmap_t local_hash_;

    friend struct Runtime<YT>;
};

template<class YT>
struct Runtime : public ValueStack<YT>  {
    using YTensor = std::shared_ptr<YT>;
    using ntensor_creator_t = YTensor (*) ();
    using nword_t = std::shared_ptr<NativeWord<YT> >;
    using nword_creator_t = nword_t (*) (Runtime<YT>&);
    using UserCode = std::vector<SyntaxElement<YT> >;        // parsed
    using UserBinary = std::vector<SyntaxElement<YT> >;      // linked

public:
    Runtime(ntensor_creator_t creator) {
        tensor_creator_ = creator;
        register_builtin_native_words();
    }
    ~Runtime() {
    }

    YTensor create_undefined_user_tensor() {
        return tensor_creator_();
    }
    void new_nword(const char* name, nword_creator_t f) {
        if ( ndict_.find(name) != ndict_.end() ) {
            yannx_panic("Can't define native word same name!");
        }
        ndict_[name] = f;
    }

    std::shared_ptr<UserWord<YT>> boostrap(const std::string& txt) {
        auto executor = std::make_shared<UserWord<YT>>();

        UserCode main_code = parse(txt);
        build(main_code, executor->bin());
        executor->boot(*this, nullptr);

        return executor;
    }

    void forward(std::shared_ptr<UserWord<YT>> executor ) {
        if ( executor == nullptr) {
            yannx_panic("Can't run in uncompiled mode");
        }
        executor->run(*this);
    }

public:
    // for stack interfaces
    virtual void push(Value<YT> v) {
        stack_.push_back(v);
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
    virtual void clear() {
        stack_.clear();
    }
    virtual std::vector<Value<YT> >& vec() {
        return stack_;
    }

    virtual void archive(YTensor t, int flag) {
        archive_.push_back( {t, flag} );
    }
    virtual std::vector<std::tuple<YTensor, int>>& archived() {
        return archive_;
    }

private:
    // help functions
    UserCode parse(const std::string& txt) {
        struct _ {
            static bool parse_number(const std::string& token, YNumber& value) {
                if (isdigit(token.at(0)) || (token.at(0) == '-' && token.length() >= 2 && isdigit(token.at(1)))) {
                    if (token.find('.') != std::string::npos || token.find('e') != std::string::npos) { // double
                        value = atof(token.c_str());
                    } else {
                        value = atol(token.c_str());
                    }
                    return true;
                }
                return false;
            }
            static void tokenize(const std::string &str, const char delim, std::vector<std::string> &out) {

                size_t start;
                size_t end = 0;

                while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
                    end = str.find(delim, start);
                    out.push_back(str.substr(start, end - start));
                }
            }

            static void tokenize_line(std::string const &str_line, std::vector<std::string> &out) {
                std::string state = "SYMBOL";
                std::string current_str = "";

                for (size_t i = 0; i < str_line.size(); i++) {
                    char cur = str_line[i];
                    if ( state == "SYMBOL") {
                        if ( cur == ' ' || cur == '('  || cur == ')' ) {
                            // ending of a symbol
                            if (current_str != "") {
                                out.push_back(current_str);
                                current_str = "";
                            }
                            continue;
                        }
                        if ( cur == '{' || cur == '}' || cur == '[' || cur == ']' ) {
                            // ending of a symbol
                            if (current_str != "") {
                                out.push_back(current_str);
                                current_str = "";
                            }
                            out.push_back( std::string(1, cur) );
                            continue;
                        }

                        if ( cur == '"' ) {
                            if (current_str != "") {
                                yannx_panic("tokenize_line error!");
                            }

                            state = "STRING";
                            current_str.push_back('"');
                            continue;
                        }
                        if ( cur == '\'' ) {
                            if (current_str != "") {
                                yannx_panic("tokenize_line error!");
                            }

                            state = "STRING";
                            current_str.push_back('\'');
                            continue;
                        }

                        if ( cur == ';' ) {
                            if (current_str != "") {
                                out.push_back(current_str);
                            }
                            return;
                        }

                        current_str.push_back(cur);
                        continue;
                    }
                    if ( state == "STRING" ) {
                        if ( cur == '"' && current_str.at(0) == '"') {
                            current_str.push_back('"');
                            out.push_back(current_str);
                            current_str = "";
                            state = "SYMBOL";
                            continue;
                        }
                        if ( cur == '\'' && current_str.at(0) == '\'') {
                            current_str.push_back('\'');
                            out.push_back(current_str);
                            current_str = "";
                            state = "SYMBOL";
                            continue;
                        }
                        current_str.push_back(cur);
                    }
                }
                if ( state == "STRING" ) {
                    yannx_panic("tokenize_line error, string must end in one line!");
                }
                if (current_str != "") {
                    out.push_back(current_str);
                }
            }
        };

        // 0. removed comments & tokenize
        std::vector<std::string> tokens;
        std::istringstream code_stream(txt);
        std::string line;
        while (std::getline(code_stream, line)) {
            _::tokenize_line(line,  tokens);
        }

        // 1. tokens post processing
        std::vector<std::string> _tokens;
        std::vector< std::tuple<size_t, char> > scope_flags;
        size_t i = 0;
        while(i < tokens.size() ) {
            if ( tokens[i] == "[" ) {
                if ( scope_flags.size() > 0 ) {
                    if ( std::get<1>(scope_flags[0]) == '[') {
                        yannx_panic("Found nested auto list");
                    }
                }
                scope_flags.push_back( {i+1, '['} );

                _tokens.push_back("[");
                i = i + 1;
                continue;
            }
            if ( tokens[i] == "{" ) {
                if ( scope_flags.size() > 0 ) {
                    yannx_panic("Found nested loop macro");
                }
                scope_flags.push_back( {i+1, '{'} );

                i = i + 1;
                continue;
            }
            if ( tokens[i] == "]" ) {
                if ( scope_flags.size() == 0) {
                    yannx_panic("List without head!");
                }
                auto last = scope_flags.back();
                if ( std::get<1>(last) != '[' ) {
                    yannx_panic("List without head!");
                }
                scope_flags.pop_back();

                _tokens.push_back("]");
                i = i + 1;
                continue;
            }

            if ( tokens[i] == "}" ) {
                if ( scope_flags.size() == 0) {
                    yannx_panic("Loop without head!");
                }
                auto last = scope_flags.back();
                if ( std::get<1>(last) != '{' ) {
                    yannx_panic("Loop without head!");
                }

                auto loop_token = tokens[i+1];
                if ( loop_token.at(0) != '%' ) {
                    yannx_panic("Loop macro loop count must begin with a %!");
                }
                loop_token = loop_token.substr(1);

                YNumber loop_count;
                if ( ! _::parse_number( loop_token, loop_count ) ) {
                    yannx_panic("Loop macro must include a immediately loop count number!");
                }
                size_t loop_flag = std::get<0>(last);
                for(int c = 1; c < (int)loop_count; c++) {
                    for(size_t j = loop_flag; j < i; j++) {
                        _tokens.push_back( tokens[j] );
                    }
                }
                scope_flags.pop_back();
                i = i + 2;
                continue;
            }
            _tokens.push_back( tokens[i] );
            i = i + 1;
        }
        if ( scope_flags.size() != -0 ) {
            yannx_panic("Loop list without ending!");
        }
        tokens = _tokens;

        // 2. begin parsing all
        UserCode main_code;
        UserCode* target_code = &main_code;
        std::string target_name = "";
        i = 0;
        while (i < tokens.size()) {
            if ( tokens[i] == "def" ) {
                // don't support word define nested
                if ( target_name != "" ) {
                    yannx_panic("Find nested word definement !");
                }

                // enter a new user word define
                bool find_end = false;
                size_t j = i + 1;
                for (; j < tokens.size(); j++) {
                    if ( tokens[j] == "end" ) {
                        find_end = true;
                        break;
                    }
                }

                if ( find_end == false ) {
                    yannx_panic("Incompleted word. can't find 'end' !");
                }
                if ( tokens[i+1] == "end" ) {
                    yannx_panic("Incompleted word, can't find def _name_!");
                }

                target_name = tokens[i+1];
                if ( ndict_.find(target_name) != ndict_.end() ) {
                    yannx_panic("Can't define user word same name with native word!");
                }
                if ( udict_.find(target_name) != udict_.end() ) {
                    yannx_panic("Can't define user word same name with a user word defined already!");
                }

                target_code = new UserCode;
                i = i + 2;
                continue;
            }
            if ( tokens[i] == "end" ) {
                if (target_name == "") {
                    yannx_panic("find 'end' token without matched def!");
                }

                // create an new user defined word.
                udict_[target_name] = *target_code;

                delete target_code;
                target_code = &main_code;
                target_name = "";

                i = i + 1;
                continue;
            }

            SyntaxElement<YT> nobj;
            auto token = tokens[i];

            // check token is number
            YNumber num_value;
            if ( _::parse_number(token, num_value) ) {
                nobj.type_ = SyntaxElement<YT>::T_Number;
                nobj.v_number = num_value;
                target_code->push_back(nobj);

                i = i + 1;
                continue;
            }

            // check token is string
            if ( token.at(0) == '$' ) {
                nobj.type_ = SyntaxElement<YT>::T_String;
                nobj.v_string = token;
                target_code->push_back(nobj);

                i = i + 1;
                continue;
            }
            if ( token.at(0) == '"' ) {
                if ( token.size() >= 2 && token.back() == '"') {
                    nobj.type_ = SyntaxElement<YT>::T_String;
                    nobj.v_string = token.substr(1, token.size() - 2);
                    target_code->push_back(nobj);

                    i = i + 1;
                    continue;
                }
            }
            if ( token.at(0) == '\'' ) {
                if ( token.size() >= 2 && token.back() == '\'') {
                    nobj.type_ = SyntaxElement<YT>::T_String;
                    nobj.v_string = token.substr(1, token.size() - 2);
                    target_code->push_back(nobj);

                    i = i + 1;
                    continue;
                }
            }

            // check none value
            if ( token == "none" ) {
                nobj.type_ = SyntaxElement<YT>::T_String;
                nobj.v_string = "";
                target_code->push_back(nobj);

                i = i + 1;
                continue;
            }
            // check boolean
            if ( token == "true" ) {
                nobj.type_ = SyntaxElement<YT>::T_Number;
                nobj.v_number = 1;
                target_code->push_back(nobj);

                i = i + 1;
                continue;
            }
            if ( token == "false" ) {
                nobj.type_ = SyntaxElement<YT>::T_Number;
                nobj.v_number = 0;
                target_code->push_back(nobj);

                i = i + 1;
                continue;
            }

            // query words in dictionary
            if ( ndict_.find(token) != ndict_.end() ) {
                nobj.type_ = SyntaxElement<YT>::T_NativeSymbol;
                nobj.v_string = token;
                target_code->push_back(nobj);

                i = i + 1;
                continue;
            }
            if ( udict_.find(token) != udict_.end() ) {
                nobj.type_ = SyntaxElement<YT>::T_UserSymbol;
                nobj.v_string = token;
                target_code->push_back(nobj);

                i = i + 1;
                continue;
            }

            std::string msg = "Can't find matched word in dictionary for " + token;
            yannx_panic(msg.c_str());
        }

        return main_code;
    }

    void build(const UserCode& code, UserBinary& binary) {
        for (size_t i = 0; i < code.size(); i++) {
            if ( code[i].type_ == SyntaxElement<YT>::T_Number ) {
                binary.push_back(code[i]);
                continue;
            }
            if ( code[i].type_ == SyntaxElement<YT>::T_String ) {
                binary.push_back(code[i]);
                continue;
            }

            if ( code[i].type_ == SyntaxElement<YT>::T_NativeSymbol ) {
                if ( ndict_.find( code[i].v_string ) == ndict_.end() ) {
                    yannx_panic("Can't find native word");
                }

                SyntaxElement<YT> nobj;
                nobj.type_ = SyntaxElement<YT>::T_NativeWord;
                nobj.v_nword = ndict_[code[i].v_string](*this);

                binary.push_back(nobj);
                continue;
            }
            if ( code[i].type_ == SyntaxElement<YT>::T_UserSymbol ) {
                if ( udict_.find( code[i].v_string ) == udict_.end() ) {
                    yannx_panic("Can't find user word");
                }

                SyntaxElement<YT> nobj;
                nobj.type_ = SyntaxElement<YT>::T_UserWord;
                nobj.v_uword = std::make_shared<UserWord<YT> >();
                build( udict_[code[i].v_string] , nobj.v_uword->bin());

                binary.push_back(nobj);
                continue;
            }
            yannx_panic("FIXME: Can't be here");
        }
    }

    void register_builtin_native_words();

private:
    // User provied factory function
    ntensor_creator_t tensor_creator_;

    // Native and User dictionary
    std::map<std::string, nword_creator_t> ndict_;
    std::map<std::string, UserCode > udict_;

    // runtime stuff
    std::vector<Value<YT> > stack_;
    std::vector<std::tuple<YTensor, int> > archive_;
};

#define NWORD_CREATOR_DEFINE_YANNX(CLS)                                                           \
    static std::shared_ptr<NativeWord<YT> >   creator(Runtime<YT>& rt ) {   \
        std::shared_ptr<NativeWord<YT> > wd(new CLS<YT> ());                        \
        return wd;                                                                          \
    }


namespace builtin {

template<class YT>
struct Get : NativeWord<YT> {
    Value<YT>* pvalue;

    Get() { pvalue = nullptr; }
    ~Get() {
        if ( pvalue ) {
            delete pvalue;
        }
    }
    virtual void boot(Runtime<YT>& stack, WordHash<YT>& hash) {
        auto var = stack.pop_string();
        pvalue = new Value( hash.get(var) );
        stack.push( *pvalue );
    }
    virtual void run(ValueStack<YT>& stack) {
        stack.pop();
        stack.push( *pvalue );
    }
    NWORD_CREATOR_DEFINE_YANNX(Get)
};

template<class YT>
struct Set : NativeWord<YT> {
    virtual void boot(Runtime<YT>& stack, WordHash<YT>& hash) {
        auto var = stack.pop_string();
        auto value = stack.pop();
        hash.set(var, value);
    }
    virtual void run(ValueStack<YT>& stack) {
        stack.pop();
        stack.pop();
    }
    NWORD_CREATOR_DEFINE_YANNX(Set)
};

template<class YT>
struct PrintOne : NativeWord<YT> {
    virtual void boot(Runtime<YT>& stack, WordHash<YT>& hash) {
        auto v = stack.top();
        std::cout << v.to_string() << std::endl;
    }
    virtual void run(ValueStack<YT>& stack) {
        auto v = stack.top();
        std::cout << v.to_string() << std::endl;
    }
    NWORD_CREATOR_DEFINE_YANNX(PrintOne)
};

template<class YT>
struct PrintAll : NativeWord<YT> {
    virtual void boot(Runtime<YT>& stack, WordHash<YT>& hash) {
        run(stack);
    }
    virtual void run(ValueStack<YT>& stack) {
        auto vec = stack.vec();
        std::cout << "--- stack top ------" << std::endl;
        for (size_t i = 0; i < vec.size(); i++) {
            size_t ri = vec.size() - 1 - i;
            std::cout << i << ":\t" << vec[ri].to_string() << std::endl;
        }
        std::cout << "---- bottom --------" << std::endl;
    }
    NWORD_CREATOR_DEFINE_YANNX(PrintAll)
};

template<class YT>
struct Drop : NativeWord<YT> {
    virtual void run(ValueStack<YT>& stack) {
        stack.drop();
    }

    NWORD_CREATOR_DEFINE_YANNX(Drop)
};

template<class YT>
struct Rot : NativeWord<YT> {
    virtual void run(ValueStack<YT>& stack) {
        stack.rot();
    }

    NWORD_CREATOR_DEFINE_YANNX(Rot)
};

template<class YT>
struct Dup : NativeWord<YT> {
    virtual void run(ValueStack<YT>& stack) {
        stack.dup();
    }

    NWORD_CREATOR_DEFINE_YANNX(Dup)
};

template<class YT>
struct Dup2 : NativeWord<YT> {
    virtual void run(ValueStack<YT>& stack) {
        stack.dup2();
    }

    NWORD_CREATOR_DEFINE_YANNX(Dup2)
};

template<class YT>
struct Swap : NativeWord<YT> {
    virtual void run(ValueStack<YT>& stack) {
        stack.swap();
    }

    NWORD_CREATOR_DEFINE_YANNX(Swap)
};

template<class YT>
struct AutoList : NativeWord<YT> {
    const bool isHead_;
    AutoList(bool head): isHead_(head) { }

    virtual void run(ValueStack<YT>& stack) {
        if ( isHead_ ) {
            int pos = stack.vec().size();
            scopes().push_back(pos);
        } else {
            int cur_pos = stack.vec().size();
            int last_pos = scopes().back();
            scopes().pop_back();

            stack.push_number(cur_pos - last_pos);
        }
    }

    static std::shared_ptr<NativeWord<YT> > left_creator(Runtime<YT>& rt ) {
        std::shared_ptr<NativeWord<YT> > wd(new AutoList<YT> (true));
        return wd;
    }
    static std::shared_ptr<NativeWord<YT> > right_creator(Runtime<YT>& rt ) {
        std::shared_ptr<NativeWord<YT> > wd(new AutoList<YT> (false));
        return wd;
    }

private:
    static std::vector<int>& scopes () {
        static std::vector<int> scopes_;
        return scopes_;
    }
};

template<class YT>
struct Archive : NativeWord<YT> {
    virtual void boot(Runtime<YT>& stack, WordHash<YT>& hash) {
        int flag = (int)stack.pop_number();
        auto value = stack.pop_tensor();
        stack.archive(value, flag);

        stack.push_tensor(value);
    }
    virtual void run(ValueStack<YT>& stack) {
        stack.pop();
    }

    NWORD_CREATOR_DEFINE_YANNX(Archive)
};

}   // namespace builtin

template<class YT>
void Runtime<YT>::register_builtin_native_words() {
    new_nword("@", builtin::Get<YT>::creator); new_nword("get~", builtin::Get<YT>::creator);
    new_nword("!", builtin::Set<YT>::creator); new_nword("set~", builtin::Set<YT>::creator);

    new_nword("?", builtin::PrintOne<YT>::creator);
    new_nword("??", builtin::PrintAll<YT>::creator);

    new_nword("drop", builtin::Drop<YT>::creator);
    new_nword("dup", builtin::Dup<YT>::creator);
    new_nword("dup2", builtin::Dup2<YT>::creator);
    new_nword("swap", builtin::Swap<YT>::creator);
    new_nword("rot", builtin::Rot<YT>::creator);

    new_nword("[", builtin::AutoList<YT>::left_creator);
    new_nword("]", builtin::AutoList<YT>::right_creator);

    new_nword("arch~", builtin::Archive<YT>::creator);
}

}   // namespace yannx

#endif
