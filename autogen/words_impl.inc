//  this file is geneated by autogen, don't edit
namespace generator {

    struct RandomNormal : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto shape = fetch_ints(stack);
            auto seed = fetch_optional_float(stack, 0);
            auto scale = fetch_optional_float(stack, 1);
            auto mean = fetch_optional_float(stack, 0);
            auto dtype = fetch_optional_int(stack, 1);



            if ( output->onnx_RandomNormal_typing(output, dtype, mean, scale, seed, shape) == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("shape", shape);
                infer_.new_attr("seed", seed);
                infer_.new_attr("scale", scale);
                infer_.new_attr("mean", mean);
                infer_.new_attr("dtype", dtype);


                auto f = query_inference_function("RandomNormal");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( output->onnx_RandomNormal(output, dtype, mean, scale, seed, shape) != YNX_OK ) {
                yannx_panic("API: RandomNormal  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto shape = fetch_ints(stack);
            auto seed = fetch_optional_float(stack, 0);
            auto scale = fetch_optional_float(stack, 1);
            auto mean = fetch_optional_float(stack, 0);
            auto dtype = fetch_optional_int(stack, 1);



            if ( output->onnx_RandomNormal(output, dtype, mean, scale, seed, shape) != YNX_OK ) {
                yannx_panic("API: RandomNormal  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RandomNormal)
    };


    struct Range : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto delta = fetch_tensor(stack);
            auto limit = fetch_tensor(stack);
            auto start = fetch_tensor(stack);


            if ( start->onnx_Range_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(start);
                infer_.new_input(limit);
                infer_.new_input(delta);

                auto f = query_inference_function("Range");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( start->onnx_Range(start, limit, delta, output) != YNX_OK ) {
                yannx_panic("API: Range  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto delta = fetch_tensor(stack);
            auto limit = fetch_tensor(stack);
            auto start = fetch_tensor(stack);


            if ( start->onnx_Range(start, limit, delta, output) != YNX_OK ) {
                yannx_panic("API: Range  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Range)
    };


    struct RandomUniform : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto shape = fetch_ints(stack);
            auto seed = fetch_optional_float(stack, 0);
            auto low = fetch_optional_float(stack, 0);
            auto high = fetch_optional_float(stack, 1);
            auto dtype = fetch_optional_int(stack, 1);



            if ( output->onnx_RandomUniform_typing(output, dtype, high, low, seed, shape) == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("shape", shape);
                infer_.new_attr("seed", seed);
                infer_.new_attr("low", low);
                infer_.new_attr("high", high);
                infer_.new_attr("dtype", dtype);


                auto f = query_inference_function("RandomUniform");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( output->onnx_RandomUniform(output, dtype, high, low, seed, shape) != YNX_OK ) {
                yannx_panic("API: RandomUniform  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto shape = fetch_ints(stack);
            auto seed = fetch_optional_float(stack, 0);
            auto low = fetch_optional_float(stack, 0);
            auto high = fetch_optional_float(stack, 1);
            auto dtype = fetch_optional_int(stack, 1);



            if ( output->onnx_RandomUniform(output, dtype, high, low, seed, shape) != YNX_OK ) {
                yannx_panic("API: RandomUniform  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RandomUniform)
    };


    struct RandomUniformLike : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto seed = fetch_optional_float(stack, 0);
            auto low = fetch_optional_float(stack, 0);
            auto high = fetch_optional_float(stack, 1);
            auto dtype = fetch_optional_int(stack, 0);

            auto input = fetch_tensor(stack);


            if ( input->onnx_RandomUniformLike_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("seed", seed);
                infer_.new_attr("low", low);
                infer_.new_attr("high", high);
                infer_.new_attr("dtype", dtype);

                infer_.new_input(input);

                auto f = query_inference_function("RandomUniformLike");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_RandomUniformLike(input, output, dtype, high, low, seed) != YNX_OK ) {
                yannx_panic("API: RandomUniformLike  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto seed = fetch_optional_float(stack, 0);
            auto low = fetch_optional_float(stack, 0);
            auto high = fetch_optional_float(stack, 1);
            auto dtype = fetch_optional_int(stack, 0);

            auto input = fetch_tensor(stack);


            if ( input->onnx_RandomUniformLike(input, output, dtype, high, low, seed) != YNX_OK ) {
                yannx_panic("API: RandomUniformLike  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RandomUniformLike)
    };


    struct EyeLike : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto k = fetch_optional_int(stack, 0);
            auto dtype = fetch_optional_int(stack, 0);

            auto input = fetch_tensor(stack);


            if ( input->onnx_EyeLike_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("k", k);
                infer_.new_attr("dtype", dtype);

                infer_.new_input(input);

                auto f = query_inference_function("EyeLike");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_EyeLike(input, output, dtype, k) != YNX_OK ) {
                yannx_panic("API: EyeLike  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto k = fetch_optional_int(stack, 0);
            auto dtype = fetch_optional_int(stack, 0);

            auto input = fetch_tensor(stack);


            if ( input->onnx_EyeLike(input, output, dtype, k) != YNX_OK ) {
                yannx_panic("API: EyeLike  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(EyeLike)
    };


    struct Bernoulli : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto seed = fetch_optional_float(stack, 0);
            auto dtype = fetch_optional_int(stack, 0);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Bernoulli_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("seed", seed);
                infer_.new_attr("dtype", dtype);

                infer_.new_input(input);

                auto f = query_inference_function("Bernoulli");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Bernoulli(input, output, dtype, seed) != YNX_OK ) {
                yannx_panic("API: Bernoulli  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto seed = fetch_optional_float(stack, 0);
            auto dtype = fetch_optional_int(stack, 0);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Bernoulli(input, output, dtype, seed) != YNX_OK ) {
                yannx_panic("API: Bernoulli  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Bernoulli)
    };


    struct RandomNormalLike : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto seed = fetch_optional_float(stack, 0);
            auto scale = fetch_optional_float(stack, 1);
            auto mean = fetch_optional_float(stack, 0);
            auto dtype = fetch_optional_int(stack, 0);

            auto input = fetch_tensor(stack);


            if ( input->onnx_RandomNormalLike_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("seed", seed);
                infer_.new_attr("scale", scale);
                infer_.new_attr("mean", mean);
                infer_.new_attr("dtype", dtype);

                infer_.new_input(input);

                auto f = query_inference_function("RandomNormalLike");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_RandomNormalLike(input, output, dtype, mean, scale, seed) != YNX_OK ) {
                yannx_panic("API: RandomNormalLike  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto seed = fetch_optional_float(stack, 0);
            auto scale = fetch_optional_float(stack, 1);
            auto mean = fetch_optional_float(stack, 0);
            auto dtype = fetch_optional_int(stack, 0);

            auto input = fetch_tensor(stack);


            if ( input->onnx_RandomNormalLike(input, output, dtype, mean, scale, seed) != YNX_OK ) {
                yannx_panic("API: RandomNormalLike  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RandomNormalLike)
    };


    struct Multinomial : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto seed = fetch_optional_float(stack, 0);
            auto sample_size = fetch_optional_int(stack, 1);
            auto dtype = fetch_optional_int(stack, 6);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Multinomial_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("seed", seed);
                infer_.new_attr("sample_size", sample_size);
                infer_.new_attr("dtype", dtype);

                infer_.new_input(input);

                auto f = query_inference_function("Multinomial");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Multinomial(input, output, dtype, sample_size, seed) != YNX_OK ) {
                yannx_panic("API: Multinomial  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto seed = fetch_optional_float(stack, 0);
            auto sample_size = fetch_optional_int(stack, 1);
            auto dtype = fetch_optional_int(stack, 6);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Multinomial(input, output, dtype, sample_size, seed) != YNX_OK ) {
                yannx_panic("API: Multinomial  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Multinomial)
    };

}
namespace logical {

    struct GreaterOrEqual : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_GreaterOrEqual_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("GreaterOrEqual");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_GreaterOrEqual(A, B, C) != YNX_OK ) {
                yannx_panic("API: GreaterOrEqual  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_GreaterOrEqual(A, B, C) != YNX_OK ) {
                yannx_panic("API: GreaterOrEqual  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GreaterOrEqual)
    };


    struct LessOrEqual : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_LessOrEqual_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("LessOrEqual");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_LessOrEqual(A, B, C) != YNX_OK ) {
                yannx_panic("API: LessOrEqual  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_LessOrEqual(A, B, C) != YNX_OK ) {
                yannx_panic("API: LessOrEqual  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LessOrEqual)
    };


    struct Or : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Or_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("Or");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_Or(A, B, C) != YNX_OK ) {
                yannx_panic("API: Or  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Or(A, B, C) != YNX_OK ) {
                yannx_panic("API: Or  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Or)
    };


    struct Not : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Not_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Not");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Not(X, Y) != YNX_OK ) {
                yannx_panic("API: Not  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Not(X, Y) != YNX_OK ) {
                yannx_panic("API: Not  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Not)
    };


    struct And : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_And_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("And");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_And(A, B, C) != YNX_OK ) {
                yannx_panic("API: And  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_And(A, B, C) != YNX_OK ) {
                yannx_panic("API: And  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(And)
    };


    struct Less : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Less_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("Less");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_Less(A, B, C) != YNX_OK ) {
                yannx_panic("API: Less  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Less(A, B, C) != YNX_OK ) {
                yannx_panic("API: Less  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Less)
    };


    struct Equal : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Equal_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("Equal");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_Equal(A, B, C) != YNX_OK ) {
                yannx_panic("API: Equal  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Equal(A, B, C) != YNX_OK ) {
                yannx_panic("API: Equal  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Equal)
    };


    struct BitShift : NativeWord<TensorType> {
        tensor_t Z;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Z = rt.create_undefined_user_tensor();

            auto direction = fetch_string(stack);

            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_BitShift_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("direction", direction);

                infer_.new_input(X);
                infer_.new_input(Y);

                auto f = query_inference_function("BitShift");
                infer_.do_inference(f);
                infer_.check_output(0, Z);


            }

            if ( X->onnx_BitShift(X, Y, Z, direction) != YNX_OK ) {
                yannx_panic("API: BitShift  return error!");
            }

            put_tensor(stack, Z);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto direction = fetch_string(stack);

            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_BitShift(X, Y, Z, direction) != YNX_OK ) {
                yannx_panic("API: BitShift  return error!");
            }

            put_tensor(stack, Z);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(BitShift)
    };


    struct Greater : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Greater_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("Greater");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_Greater(A, B, C) != YNX_OK ) {
                yannx_panic("API: Greater  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Greater(A, B, C) != YNX_OK ) {
                yannx_panic("API: Greater  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Greater)
    };


    struct Xor : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Xor_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("Xor");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_Xor(A, B, C) != YNX_OK ) {
                yannx_panic("API: Xor  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Xor(A, B, C) != YNX_OK ) {
                yannx_panic("API: Xor  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Xor)
    };

}
namespace math {

    struct MelWeightMatrix : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto output_datatype = fetch_optional_int(stack, 1);

            auto upper_edge_hertz = fetch_tensor(stack);
            auto lower_edge_hertz = fetch_tensor(stack);
            auto sample_rate = fetch_tensor(stack);
            auto dft_length = fetch_tensor(stack);
            auto num_mel_bins = fetch_tensor(stack);


            if ( num_mel_bins->onnx_MelWeightMatrix_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("output_datatype", output_datatype);

                infer_.new_input(num_mel_bins);
                infer_.new_input(dft_length);
                infer_.new_input(sample_rate);
                infer_.new_input(lower_edge_hertz);
                infer_.new_input(upper_edge_hertz);

                auto f = query_inference_function("MelWeightMatrix");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( num_mel_bins->onnx_MelWeightMatrix(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz, output, output_datatype) != YNX_OK ) {
                yannx_panic("API: MelWeightMatrix  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto output_datatype = fetch_optional_int(stack, 1);

            auto upper_edge_hertz = fetch_tensor(stack);
            auto lower_edge_hertz = fetch_tensor(stack);
            auto sample_rate = fetch_tensor(stack);
            auto dft_length = fetch_tensor(stack);
            auto num_mel_bins = fetch_tensor(stack);


            if ( num_mel_bins->onnx_MelWeightMatrix(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz, output, output_datatype) != YNX_OK ) {
                yannx_panic("API: MelWeightMatrix  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MelWeightMatrix)
    };


    struct BlackmanWindow : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto periodic = fetch_optional_int(stack, 1);
            auto output_datatype = fetch_optional_int(stack, 1);

            auto size = fetch_tensor(stack);


            if ( size->onnx_BlackmanWindow_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("periodic", periodic);
                infer_.new_attr("output_datatype", output_datatype);

                infer_.new_input(size);

                auto f = query_inference_function("BlackmanWindow");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( size->onnx_BlackmanWindow(size, output, output_datatype, periodic) != YNX_OK ) {
                yannx_panic("API: BlackmanWindow  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto periodic = fetch_optional_int(stack, 1);
            auto output_datatype = fetch_optional_int(stack, 1);

            auto size = fetch_tensor(stack);


            if ( size->onnx_BlackmanWindow(size, output, output_datatype, periodic) != YNX_OK ) {
                yannx_panic("API: BlackmanWindow  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(BlackmanWindow)
    };


    struct HammingWindow : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto periodic = fetch_optional_int(stack, 1);
            auto output_datatype = fetch_optional_int(stack, 1);

            auto size = fetch_tensor(stack);


            if ( size->onnx_HammingWindow_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("periodic", periodic);
                infer_.new_attr("output_datatype", output_datatype);

                infer_.new_input(size);

                auto f = query_inference_function("HammingWindow");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( size->onnx_HammingWindow(size, output, output_datatype, periodic) != YNX_OK ) {
                yannx_panic("API: HammingWindow  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto periodic = fetch_optional_int(stack, 1);
            auto output_datatype = fetch_optional_int(stack, 1);

            auto size = fetch_tensor(stack);


            if ( size->onnx_HammingWindow(size, output, output_datatype, periodic) != YNX_OK ) {
                yannx_panic("API: HammingWindow  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(HammingWindow)
    };


    struct HannWindow : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto periodic = fetch_optional_int(stack, 1);
            auto output_datatype = fetch_optional_int(stack, 1);

            auto size = fetch_tensor(stack);


            if ( size->onnx_HannWindow_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("periodic", periodic);
                infer_.new_attr("output_datatype", output_datatype);

                infer_.new_input(size);

                auto f = query_inference_function("HannWindow");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( size->onnx_HannWindow(size, output, output_datatype, periodic) != YNX_OK ) {
                yannx_panic("API: HannWindow  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto periodic = fetch_optional_int(stack, 1);
            auto output_datatype = fetch_optional_int(stack, 1);

            auto size = fetch_tensor(stack);


            if ( size->onnx_HannWindow(size, output, output_datatype, periodic) != YNX_OK ) {
                yannx_panic("API: HannWindow  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(HannWindow)
    };


    struct SoftmaxCrossEntropyLoss : NativeWord<TensorType> {
        std::variant<void *, tensor_t> log_prob;
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            if ( fetch_bool(stack) == true) {
                log_prob = rt.create_undefined_user_tensor();
            }
            output = rt.create_undefined_user_tensor();

            auto reduction = fetch_optional_string(stack, "mean");
            auto ignore_index = fetch_optional_int(stack, 0);

            auto weights = fetch_optional_tensor(stack);
            auto labels = fetch_tensor(stack);
            auto scores = fetch_tensor(stack);


            if ( scores->onnx_SoftmaxCrossEntropyLoss_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                if ( log_prob.index() != 0) {
                    outputs_.push_back(1);
                }
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("reduction", reduction);
                infer_.new_attr("ignore_index", ignore_index);

                infer_.new_input(scores);
                infer_.new_input(labels);
                infer_.new_input(weights);

                auto f = query_inference_function("SoftmaxCrossEntropyLoss");
                infer_.do_inference(f);
                infer_.check_output(0, output);
                if ( log_prob.index() != 0) {
                    infer_.check_output(1, std::get<1>(log_prob));
                }


            }

            if ( scores->onnx_SoftmaxCrossEntropyLoss(scores, labels, weights, output, log_prob, ignore_index, reduction) != YNX_OK ) {
                yannx_panic("API: SoftmaxCrossEntropyLoss  return error!");
            }

            put_tensor(stack, output);
            if ( log_prob.index() != 0) {
                put_optional_tensor(stack, log_prob);
            }

        }
        virtual void run(ValueStack<TensorType>& stack) {

            fetch_bool(stack);

            auto reduction = fetch_optional_string(stack, "mean");
            auto ignore_index = fetch_optional_int(stack, 0);

            auto weights = fetch_optional_tensor(stack);
            auto labels = fetch_tensor(stack);
            auto scores = fetch_tensor(stack);


            if ( scores->onnx_SoftmaxCrossEntropyLoss(scores, labels, weights, output, log_prob, ignore_index, reduction) != YNX_OK ) {
                yannx_panic("API: SoftmaxCrossEntropyLoss  return error!");
            }

            put_tensor(stack, output);
            if ( log_prob.index() != 0) {
                put_optional_tensor(stack, log_prob);
            }

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SoftmaxCrossEntropyLoss)
    };


    struct Celu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto alpha = fetch_optional_float(stack, 1);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Celu_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("alpha", alpha);

                infer_.new_input(X);

                auto f = query_inference_function("Celu");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Celu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: Celu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto alpha = fetch_optional_float(stack, 1);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Celu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: Celu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Celu)
    };


    struct CumSum : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            y = rt.create_undefined_user_tensor();

            auto reverse = fetch_optional_int(stack, 0);
            auto exclusive = fetch_optional_int(stack, 0);

            auto axis = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_CumSum_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("reverse", reverse);
                infer_.new_attr("exclusive", exclusive);

                infer_.new_input(x);
                infer_.new_input(axis);

                auto f = query_inference_function("CumSum");
                infer_.do_inference(f);
                infer_.check_output(0, y);


            }

            if ( x->onnx_CumSum(x, axis, y, exclusive, reverse) != YNX_OK ) {
                yannx_panic("API: CumSum  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto reverse = fetch_optional_int(stack, 0);
            auto exclusive = fetch_optional_int(stack, 0);

            auto axis = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_CumSum(x, axis, y, exclusive, reverse) != YNX_OK ) {
                yannx_panic("API: CumSum  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(CumSum)
    };


    struct DFT : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto onesided = fetch_optional_int(stack, 0);
            auto inverse = fetch_optional_int(stack, 0);
            auto axis = fetch_optional_int(stack, 1);

            auto dft_length = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_DFT_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("onesided", onesided);
                infer_.new_attr("inverse", inverse);
                infer_.new_attr("axis", axis);

                infer_.new_input(input);
                infer_.new_input(dft_length);

                auto f = query_inference_function("DFT");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_DFT(input, dft_length, output, axis, inverse, onesided) != YNX_OK ) {
                yannx_panic("API: DFT  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto onesided = fetch_optional_int(stack, 0);
            auto inverse = fetch_optional_int(stack, 0);
            auto axis = fetch_optional_int(stack, 1);

            auto dft_length = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_DFT(input, dft_length, output, axis, inverse, onesided) != YNX_OK ) {
                yannx_panic("API: DFT  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(DFT)
    };


    struct Det : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Det_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Det");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Det(X, Y) != YNX_OK ) {
                yannx_panic("API: Det  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Det(X, Y) != YNX_OK ) {
                yannx_panic("API: Det  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Det)
    };


    struct QLinearMatMul : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            y = rt.create_undefined_user_tensor();


            auto y_zero_point = fetch_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto b_zero_point = fetch_tensor(stack);
            auto b_scale = fetch_tensor(stack);
            auto b = fetch_tensor(stack);
            auto a_zero_point = fetch_tensor(stack);
            auto a_scale = fetch_tensor(stack);
            auto a = fetch_tensor(stack);


            if ( a->onnx_QLinearMatMul_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(a);
                infer_.new_input(a_scale);
                infer_.new_input(a_zero_point);
                infer_.new_input(b);
                infer_.new_input(b_scale);
                infer_.new_input(b_zero_point);
                infer_.new_input(y_scale);
                infer_.new_input(y_zero_point);

                auto f = query_inference_function("QLinearMatMul");
                infer_.do_inference(f);
                infer_.check_output(0, y);


            }

            if ( a->onnx_QLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, y) != YNX_OK ) {
                yannx_panic("API: QLinearMatMul  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto y_zero_point = fetch_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto b_zero_point = fetch_tensor(stack);
            auto b_scale = fetch_tensor(stack);
            auto b = fetch_tensor(stack);
            auto a_zero_point = fetch_tensor(stack);
            auto a_scale = fetch_tensor(stack);
            auto a = fetch_tensor(stack);


            if ( a->onnx_QLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, y) != YNX_OK ) {
                yannx_panic("API: QLinearMatMul  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(QLinearMatMul)
    };


    struct MatMulInteger : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto b_zero_point = fetch_optional_tensor(stack);
            auto a_zero_point = fetch_optional_tensor(stack);
            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_MatMulInteger_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);
                infer_.new_input(a_zero_point);
                infer_.new_input(b_zero_point);

                auto f = query_inference_function("MatMulInteger");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( A->onnx_MatMulInteger(A, B, a_zero_point, b_zero_point, Y) != YNX_OK ) {
                yannx_panic("API: MatMulInteger  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto b_zero_point = fetch_optional_tensor(stack);
            auto a_zero_point = fetch_optional_tensor(stack);
            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_MatMulInteger(A, B, a_zero_point, b_zero_point, Y) != YNX_OK ) {
                yannx_panic("API: MatMulInteger  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MatMulInteger)
    };


    struct Mul : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Mul_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("Mul");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_Mul(A, B, C) != YNX_OK ) {
                yannx_panic("API: Mul  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Mul(A, B, C) != YNX_OK ) {
                yannx_panic("API: Mul  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Mul)
    };


    struct Max : NativeWord<TensorType> {
        tensor_t max;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            max = rt.create_undefined_user_tensor();


            auto data_0 = fetch_tensors(stack);


            if ( max->onnx_Max_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(data_0);

                auto f = query_inference_function("Max");
                infer_.do_inference(f);
                infer_.check_output(0, max);


            }

            if ( max->onnx_Max(data_0, max) != YNX_OK ) {
                yannx_panic("API: Max  return error!");
            }

            put_tensor(stack, max);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto data_0 = fetch_tensors(stack);


            if ( max->onnx_Max(data_0, max) != YNX_OK ) {
                yannx_panic("API: Max  return error!");
            }

            put_tensor(stack, max);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Max)
    };


    struct Mod : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();

            auto fmod = fetch_optional_int(stack, 0);

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Mod_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("fmod", fmod);

                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("Mod");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_Mod(A, B, C, fmod) != YNX_OK ) {
                yannx_panic("API: Mod  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto fmod = fetch_optional_int(stack, 0);

            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Mod(A, B, C, fmod) != YNX_OK ) {
                yannx_panic("API: Mod  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Mod)
    };


    struct Log : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Log_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Log");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Log(input, output) != YNX_OK ) {
                yannx_panic("API: Log  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Log(input, output) != YNX_OK ) {
                yannx_panic("API: Log  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Log)
    };


    struct Sign : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Sign_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Sign");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Sign(input, output) != YNX_OK ) {
                yannx_panic("API: Sign  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Sign(input, output) != YNX_OK ) {
                yannx_panic("API: Sign  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sign)
    };


    struct Min : NativeWord<TensorType> {
        tensor_t min;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            min = rt.create_undefined_user_tensor();


            auto data_0 = fetch_tensors(stack);


            if ( min->onnx_Min_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(data_0);

                auto f = query_inference_function("Min");
                infer_.do_inference(f);
                infer_.check_output(0, min);


            }

            if ( min->onnx_Min(data_0, min) != YNX_OK ) {
                yannx_panic("API: Min  return error!");
            }

            put_tensor(stack, min);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto data_0 = fetch_tensors(stack);


            if ( min->onnx_Min(data_0, min) != YNX_OK ) {
                yannx_panic("API: Min  return error!");
            }

            put_tensor(stack, min);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Min)
    };


    struct PRelu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto slope = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_PRelu_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);
                infer_.new_input(slope);

                auto f = query_inference_function("PRelu");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_PRelu(X, slope, Y) != YNX_OK ) {
                yannx_panic("API: PRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto slope = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_PRelu(X, slope, Y) != YNX_OK ) {
                yannx_panic("API: PRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(PRelu)
    };


    struct Ceil : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Ceil_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Ceil");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Ceil(X, Y) != YNX_OK ) {
                yannx_panic("API: Ceil  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Ceil(X, Y) != YNX_OK ) {
                yannx_panic("API: Ceil  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Ceil)
    };


    struct Tan : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Tan_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Tan");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Tan(input, output) != YNX_OK ) {
                yannx_panic("API: Tan  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Tan(input, output) != YNX_OK ) {
                yannx_panic("API: Tan  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Tan)
    };


    struct Clip : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto max = fetch_optional_tensor(stack);
            auto min = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Clip_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);
                infer_.new_input(min);
                infer_.new_input(max);

                auto f = query_inference_function("Clip");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Clip(input, min, max, output) != YNX_OK ) {
                yannx_panic("API: Clip  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto max = fetch_optional_tensor(stack);
            auto min = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Clip(input, min, max, output) != YNX_OK ) {
                yannx_panic("API: Clip  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Clip)
    };


    struct Neg : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Neg_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Neg");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Neg(X, Y) != YNX_OK ) {
                yannx_panic("API: Neg  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Neg(X, Y) != YNX_OK ) {
                yannx_panic("API: Neg  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Neg)
    };


    struct Abs : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Abs_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Abs");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Abs(X, Y) != YNX_OK ) {
                yannx_panic("API: Abs  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Abs(X, Y) != YNX_OK ) {
                yannx_panic("API: Abs  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Abs)
    };


    struct Softplus : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Softplus_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Softplus");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Softplus(X, Y) != YNX_OK ) {
                yannx_panic("API: Softplus  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Softplus(X, Y) != YNX_OK ) {
                yannx_panic("API: Softplus  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Softplus)
    };


    struct Einsum : NativeWord<TensorType> {
        tensor_t Output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Output = rt.create_undefined_user_tensor();

            auto equation = fetch_string(stack);

            auto Inputs = fetch_tensors(stack);


            if ( Output->onnx_Einsum_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("equation", equation);

                infer_.new_input(Inputs);

                auto f = query_inference_function("Einsum");
                infer_.do_inference(f);
                infer_.check_output(0, Output);


            }

            if ( Output->onnx_Einsum(Inputs, Output, equation) != YNX_OK ) {
                yannx_panic("API: Einsum  return error!");
            }

            put_tensor(stack, Output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto equation = fetch_string(stack);

            auto Inputs = fetch_tensors(stack);


            if ( Output->onnx_Einsum(Inputs, Output, equation) != YNX_OK ) {
                yannx_panic("API: Einsum  return error!");
            }

            put_tensor(stack, Output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Einsum)
    };


    struct Sub : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Sub_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("Sub");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_Sub(A, B, C) != YNX_OK ) {
                yannx_panic("API: Sub  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Sub(A, B, C) != YNX_OK ) {
                yannx_panic("API: Sub  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sub)
    };


    struct Floor : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Floor_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Floor");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Floor(X, Y) != YNX_OK ) {
                yannx_panic("API: Floor  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Floor(X, Y) != YNX_OK ) {
                yannx_panic("API: Floor  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Floor)
    };


    struct Sigmoid : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Sigmoid_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Sigmoid");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Sigmoid(X, Y) != YNX_OK ) {
                yannx_panic("API: Sigmoid  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Sigmoid(X, Y) != YNX_OK ) {
                yannx_panic("API: Sigmoid  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sigmoid)
    };


    struct Softmax : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto axis = fetch_optional_int(stack, -1);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Softmax_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(input);

                auto f = query_inference_function("Softmax");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Softmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Softmax  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, -1);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Softmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Softmax  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Softmax)
    };


    struct Add : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Add_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("Add");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_Add(A, B, C) != YNX_OK ) {
                yannx_panic("API: Add  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Add(A, B, C) != YNX_OK ) {
                yannx_panic("API: Add  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Add)
    };


    struct Round : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Round_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Round");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Round(X, Y) != YNX_OK ) {
                yannx_panic("API: Round  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Round(X, Y) != YNX_OK ) {
                yannx_panic("API: Round  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Round)
    };


    struct Exp : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Exp_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Exp");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Exp(input, output) != YNX_OK ) {
                yannx_panic("API: Exp  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Exp(input, output) != YNX_OK ) {
                yannx_panic("API: Exp  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Exp)
    };


    struct MatMul : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_MatMul_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("MatMul");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( A->onnx_MatMul(A, B, Y) != YNX_OK ) {
                yannx_panic("API: MatMul  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_MatMul(A, B, Y) != YNX_OK ) {
                yannx_panic("API: MatMul  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MatMul)
    };


    struct LeakyRelu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto alpha = fetch_optional_float(stack, 0.01);

            auto X = fetch_tensor(stack);


            if ( X->onnx_LeakyRelu_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("alpha", alpha);

                infer_.new_input(X);

                auto f = query_inference_function("LeakyRelu");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_LeakyRelu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: LeakyRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto alpha = fetch_optional_float(stack, 0.01);

            auto X = fetch_tensor(stack);


            if ( X->onnx_LeakyRelu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: LeakyRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LeakyRelu)
    };


    struct Gemm : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto transB = fetch_optional_int(stack, 0);
            auto transA = fetch_optional_int(stack, 0);
            auto beta = fetch_optional_float(stack, 1);
            auto alpha = fetch_optional_float(stack, 1);

            auto C = fetch_optional_tensor(stack);
            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Gemm_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("transB", transB);
                infer_.new_attr("transA", transA);
                infer_.new_attr("beta", beta);
                infer_.new_attr("alpha", alpha);

                infer_.new_input(A);
                infer_.new_input(B);
                infer_.new_input(C);

                auto f = query_inference_function("Gemm");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( A->onnx_Gemm(A, B, C, Y, alpha, beta, transA, transB) != YNX_OK ) {
                yannx_panic("API: Gemm  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto transB = fetch_optional_int(stack, 0);
            auto transA = fetch_optional_int(stack, 0);
            auto beta = fetch_optional_float(stack, 1);
            auto alpha = fetch_optional_float(stack, 1);

            auto C = fetch_optional_tensor(stack);
            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Gemm(A, B, C, Y, alpha, beta, transA, transB) != YNX_OK ) {
                yannx_panic("API: Gemm  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Gemm)
    };


    struct NegativeLogLikelihoodLoss : NativeWord<TensorType> {
        tensor_t loss;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            loss = rt.create_undefined_user_tensor();

            auto reduction = fetch_optional_string(stack, "mean");
            auto ignore_index = fetch_optional_int(stack, 0);

            auto weight = fetch_optional_tensor(stack);
            auto target = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_NegativeLogLikelihoodLoss_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("reduction", reduction);
                infer_.new_attr("ignore_index", ignore_index);

                infer_.new_input(input);
                infer_.new_input(target);
                infer_.new_input(weight);

                auto f = query_inference_function("NegativeLogLikelihoodLoss");
                infer_.do_inference(f);
                infer_.check_output(0, loss);


            }

            if ( input->onnx_NegativeLogLikelihoodLoss(input, target, weight, loss, ignore_index, reduction) != YNX_OK ) {
                yannx_panic("API: NegativeLogLikelihoodLoss  return error!");
            }

            put_tensor(stack, loss);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto reduction = fetch_optional_string(stack, "mean");
            auto ignore_index = fetch_optional_int(stack, 0);

            auto weight = fetch_optional_tensor(stack);
            auto target = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_NegativeLogLikelihoodLoss(input, target, weight, loss, ignore_index, reduction) != YNX_OK ) {
                yannx_panic("API: NegativeLogLikelihoodLoss  return error!");
            }

            put_tensor(stack, loss);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(NegativeLogLikelihoodLoss)
    };


    struct HardSwish : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_HardSwish_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("HardSwish");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_HardSwish(X, Y) != YNX_OK ) {
                yannx_panic("API: HardSwish  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_HardSwish(X, Y) != YNX_OK ) {
                yannx_panic("API: HardSwish  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(HardSwish)
    };


    struct Mean : NativeWord<TensorType> {
        tensor_t mean;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            mean = rt.create_undefined_user_tensor();


            auto data_0 = fetch_tensors(stack);


            if ( mean->onnx_Mean_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(data_0);

                auto f = query_inference_function("Mean");
                infer_.do_inference(f);
                infer_.check_output(0, mean);


            }

            if ( mean->onnx_Mean(data_0, mean) != YNX_OK ) {
                yannx_panic("API: Mean  return error!");
            }

            put_tensor(stack, mean);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto data_0 = fetch_tensors(stack);


            if ( mean->onnx_Mean(data_0, mean) != YNX_OK ) {
                yannx_panic("API: Mean  return error!");
            }

            put_tensor(stack, mean);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Mean)
    };


    struct Asin : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Asin_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Asin");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Asin(input, output) != YNX_OK ) {
                yannx_panic("API: Asin  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Asin(input, output) != YNX_OK ) {
                yannx_panic("API: Asin  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Asin)
    };


    struct Div : NativeWord<TensorType> {
        tensor_t C;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            C = rt.create_undefined_user_tensor();


            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Div_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(A);
                infer_.new_input(B);

                auto f = query_inference_function("Div");
                infer_.do_inference(f);
                infer_.check_output(0, C);


            }

            if ( A->onnx_Div(A, B, C) != YNX_OK ) {
                yannx_panic("API: Div  return error!");
            }

            put_tensor(stack, C);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto B = fetch_tensor(stack);
            auto A = fetch_tensor(stack);


            if ( A->onnx_Div(A, B, C) != YNX_OK ) {
                yannx_panic("API: Div  return error!");
            }

            put_tensor(stack, C);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Div)
    };


    struct Softsign : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Softsign_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Softsign");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Softsign(input, output) != YNX_OK ) {
                yannx_panic("API: Softsign  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Softsign(input, output) != YNX_OK ) {
                yannx_panic("API: Softsign  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Softsign)
    };


    struct Reciprocal : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Reciprocal_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Reciprocal");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Reciprocal(X, Y) != YNX_OK ) {
                yannx_panic("API: Reciprocal  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Reciprocal(X, Y) != YNX_OK ) {
                yannx_panic("API: Reciprocal  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Reciprocal)
    };


    struct Relu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Relu_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Relu");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Relu(X, Y) != YNX_OK ) {
                yannx_panic("API: Relu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Relu(X, Y) != YNX_OK ) {
                yannx_panic("API: Relu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Relu)
    };


    struct Elu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto alpha = fetch_optional_float(stack, 1);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Elu_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("alpha", alpha);

                infer_.new_input(X);

                auto f = query_inference_function("Elu");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Elu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: Elu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto alpha = fetch_optional_float(stack, 1);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Elu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: Elu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Elu)
    };


    struct Selu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto gamma = fetch_optional_float(stack, 1.0507);
            auto alpha = fetch_optional_float(stack, 1.67326);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Selu_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("gamma", gamma);
                infer_.new_attr("alpha", alpha);

                infer_.new_input(X);

                auto f = query_inference_function("Selu");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Selu(X, Y, alpha, gamma) != YNX_OK ) {
                yannx_panic("API: Selu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto gamma = fetch_optional_float(stack, 1.0507);
            auto alpha = fetch_optional_float(stack, 1.67326);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Selu(X, Y, alpha, gamma) != YNX_OK ) {
                yannx_panic("API: Selu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Selu)
    };


    struct HardSigmoid : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto beta = fetch_optional_float(stack, 0.5);
            auto alpha = fetch_optional_float(stack, 0.2);

            auto X = fetch_tensor(stack);


            if ( X->onnx_HardSigmoid_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("beta", beta);
                infer_.new_attr("alpha", alpha);

                infer_.new_input(X);

                auto f = query_inference_function("HardSigmoid");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_HardSigmoid(X, Y, alpha, beta) != YNX_OK ) {
                yannx_panic("API: HardSigmoid  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto beta = fetch_optional_float(stack, 0.5);
            auto alpha = fetch_optional_float(stack, 0.2);

            auto X = fetch_tensor(stack);


            if ( X->onnx_HardSigmoid(X, Y, alpha, beta) != YNX_OK ) {
                yannx_panic("API: HardSigmoid  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(HardSigmoid)
    };


    struct LogSoftmax : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto axis = fetch_optional_int(stack, -1);

            auto input = fetch_tensor(stack);


            if ( input->onnx_LogSoftmax_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(input);

                auto f = query_inference_function("LogSoftmax");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_LogSoftmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: LogSoftmax  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, -1);

            auto input = fetch_tensor(stack);


            if ( input->onnx_LogSoftmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: LogSoftmax  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LogSoftmax)
    };


    struct Sqrt : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_Sqrt_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("Sqrt");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Sqrt(X, Y) != YNX_OK ) {
                yannx_panic("API: Sqrt  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_Sqrt(X, Y) != YNX_OK ) {
                yannx_panic("API: Sqrt  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sqrt)
    };


    struct Expand : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto shape = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Expand_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);
                infer_.new_input(shape);

                auto f = query_inference_function("Expand");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Expand(input, shape, output) != YNX_OK ) {
                yannx_panic("API: Expand  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto shape = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Expand(input, shape, output) != YNX_OK ) {
                yannx_panic("API: Expand  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Expand)
    };


    struct STFT : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto onesided = fetch_optional_int(stack, 1);

            auto frame_length = fetch_optional_tensor(stack);
            auto window = fetch_optional_tensor(stack);
            auto frame_step = fetch_tensor(stack);
            auto signal = fetch_tensor(stack);


            if ( signal->onnx_STFT_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("onesided", onesided);

                infer_.new_input(signal);
                infer_.new_input(frame_step);
                infer_.new_input(window);
                infer_.new_input(frame_length);

                auto f = query_inference_function("STFT");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( signal->onnx_STFT(signal, frame_step, window, frame_length, output, onesided) != YNX_OK ) {
                yannx_panic("API: STFT  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto onesided = fetch_optional_int(stack, 1);

            auto frame_length = fetch_optional_tensor(stack);
            auto window = fetch_optional_tensor(stack);
            auto frame_step = fetch_tensor(stack);
            auto signal = fetch_tensor(stack);


            if ( signal->onnx_STFT(signal, frame_step, window, frame_length, output, onesided) != YNX_OK ) {
                yannx_panic("API: STFT  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(STFT)
    };


    struct Sum : NativeWord<TensorType> {
        tensor_t sum;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            sum = rt.create_undefined_user_tensor();


            auto data_0 = fetch_tensors(stack);


            if ( sum->onnx_Sum_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(data_0);

                auto f = query_inference_function("Sum");
                infer_.do_inference(f);
                infer_.check_output(0, sum);


            }

            if ( sum->onnx_Sum(data_0, sum) != YNX_OK ) {
                yannx_panic("API: Sum  return error!");
            }

            put_tensor(stack, sum);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto data_0 = fetch_tensors(stack);


            if ( sum->onnx_Sum(data_0, sum) != YNX_OK ) {
                yannx_panic("API: Sum  return error!");
            }

            put_tensor(stack, sum);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sum)
    };


    struct Tanh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Tanh_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Tanh");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Tanh(input, output) != YNX_OK ) {
                yannx_panic("API: Tanh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Tanh(input, output) != YNX_OK ) {
                yannx_panic("API: Tanh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Tanh)
    };


    struct TopK : NativeWord<TensorType> {
        tensor_t Indices;
        tensor_t Values;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Indices = rt.create_undefined_user_tensor();
            Values = rt.create_undefined_user_tensor();

            auto sorted = fetch_optional_int(stack, 1);
            auto largest = fetch_optional_int(stack, 1);
            auto axis = fetch_optional_int(stack, -1);

            auto K = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_TopK_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                outputs_.push_back(1);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("sorted", sorted);
                infer_.new_attr("largest", largest);
                infer_.new_attr("axis", axis);

                infer_.new_input(X);
                infer_.new_input(K);

                auto f = query_inference_function("TopK");
                infer_.do_inference(f);
                infer_.check_output(0, Values);
                infer_.check_output(1, Indices);


            }

            if ( X->onnx_TopK(X, K, Values, Indices, axis, largest, sorted) != YNX_OK ) {
                yannx_panic("API: TopK  return error!");
            }

            put_tensor(stack, Values);
            put_tensor(stack, Indices);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto sorted = fetch_optional_int(stack, 1);
            auto largest = fetch_optional_int(stack, 1);
            auto axis = fetch_optional_int(stack, -1);

            auto K = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_TopK(X, K, Values, Indices, axis, largest, sorted) != YNX_OK ) {
                yannx_panic("API: TopK  return error!");
            }

            put_tensor(stack, Values);
            put_tensor(stack, Indices);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(TopK)
    };


    struct ThresholdedRelu : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto alpha = fetch_optional_float(stack, 1);

            auto X = fetch_tensor(stack);


            if ( X->onnx_ThresholdedRelu_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("alpha", alpha);

                infer_.new_input(X);

                auto f = query_inference_function("ThresholdedRelu");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_ThresholdedRelu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: ThresholdedRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto alpha = fetch_optional_float(stack, 1);

            auto X = fetch_tensor(stack);


            if ( X->onnx_ThresholdedRelu(X, Y, alpha) != YNX_OK ) {
                yannx_panic("API: ThresholdedRelu  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ThresholdedRelu)
    };


    struct Acos : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Acos_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Acos");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Acos(input, output) != YNX_OK ) {
                yannx_panic("API: Acos  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Acos(input, output) != YNX_OK ) {
                yannx_panic("API: Acos  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Acos)
    };


    struct Pow : NativeWord<TensorType> {
        tensor_t Z;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Z = rt.create_undefined_user_tensor();


            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_Pow_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);
                infer_.new_input(Y);

                auto f = query_inference_function("Pow");
                infer_.do_inference(f);
                infer_.check_output(0, Z);


            }

            if ( X->onnx_Pow(X, Y, Z) != YNX_OK ) {
                yannx_panic("API: Pow  return error!");
            }

            put_tensor(stack, Z);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_Pow(X, Y, Z) != YNX_OK ) {
                yannx_panic("API: Pow  return error!");
            }

            put_tensor(stack, Z);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Pow)
    };


    struct Atan : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Atan_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Atan");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Atan(input, output) != YNX_OK ) {
                yannx_panic("API: Atan  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Atan(input, output) != YNX_OK ) {
                yannx_panic("API: Atan  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Atan)
    };


    struct Cos : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Cos_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Cos");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Cos(input, output) != YNX_OK ) {
                yannx_panic("API: Cos  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Cos(input, output) != YNX_OK ) {
                yannx_panic("API: Cos  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Cos)
    };


    struct Sin : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Sin_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Sin");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Sin(input, output) != YNX_OK ) {
                yannx_panic("API: Sin  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Sin(input, output) != YNX_OK ) {
                yannx_panic("API: Sin  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sin)
    };


    struct Sinh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Sinh_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Sinh");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Sinh(input, output) != YNX_OK ) {
                yannx_panic("API: Sinh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Sinh(input, output) != YNX_OK ) {
                yannx_panic("API: Sinh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Sinh)
    };


    struct Asinh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Asinh_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Asinh");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Asinh(input, output) != YNX_OK ) {
                yannx_panic("API: Asinh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Asinh(input, output) != YNX_OK ) {
                yannx_panic("API: Asinh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Asinh)
    };


    struct Acosh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Acosh_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Acosh");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Acosh(input, output) != YNX_OK ) {
                yannx_panic("API: Acosh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Acosh(input, output) != YNX_OK ) {
                yannx_panic("API: Acosh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Acosh)
    };


    struct Cosh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Cosh_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Cosh");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Cosh(input, output) != YNX_OK ) {
                yannx_panic("API: Cosh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Cosh(input, output) != YNX_OK ) {
                yannx_panic("API: Cosh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Cosh)
    };


    struct Atanh : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Atanh_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Atanh");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Atanh(input, output) != YNX_OK ) {
                yannx_panic("API: Atanh  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Atanh(input, output) != YNX_OK ) {
                yannx_panic("API: Atanh  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Atanh)
    };


    struct Hardmax : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto axis = fetch_optional_int(stack, -1);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Hardmax_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(input);

                auto f = query_inference_function("Hardmax");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Hardmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Hardmax  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, -1);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Hardmax(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Hardmax  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Hardmax)
    };


    struct Erf : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Erf_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Erf");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Erf(input, output) != YNX_OK ) {
                yannx_panic("API: Erf  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Erf(input, output) != YNX_OK ) {
                yannx_panic("API: Erf  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Erf)
    };

}
namespace nn {

    struct LayerNormalization : NativeWord<TensorType> {
        std::variant<void *, tensor_t> InvStdDev;
        std::variant<void *, tensor_t> Mean;
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            if ( fetch_bool(stack) == true) {
                InvStdDev = rt.create_undefined_user_tensor();
            }
            if ( fetch_bool(stack) == true) {
                Mean = rt.create_undefined_user_tensor();
            }
            Y = rt.create_undefined_user_tensor();

            auto stash_type = fetch_optional_int(stack, 1);
            auto epsilon = fetch_optional_float(stack, 1e-05);
            auto axis = fetch_optional_int(stack, -1);

            auto B = fetch_optional_tensor(stack);
            auto Scale = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_LayerNormalization_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                if ( Mean.index() != 0) {
                    outputs_.push_back(1);
                }
                if ( InvStdDev.index() != 0) {
                    outputs_.push_back(2);
                }
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("stash_type", stash_type);
                infer_.new_attr("epsilon", epsilon);
                infer_.new_attr("axis", axis);

                infer_.new_input(X);
                infer_.new_input(Scale);
                infer_.new_input(B);

                auto f = query_inference_function("LayerNormalization");
                infer_.do_inference(f);
                infer_.check_output(0, Y);
                if ( Mean.index() != 0) {
                    infer_.check_output(1, std::get<1>(Mean));
                }
                if ( InvStdDev.index() != 0) {
                    infer_.check_output(2, std::get<1>(InvStdDev));
                }


            }

            if ( X->onnx_LayerNormalization(X, Scale, B, Y, Mean, InvStdDev, axis, epsilon, stash_type) != YNX_OK ) {
                yannx_panic("API: LayerNormalization  return error!");
            }

            put_tensor(stack, Y);
            if ( Mean.index() != 0) {
                put_optional_tensor(stack, Mean);
            }
            if ( InvStdDev.index() != 0) {
                put_optional_tensor(stack, InvStdDev);
            }

        }
        virtual void run(ValueStack<TensorType>& stack) {

            fetch_bool(stack);
            fetch_bool(stack);

            auto stash_type = fetch_optional_int(stack, 1);
            auto epsilon = fetch_optional_float(stack, 1e-05);
            auto axis = fetch_optional_int(stack, -1);

            auto B = fetch_optional_tensor(stack);
            auto Scale = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_LayerNormalization(X, Scale, B, Y, Mean, InvStdDev, axis, epsilon, stash_type) != YNX_OK ) {
                yannx_panic("API: LayerNormalization  return error!");
            }

            put_tensor(stack, Y);
            if ( Mean.index() != 0) {
                put_optional_tensor(stack, Mean);
            }
            if ( InvStdDev.index() != 0) {
                put_optional_tensor(stack, InvStdDev);
            }

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LayerNormalization)
    };


    struct QLinearConv : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            y = rt.create_undefined_user_tensor();

            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_optional_ints(stack, {});
            auto group = fetch_optional_int(stack, 1);
            auto dilations = fetch_optional_ints(stack, {});
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto B = fetch_optional_tensor(stack);
            auto y_zero_point = fetch_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto w_zero_point = fetch_tensor(stack);
            auto w_scale = fetch_tensor(stack);
            auto w = fetch_tensor(stack);
            auto x_zero_point = fetch_tensor(stack);
            auto x_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_QLinearConv_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("strides", strides);
                infer_.new_attr("pads", pads);
                infer_.new_attr("kernel_shape", kernel_shape);
                infer_.new_attr("group", group);
                infer_.new_attr("dilations", dilations);
                infer_.new_attr("auto_pad", auto_pad);

                infer_.new_input(x);
                infer_.new_input(x_scale);
                infer_.new_input(x_zero_point);
                infer_.new_input(w);
                infer_.new_input(w_scale);
                infer_.new_input(w_zero_point);
                infer_.new_input(y_scale);
                infer_.new_input(y_zero_point);
                infer_.new_input(B);

                auto f = query_inference_function("QLinearConv");
                infer_.do_inference(f);
                infer_.check_output(0, y);


            }

            if ( x->onnx_QLinearConv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: QLinearConv  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_optional_ints(stack, {});
            auto group = fetch_optional_int(stack, 1);
            auto dilations = fetch_optional_ints(stack, {});
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto B = fetch_optional_tensor(stack);
            auto y_zero_point = fetch_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto w_zero_point = fetch_tensor(stack);
            auto w_scale = fetch_tensor(stack);
            auto w = fetch_tensor(stack);
            auto x_zero_point = fetch_tensor(stack);
            auto x_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_QLinearConv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: QLinearConv  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(QLinearConv)
    };


    struct ConvInteger : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            y = rt.create_undefined_user_tensor();

            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_optional_ints(stack, {});
            auto group = fetch_optional_int(stack, 1);
            auto dilations = fetch_optional_ints(stack, {});
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto w_zero_point = fetch_optional_tensor(stack);
            auto x_zero_point = fetch_optional_tensor(stack);
            auto w = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_ConvInteger_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("strides", strides);
                infer_.new_attr("pads", pads);
                infer_.new_attr("kernel_shape", kernel_shape);
                infer_.new_attr("group", group);
                infer_.new_attr("dilations", dilations);
                infer_.new_attr("auto_pad", auto_pad);

                infer_.new_input(x);
                infer_.new_input(w);
                infer_.new_input(x_zero_point);
                infer_.new_input(w_zero_point);

                auto f = query_inference_function("ConvInteger");
                infer_.do_inference(f);
                infer_.check_output(0, y);


            }

            if ( x->onnx_ConvInteger(x, w, x_zero_point, w_zero_point, y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: ConvInteger  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_optional_ints(stack, {});
            auto group = fetch_optional_int(stack, 1);
            auto dilations = fetch_optional_ints(stack, {});
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto w_zero_point = fetch_optional_tensor(stack);
            auto x_zero_point = fetch_optional_tensor(stack);
            auto w = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_ConvInteger(x, w, x_zero_point, w_zero_point, y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: ConvInteger  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ConvInteger)
    };


    struct MaxPool : NativeWord<TensorType> {
        std::variant<void *, tensor_t> Indices;
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            if ( fetch_bool(stack) == true) {
                Indices = rt.create_undefined_user_tensor();
            }
            Y = rt.create_undefined_user_tensor();

            auto strides = fetch_optional_ints(stack, {});
            auto storage_order = fetch_optional_int(stack, 0);
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_ints(stack);
            auto dilations = fetch_optional_ints(stack, {});
            auto ceil_mode = fetch_optional_int(stack, 0);
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto X = fetch_tensor(stack);


            if ( X->onnx_MaxPool_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                if ( Indices.index() != 0) {
                    outputs_.push_back(1);
                }
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("strides", strides);
                infer_.new_attr("storage_order", storage_order);
                infer_.new_attr("pads", pads);
                infer_.new_attr("kernel_shape", kernel_shape);
                infer_.new_attr("dilations", dilations);
                infer_.new_attr("ceil_mode", ceil_mode);
                infer_.new_attr("auto_pad", auto_pad);

                infer_.new_input(X);

                auto f = query_inference_function("MaxPool");
                infer_.do_inference(f);
                infer_.check_output(0, Y);
                if ( Indices.index() != 0) {
                    infer_.check_output(1, std::get<1>(Indices));
                }


            }

            if ( X->onnx_MaxPool(X, Y, Indices, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides) != YNX_OK ) {
                yannx_panic("API: MaxPool  return error!");
            }

            put_tensor(stack, Y);
            if ( Indices.index() != 0) {
                put_optional_tensor(stack, Indices);
            }

        }
        virtual void run(ValueStack<TensorType>& stack) {

            fetch_bool(stack);

            auto strides = fetch_optional_ints(stack, {});
            auto storage_order = fetch_optional_int(stack, 0);
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_ints(stack);
            auto dilations = fetch_optional_ints(stack, {});
            auto ceil_mode = fetch_optional_int(stack, 0);
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto X = fetch_tensor(stack);


            if ( X->onnx_MaxPool(X, Y, Indices, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides) != YNX_OK ) {
                yannx_panic("API: MaxPool  return error!");
            }

            put_tensor(stack, Y);
            if ( Indices.index() != 0) {
                put_optional_tensor(stack, Indices);
            }

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MaxPool)
    };


    struct Conv : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_optional_ints(stack, {});
            auto group = fetch_optional_int(stack, 1);
            auto dilations = fetch_optional_ints(stack, {});
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto B = fetch_optional_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_Conv_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("strides", strides);
                infer_.new_attr("pads", pads);
                infer_.new_attr("kernel_shape", kernel_shape);
                infer_.new_attr("group", group);
                infer_.new_attr("dilations", dilations);
                infer_.new_attr("auto_pad", auto_pad);

                infer_.new_input(X);
                infer_.new_input(W);
                infer_.new_input(B);

                auto f = query_inference_function("Conv");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Conv(X, W, B, Y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: Conv  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_optional_ints(stack, {});
            auto group = fetch_optional_int(stack, 1);
            auto dilations = fetch_optional_ints(stack, {});
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto B = fetch_optional_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_Conv(X, W, B, Y, auto_pad, dilations, group, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: Conv  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Conv)
    };


    struct MaxUnpool : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_ints(stack);

            auto output_shape = fetch_optional_tensor(stack);
            auto I = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_MaxUnpool_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("strides", strides);
                infer_.new_attr("pads", pads);
                infer_.new_attr("kernel_shape", kernel_shape);

                infer_.new_input(X);
                infer_.new_input(I);
                infer_.new_input(output_shape);

                auto f = query_inference_function("MaxUnpool");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( X->onnx_MaxUnpool(X, I, output_shape, output, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: MaxUnpool  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_ints(stack);

            auto output_shape = fetch_optional_tensor(stack);
            auto I = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_MaxUnpool(X, I, output_shape, output, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: MaxUnpool  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MaxUnpool)
    };


    struct TfIdfVectorizer : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto weights = fetch_optional_floats(stack, {});
            auto pool_strings = fetch_optional_strings(stack, {});
            auto pool_int64s = fetch_optional_ints(stack, {});
            auto ngram_indexes = fetch_ints(stack);
            auto ngram_counts = fetch_ints(stack);
            auto mode = fetch_string(stack);
            auto min_gram_length = fetch_int(stack);
            auto max_skip_count = fetch_int(stack);
            auto max_gram_length = fetch_int(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_TfIdfVectorizer_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("weights", weights);
                infer_.new_attr("pool_strings", pool_strings);
                infer_.new_attr("pool_int64s", pool_int64s);
                infer_.new_attr("ngram_indexes", ngram_indexes);
                infer_.new_attr("ngram_counts", ngram_counts);
                infer_.new_attr("mode", mode);
                infer_.new_attr("min_gram_length", min_gram_length);
                infer_.new_attr("max_skip_count", max_skip_count);
                infer_.new_attr("max_gram_length", max_gram_length);

                infer_.new_input(X);

                auto f = query_inference_function("TfIdfVectorizer");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_TfIdfVectorizer(X, Y, max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings, weights) != YNX_OK ) {
                yannx_panic("API: TfIdfVectorizer  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto weights = fetch_optional_floats(stack, {});
            auto pool_strings = fetch_optional_strings(stack, {});
            auto pool_int64s = fetch_optional_ints(stack, {});
            auto ngram_indexes = fetch_ints(stack);
            auto ngram_counts = fetch_ints(stack);
            auto mode = fetch_string(stack);
            auto min_gram_length = fetch_int(stack);
            auto max_skip_count = fetch_int(stack);
            auto max_gram_length = fetch_int(stack);

            auto X = fetch_tensor(stack);


            if ( X->onnx_TfIdfVectorizer(X, Y, max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings, weights) != YNX_OK ) {
                yannx_panic("API: TfIdfVectorizer  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(TfIdfVectorizer)
    };


    struct ConvTranspose : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto output_shape = fetch_optional_ints(stack, {});
            auto output_padding = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_optional_ints(stack, {});
            auto group = fetch_optional_int(stack, 1);
            auto dilations = fetch_optional_ints(stack, {});
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto B = fetch_optional_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_ConvTranspose_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("strides", strides);
                infer_.new_attr("pads", pads);
                infer_.new_attr("output_shape", output_shape);
                infer_.new_attr("output_padding", output_padding);
                infer_.new_attr("kernel_shape", kernel_shape);
                infer_.new_attr("group", group);
                infer_.new_attr("dilations", dilations);
                infer_.new_attr("auto_pad", auto_pad);

                infer_.new_input(X);
                infer_.new_input(W);
                infer_.new_input(B);

                auto f = query_inference_function("ConvTranspose");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_ConvTranspose(X, W, B, Y, auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: ConvTranspose  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto output_shape = fetch_optional_ints(stack, {});
            auto output_padding = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_optional_ints(stack, {});
            auto group = fetch_optional_int(stack, 1);
            auto dilations = fetch_optional_ints(stack, {});
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto B = fetch_optional_tensor(stack);
            auto W = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_ConvTranspose(X, W, B, Y, auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: ConvTranspose  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ConvTranspose)
    };


    struct Flatten : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto axis = fetch_optional_int(stack, 1);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Flatten_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(input);

                auto f = query_inference_function("Flatten");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Flatten(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Flatten  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, 1);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Flatten(input, output, axis) != YNX_OK ) {
                yannx_panic("API: Flatten  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Flatten)
    };


    struct MaxRoiPool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto spatial_scale = fetch_optional_float(stack, 1);
            auto pooled_shape = fetch_ints(stack);

            auto rois = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_MaxRoiPool_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("spatial_scale", spatial_scale);
                infer_.new_attr("pooled_shape", pooled_shape);

                infer_.new_input(X);
                infer_.new_input(rois);

                auto f = query_inference_function("MaxRoiPool");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_MaxRoiPool(X, rois, Y, pooled_shape, spatial_scale) != YNX_OK ) {
                yannx_panic("API: MaxRoiPool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto spatial_scale = fetch_optional_float(stack, 1);
            auto pooled_shape = fetch_ints(stack);

            auto rois = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_MaxRoiPool(X, rois, Y, pooled_shape, spatial_scale) != YNX_OK ) {
                yannx_panic("API: MaxRoiPool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MaxRoiPool)
    };


    struct InstanceNormalization : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto epsilon = fetch_optional_float(stack, 1e-05);

            auto B = fetch_tensor(stack);
            auto scale = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_InstanceNormalization_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("epsilon", epsilon);

                infer_.new_input(input);
                infer_.new_input(scale);
                infer_.new_input(B);

                auto f = query_inference_function("InstanceNormalization");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_InstanceNormalization(input, scale, B, output, epsilon) != YNX_OK ) {
                yannx_panic("API: InstanceNormalization  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto epsilon = fetch_optional_float(stack, 1e-05);

            auto B = fetch_tensor(stack);
            auto scale = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_InstanceNormalization(input, scale, B, output, epsilon) != YNX_OK ) {
                yannx_panic("API: InstanceNormalization  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(InstanceNormalization)
    };


    struct LpPool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto p = fetch_optional_int(stack, 2);
            auto kernel_shape = fetch_ints(stack);
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto X = fetch_tensor(stack);


            if ( X->onnx_LpPool_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("strides", strides);
                infer_.new_attr("pads", pads);
                infer_.new_attr("p", p);
                infer_.new_attr("kernel_shape", kernel_shape);
                infer_.new_attr("auto_pad", auto_pad);

                infer_.new_input(X);

                auto f = query_inference_function("LpPool");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_LpPool(X, Y, auto_pad, kernel_shape, p, pads, strides) != YNX_OK ) {
                yannx_panic("API: LpPool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto p = fetch_optional_int(stack, 2);
            auto kernel_shape = fetch_ints(stack);
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto X = fetch_tensor(stack);


            if ( X->onnx_LpPool(X, Y, auto_pad, kernel_shape, p, pads, strides) != YNX_OK ) {
                yannx_panic("API: LpPool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LpPool)
    };


    struct AveragePool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_ints(stack);
            auto count_include_pad = fetch_optional_int(stack, 0);
            auto ceil_mode = fetch_optional_int(stack, 0);
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto X = fetch_tensor(stack);


            if ( X->onnx_AveragePool_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("strides", strides);
                infer_.new_attr("pads", pads);
                infer_.new_attr("kernel_shape", kernel_shape);
                infer_.new_attr("count_include_pad", count_include_pad);
                infer_.new_attr("ceil_mode", ceil_mode);
                infer_.new_attr("auto_pad", auto_pad);

                infer_.new_input(X);

                auto f = query_inference_function("AveragePool");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_AveragePool(X, Y, auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: AveragePool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto strides = fetch_optional_ints(stack, {});
            auto pads = fetch_optional_ints(stack, {});
            auto kernel_shape = fetch_ints(stack);
            auto count_include_pad = fetch_optional_int(stack, 0);
            auto ceil_mode = fetch_optional_int(stack, 0);
            auto auto_pad = fetch_optional_string(stack, "NOTSET");

            auto X = fetch_tensor(stack);


            if ( X->onnx_AveragePool(X, Y, auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides) != YNX_OK ) {
                yannx_panic("API: AveragePool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(AveragePool)
    };


    struct Dropout : NativeWord<TensorType> {
        std::variant<void *, tensor_t> mask;
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            if ( fetch_bool(stack) == true) {
                mask = rt.create_undefined_user_tensor();
            }
            output = rt.create_undefined_user_tensor();

            auto seed = fetch_optional_int(stack, 0);

            auto training_mode = fetch_optional_tensor(stack);
            auto ratio = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Dropout_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                if ( mask.index() != 0) {
                    outputs_.push_back(1);
                }
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("seed", seed);

                infer_.new_input(data);
                infer_.new_input(ratio);
                infer_.new_input(training_mode);

                auto f = query_inference_function("Dropout");
                infer_.do_inference(f);
                infer_.check_output(0, output);
                if ( mask.index() != 0) {
                    infer_.check_output(1, std::get<1>(mask));
                }


            }

            if ( data->onnx_Dropout(data, ratio, training_mode, output, mask, seed) != YNX_OK ) {
                yannx_panic("API: Dropout  return error!");
            }

            put_tensor(stack, output);
            if ( mask.index() != 0) {
                put_optional_tensor(stack, mask);
            }

        }
        virtual void run(ValueStack<TensorType>& stack) {

            fetch_bool(stack);

            auto seed = fetch_optional_int(stack, 0);

            auto training_mode = fetch_optional_tensor(stack);
            auto ratio = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Dropout(data, ratio, training_mode, output, mask, seed) != YNX_OK ) {
                yannx_panic("API: Dropout  return error!");
            }

            put_tensor(stack, output);
            if ( mask.index() != 0) {
                put_optional_tensor(stack, mask);
            }

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Dropout)
    };


    struct BatchNormalization : NativeWord<TensorType> {
        std::variant<void *, tensor_t> running_var;
        std::variant<void *, tensor_t> running_mean;
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            if ( fetch_bool(stack) == true) {
                running_var = rt.create_undefined_user_tensor();
            }
            if ( fetch_bool(stack) == true) {
                running_mean = rt.create_undefined_user_tensor();
            }
            Y = rt.create_undefined_user_tensor();

            auto training_mode = fetch_optional_int(stack, 0);
            auto momentum = fetch_optional_float(stack, 0.9);
            auto epsilon = fetch_optional_float(stack, 1e-05);

            auto input_var = fetch_tensor(stack);
            auto input_mean = fetch_tensor(stack);
            auto B = fetch_tensor(stack);
            auto scale = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_BatchNormalization_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                if ( running_mean.index() != 0) {
                    outputs_.push_back(1);
                }
                if ( running_var.index() != 0) {
                    outputs_.push_back(2);
                }
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("training_mode", training_mode);
                infer_.new_attr("momentum", momentum);
                infer_.new_attr("epsilon", epsilon);

                infer_.new_input(X);
                infer_.new_input(scale);
                infer_.new_input(B);
                infer_.new_input(input_mean);
                infer_.new_input(input_var);

                auto f = query_inference_function("BatchNormalization");
                infer_.do_inference(f);
                infer_.check_output(0, Y);
                if ( running_mean.index() != 0) {
                    infer_.check_output(1, std::get<1>(running_mean));
                }
                if ( running_var.index() != 0) {
                    infer_.check_output(2, std::get<1>(running_var));
                }


            }

            if ( X->onnx_BatchNormalization(X, scale, B, input_mean, input_var, Y, running_mean, running_var, epsilon, momentum, training_mode) != YNX_OK ) {
                yannx_panic("API: BatchNormalization  return error!");
            }

            put_tensor(stack, Y);
            if ( running_mean.index() != 0) {
                put_optional_tensor(stack, running_mean);
            }
            if ( running_var.index() != 0) {
                put_optional_tensor(stack, running_var);
            }

        }
        virtual void run(ValueStack<TensorType>& stack) {

            fetch_bool(stack);
            fetch_bool(stack);

            auto training_mode = fetch_optional_int(stack, 0);
            auto momentum = fetch_optional_float(stack, 0.9);
            auto epsilon = fetch_optional_float(stack, 1e-05);

            auto input_var = fetch_tensor(stack);
            auto input_mean = fetch_tensor(stack);
            auto B = fetch_tensor(stack);
            auto scale = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_BatchNormalization(X, scale, B, input_mean, input_var, Y, running_mean, running_var, epsilon, momentum, training_mode) != YNX_OK ) {
                yannx_panic("API: BatchNormalization  return error!");
            }

            put_tensor(stack, Y);
            if ( running_mean.index() != 0) {
                put_optional_tensor(stack, running_mean);
            }
            if ( running_var.index() != 0) {
                put_optional_tensor(stack, running_var);
            }

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(BatchNormalization)
    };


    struct LpNormalization : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto p = fetch_optional_int(stack, 2);
            auto axis = fetch_optional_int(stack, -1);

            auto input = fetch_tensor(stack);


            if ( input->onnx_LpNormalization_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("p", p);
                infer_.new_attr("axis", axis);

                infer_.new_input(input);

                auto f = query_inference_function("LpNormalization");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_LpNormalization(input, output, axis, p) != YNX_OK ) {
                yannx_panic("API: LpNormalization  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto p = fetch_optional_int(stack, 2);
            auto axis = fetch_optional_int(stack, -1);

            auto input = fetch_tensor(stack);


            if ( input->onnx_LpNormalization(input, output, axis, p) != YNX_OK ) {
                yannx_panic("API: LpNormalization  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LpNormalization)
    };


    struct GlobalLpPool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto p = fetch_optional_int(stack, 2);

            auto X = fetch_tensor(stack);


            if ( X->onnx_GlobalLpPool_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("p", p);

                infer_.new_input(X);

                auto f = query_inference_function("GlobalLpPool");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_GlobalLpPool(X, Y, p) != YNX_OK ) {
                yannx_panic("API: GlobalLpPool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto p = fetch_optional_int(stack, 2);

            auto X = fetch_tensor(stack);


            if ( X->onnx_GlobalLpPool(X, Y, p) != YNX_OK ) {
                yannx_panic("API: GlobalLpPool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GlobalLpPool)
    };


    struct GlobalMaxPool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_GlobalMaxPool_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("GlobalMaxPool");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_GlobalMaxPool(X, Y) != YNX_OK ) {
                yannx_panic("API: GlobalMaxPool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_GlobalMaxPool(X, Y) != YNX_OK ) {
                yannx_panic("API: GlobalMaxPool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GlobalMaxPool)
    };


    struct MeanVarianceNormalization : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto axes = fetch_optional_ints(stack, {0, 2, 3, });

            auto X = fetch_tensor(stack);


            if ( X->onnx_MeanVarianceNormalization_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axes", axes);

                infer_.new_input(X);

                auto f = query_inference_function("MeanVarianceNormalization");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_MeanVarianceNormalization(X, Y, axes) != YNX_OK ) {
                yannx_panic("API: MeanVarianceNormalization  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axes = fetch_optional_ints(stack, {0, 2, 3, });

            auto X = fetch_tensor(stack);


            if ( X->onnx_MeanVarianceNormalization(X, Y, axes) != YNX_OK ) {
                yannx_panic("API: MeanVarianceNormalization  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(MeanVarianceNormalization)
    };


    struct GlobalAveragePool : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_GlobalAveragePool_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("GlobalAveragePool");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_GlobalAveragePool(X, Y) != YNX_OK ) {
                yannx_panic("API: GlobalAveragePool  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_GlobalAveragePool(X, Y) != YNX_OK ) {
                yannx_panic("API: GlobalAveragePool  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GlobalAveragePool)
    };


    struct LRN : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto size = fetch_int(stack);
            auto bias = fetch_optional_float(stack, 1);
            auto beta = fetch_optional_float(stack, 0.75);
            auto alpha = fetch_optional_float(stack, 0.0001);

            auto X = fetch_tensor(stack);


            if ( X->onnx_LRN_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("size", size);
                infer_.new_attr("bias", bias);
                infer_.new_attr("beta", beta);
                infer_.new_attr("alpha", alpha);

                infer_.new_input(X);

                auto f = query_inference_function("LRN");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_LRN(X, Y, alpha, beta, bias, size) != YNX_OK ) {
                yannx_panic("API: LRN  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto size = fetch_int(stack);
            auto bias = fetch_optional_float(stack, 1);
            auto beta = fetch_optional_float(stack, 0.75);
            auto alpha = fetch_optional_float(stack, 0.0001);

            auto X = fetch_tensor(stack);


            if ( X->onnx_LRN(X, Y, alpha, beta, bias, size) != YNX_OK ) {
                yannx_panic("API: LRN  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(LRN)
    };


    struct StringNormalizer : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto stopwords = fetch_optional_strings(stack, {});
            auto locale = fetch_optional_string(stack, "");
            auto is_case_sensitive = fetch_optional_int(stack, 0);
            auto case_change_action = fetch_optional_string(stack, "NONE");

            auto X = fetch_tensor(stack);


            if ( X->onnx_StringNormalizer_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("stopwords", stopwords);
                infer_.new_attr("locale", locale);
                infer_.new_attr("is_case_sensitive", is_case_sensitive);
                infer_.new_attr("case_change_action", case_change_action);

                infer_.new_input(X);

                auto f = query_inference_function("StringNormalizer");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_StringNormalizer(X, Y, case_change_action, is_case_sensitive, locale, stopwords) != YNX_OK ) {
                yannx_panic("API: StringNormalizer  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto stopwords = fetch_optional_strings(stack, {});
            auto locale = fetch_optional_string(stack, "");
            auto is_case_sensitive = fetch_optional_int(stack, 0);
            auto case_change_action = fetch_optional_string(stack, "NONE");

            auto X = fetch_tensor(stack);


            if ( X->onnx_StringNormalizer(X, Y, case_change_action, is_case_sensitive, locale, stopwords) != YNX_OK ) {
                yannx_panic("API: StringNormalizer  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(StringNormalizer)
    };


    struct Shrink : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto lambd = fetch_optional_float(stack, 0.5);
            auto bias = fetch_optional_float(stack, 0);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Shrink_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("lambd", lambd);
                infer_.new_attr("bias", bias);

                infer_.new_input(input);

                auto f = query_inference_function("Shrink");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Shrink(input, output, bias, lambd) != YNX_OK ) {
                yannx_panic("API: Shrink  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto lambd = fetch_optional_float(stack, 0.5);
            auto bias = fetch_optional_float(stack, 0);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Shrink(input, output, bias, lambd) != YNX_OK ) {
                yannx_panic("API: Shrink  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Shrink)
    };

}
namespace object_detection {

    struct RoiAlign : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto spatial_scale = fetch_optional_float(stack, 1);
            auto sampling_ratio = fetch_optional_int(stack, 0);
            auto output_width = fetch_optional_int(stack, 1);
            auto output_height = fetch_optional_int(stack, 1);
            auto mode = fetch_optional_string(stack, "avg");
            auto coordinate_transformation_mode = fetch_optional_string(stack, "half_pixel");

            auto batch_indices = fetch_tensor(stack);
            auto rois = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_RoiAlign_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("spatial_scale", spatial_scale);
                infer_.new_attr("sampling_ratio", sampling_ratio);
                infer_.new_attr("output_width", output_width);
                infer_.new_attr("output_height", output_height);
                infer_.new_attr("mode", mode);
                infer_.new_attr("coordinate_transformation_mode", coordinate_transformation_mode);

                infer_.new_input(X);
                infer_.new_input(rois);
                infer_.new_input(batch_indices);

                auto f = query_inference_function("RoiAlign");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_RoiAlign(X, rois, batch_indices, Y, coordinate_transformation_mode, mode, output_height, output_width, sampling_ratio, spatial_scale) != YNX_OK ) {
                yannx_panic("API: RoiAlign  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto spatial_scale = fetch_optional_float(stack, 1);
            auto sampling_ratio = fetch_optional_int(stack, 0);
            auto output_width = fetch_optional_int(stack, 1);
            auto output_height = fetch_optional_int(stack, 1);
            auto mode = fetch_optional_string(stack, "avg");
            auto coordinate_transformation_mode = fetch_optional_string(stack, "half_pixel");

            auto batch_indices = fetch_tensor(stack);
            auto rois = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_RoiAlign(X, rois, batch_indices, Y, coordinate_transformation_mode, mode, output_height, output_width, sampling_ratio, spatial_scale) != YNX_OK ) {
                yannx_panic("API: RoiAlign  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(RoiAlign)
    };


    struct NonMaxSuppression : NativeWord<TensorType> {
        tensor_t selected_indices;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            selected_indices = rt.create_undefined_user_tensor();

            auto center_point_box = fetch_optional_int(stack, 0);

            auto score_threshold = fetch_optional_tensor(stack);
            auto iou_threshold = fetch_optional_tensor(stack);
            auto max_output_boxes_per_class = fetch_optional_tensor(stack);
            auto scores = fetch_tensor(stack);
            auto boxes = fetch_tensor(stack);


            if ( boxes->onnx_NonMaxSuppression_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("center_point_box", center_point_box);

                infer_.new_input(boxes);
                infer_.new_input(scores);
                infer_.new_input(max_output_boxes_per_class);
                infer_.new_input(iou_threshold);
                infer_.new_input(score_threshold);

                auto f = query_inference_function("NonMaxSuppression");
                infer_.do_inference(f);
                infer_.check_output(0, selected_indices);


            }

            if ( boxes->onnx_NonMaxSuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, selected_indices, center_point_box) != YNX_OK ) {
                yannx_panic("API: NonMaxSuppression  return error!");
            }

            put_tensor(stack, selected_indices);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto center_point_box = fetch_optional_int(stack, 0);

            auto score_threshold = fetch_optional_tensor(stack);
            auto iou_threshold = fetch_optional_tensor(stack);
            auto max_output_boxes_per_class = fetch_optional_tensor(stack);
            auto scores = fetch_tensor(stack);
            auto boxes = fetch_tensor(stack);


            if ( boxes->onnx_NonMaxSuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, selected_indices, center_point_box) != YNX_OK ) {
                yannx_panic("API: NonMaxSuppression  return error!");
            }

            put_tensor(stack, selected_indices);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(NonMaxSuppression)
    };

}
namespace quantization {

    struct DequantizeLinear : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            y = rt.create_undefined_user_tensor();

            auto axis = fetch_optional_int(stack, 1);

            auto x_zero_point = fetch_optional_tensor(stack);
            auto x_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_DequantizeLinear_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(x);
                infer_.new_input(x_scale);
                infer_.new_input(x_zero_point);

                auto f = query_inference_function("DequantizeLinear");
                infer_.do_inference(f);
                infer_.check_output(0, y);


            }

            if ( x->onnx_DequantizeLinear(x, x_scale, x_zero_point, y, axis) != YNX_OK ) {
                yannx_panic("API: DequantizeLinear  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, 1);

            auto x_zero_point = fetch_optional_tensor(stack);
            auto x_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_DequantizeLinear(x, x_scale, x_zero_point, y, axis) != YNX_OK ) {
                yannx_panic("API: DequantizeLinear  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(DequantizeLinear)
    };


    struct QuantizeLinear : NativeWord<TensorType> {
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            y = rt.create_undefined_user_tensor();

            auto axis = fetch_optional_int(stack, 1);

            auto y_zero_point = fetch_optional_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_QuantizeLinear_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(x);
                infer_.new_input(y_scale);
                infer_.new_input(y_zero_point);

                auto f = query_inference_function("QuantizeLinear");
                infer_.do_inference(f);
                infer_.check_output(0, y);


            }

            if ( x->onnx_QuantizeLinear(x, y_scale, y_zero_point, y, axis) != YNX_OK ) {
                yannx_panic("API: QuantizeLinear  return error!");
            }

            put_tensor(stack, y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, 1);

            auto y_zero_point = fetch_optional_tensor(stack);
            auto y_scale = fetch_tensor(stack);
            auto x = fetch_tensor(stack);


            if ( x->onnx_QuantizeLinear(x, y_scale, y_zero_point, y, axis) != YNX_OK ) {
                yannx_panic("API: QuantizeLinear  return error!");
            }

            put_tensor(stack, y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(QuantizeLinear)
    };


    struct DynamicQuantizeLinear : NativeWord<TensorType> {
        tensor_t y_zero_point;
        tensor_t y_scale;
        tensor_t y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            y_zero_point = rt.create_undefined_user_tensor();
            y_scale = rt.create_undefined_user_tensor();
            y = rt.create_undefined_user_tensor();


            auto x = fetch_tensor(stack);


            if ( x->onnx_DynamicQuantizeLinear_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                outputs_.push_back(1);
                outputs_.push_back(2);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(x);

                auto f = query_inference_function("DynamicQuantizeLinear");
                infer_.do_inference(f);
                infer_.check_output(0, y);
                infer_.check_output(1, y_scale);
                infer_.check_output(2, y_zero_point);


            }

            if ( x->onnx_DynamicQuantizeLinear(x, y, y_scale, y_zero_point) != YNX_OK ) {
                yannx_panic("API: DynamicQuantizeLinear  return error!");
            }

            put_tensor(stack, y);
            put_tensor(stack, y_scale);
            put_tensor(stack, y_zero_point);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto x = fetch_tensor(stack);


            if ( x->onnx_DynamicQuantizeLinear(x, y, y_scale, y_zero_point) != YNX_OK ) {
                yannx_panic("API: DynamicQuantizeLinear  return error!");
            }

            put_tensor(stack, y);
            put_tensor(stack, y_scale);
            put_tensor(stack, y_zero_point);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(DynamicQuantizeLinear)
    };

}
namespace reduction {

    struct ArgMax : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto select_last_index = fetch_optional_int(stack, 0);
            auto keepdims = fetch_optional_int(stack, 1);
            auto axis = fetch_optional_int(stack, 0);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ArgMax_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("select_last_index", select_last_index);
                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axis", axis);

                infer_.new_input(data);

                auto f = query_inference_function("ArgMax");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ArgMax(data, reduced, axis, keepdims, select_last_index) != YNX_OK ) {
                yannx_panic("API: ArgMax  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto select_last_index = fetch_optional_int(stack, 0);
            auto keepdims = fetch_optional_int(stack, 1);
            auto axis = fetch_optional_int(stack, 0);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ArgMax(data, reduced, axis, keepdims, select_last_index) != YNX_OK ) {
                yannx_panic("API: ArgMax  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ArgMax)
    };


    struct ReduceMax : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceMax_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axes", axes);

                infer_.new_input(data);

                auto f = query_inference_function("ReduceMax");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ReduceMax(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMax  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceMax(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMax  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceMax)
    };


    struct ReduceMin : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceMin_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axes", axes);

                infer_.new_input(data);

                auto f = query_inference_function("ReduceMin");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ReduceMin(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMin  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceMin(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMin  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceMin)
    };


    struct ReduceL2 : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceL2_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axes", axes);

                infer_.new_input(data);

                auto f = query_inference_function("ReduceL2");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ReduceL2(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceL2  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceL2(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceL2  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceL2)
    };


    struct ReduceLogSum : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceLogSum_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axes", axes);

                infer_.new_input(data);

                auto f = query_inference_function("ReduceLogSum");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ReduceLogSum(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceLogSum  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceLogSum(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceLogSum  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceLogSum)
    };


    struct ReduceLogSumExp : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceLogSumExp_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axes", axes);

                infer_.new_input(data);

                auto f = query_inference_function("ReduceLogSumExp");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ReduceLogSumExp(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceLogSumExp  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceLogSumExp(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceLogSumExp  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceLogSumExp)
    };


    struct ArgMin : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto select_last_index = fetch_optional_int(stack, 0);
            auto keepdims = fetch_optional_int(stack, 1);
            auto axis = fetch_optional_int(stack, 0);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ArgMin_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("select_last_index", select_last_index);
                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axis", axis);

                infer_.new_input(data);

                auto f = query_inference_function("ArgMin");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ArgMin(data, reduced, axis, keepdims, select_last_index) != YNX_OK ) {
                yannx_panic("API: ArgMin  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto select_last_index = fetch_optional_int(stack, 0);
            auto keepdims = fetch_optional_int(stack, 1);
            auto axis = fetch_optional_int(stack, 0);

            auto data = fetch_tensor(stack);


            if ( data->onnx_ArgMin(data, reduced, axis, keepdims, select_last_index) != YNX_OK ) {
                yannx_panic("API: ArgMin  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ArgMin)
    };


    struct ReduceMean : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceMean_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axes", axes);

                infer_.new_input(data);

                auto f = query_inference_function("ReduceMean");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ReduceMean(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMean  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceMean(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceMean  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceMean)
    };


    struct ReduceL1 : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceL1_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axes", axes);

                infer_.new_input(data);

                auto f = query_inference_function("ReduceL1");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ReduceL1(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceL1  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceL1(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceL1  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceL1)
    };


    struct ReduceSum : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto noop_with_empty_axes = fetch_optional_int(stack, 0);
            auto keepdims = fetch_optional_int(stack, 1);

            auto axes = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceSum_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("noop_with_empty_axes", noop_with_empty_axes);
                infer_.new_attr("keepdims", keepdims);

                infer_.new_input(data);
                infer_.new_input(axes);

                auto f = query_inference_function("ReduceSum");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ReduceSum(data, axes, reduced, keepdims, noop_with_empty_axes) != YNX_OK ) {
                yannx_panic("API: ReduceSum  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto noop_with_empty_axes = fetch_optional_int(stack, 0);
            auto keepdims = fetch_optional_int(stack, 1);

            auto axes = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceSum(data, axes, reduced, keepdims, noop_with_empty_axes) != YNX_OK ) {
                yannx_panic("API: ReduceSum  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceSum)
    };


    struct ReduceSumSquare : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceSumSquare_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axes", axes);

                infer_.new_input(data);

                auto f = query_inference_function("ReduceSumSquare");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ReduceSumSquare(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceSumSquare  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceSumSquare(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceSumSquare  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceSumSquare)
    };


    struct ReduceProd : NativeWord<TensorType> {
        tensor_t reduced;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reduced = rt.create_undefined_user_tensor();

            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceProd_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("keepdims", keepdims);
                infer_.new_attr("axes", axes);

                infer_.new_input(data);

                auto f = query_inference_function("ReduceProd");
                infer_.do_inference(f);
                infer_.check_output(0, reduced);


            }

            if ( data->onnx_ReduceProd(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceProd  return error!");
            }

            put_tensor(stack, reduced);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto keepdims = fetch_optional_int(stack, 1);
            auto axes = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_ReduceProd(data, reduced, axes, keepdims) != YNX_OK ) {
                yannx_panic("API: ReduceProd  return error!");
            }

            put_tensor(stack, reduced);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReduceProd)
    };

}
namespace tensor {

    struct GridSample : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto padding_mode = fetch_optional_string(stack, "zeros");
            auto mode = fetch_optional_string(stack, "bilinear");
            auto align_corners = fetch_optional_int(stack, 0);

            auto grid = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_GridSample_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("padding_mode", padding_mode);
                infer_.new_attr("mode", mode);
                infer_.new_attr("align_corners", align_corners);

                infer_.new_input(X);
                infer_.new_input(grid);

                auto f = query_inference_function("GridSample");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_GridSample(X, grid, Y, align_corners, mode, padding_mode) != YNX_OK ) {
                yannx_panic("API: GridSample  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto padding_mode = fetch_optional_string(stack, "zeros");
            auto mode = fetch_optional_string(stack, "bilinear");
            auto align_corners = fetch_optional_int(stack, 0);

            auto grid = fetch_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_GridSample(X, grid, Y, align_corners, mode, padding_mode) != YNX_OK ) {
                yannx_panic("API: GridSample  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GridSample)
    };


    struct CastLike : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto target_type = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_CastLike_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);
                infer_.new_input(target_type);

                auto f = query_inference_function("CastLike");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_CastLike(input, target_type, output) != YNX_OK ) {
                yannx_panic("API: CastLike  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto target_type = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_CastLike(input, target_type, output) != YNX_OK ) {
                yannx_panic("API: CastLike  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(CastLike)
    };


    struct GatherND : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto batch_dims = fetch_optional_int(stack, 0);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_GatherND_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("batch_dims", batch_dims);

                infer_.new_input(data);
                infer_.new_input(indices);

                auto f = query_inference_function("GatherND");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( data->onnx_GatherND(data, indices, output, batch_dims) != YNX_OK ) {
                yannx_panic("API: GatherND  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto batch_dims = fetch_optional_int(stack, 0);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_GatherND(data, indices, output, batch_dims) != YNX_OK ) {
                yannx_panic("API: GatherND  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GatherND)
    };


    struct ScatterND : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto reduction = fetch_optional_string(stack, "none");

            auto updates = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_ScatterND_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("reduction", reduction);

                infer_.new_input(data);
                infer_.new_input(indices);
                infer_.new_input(updates);

                auto f = query_inference_function("ScatterND");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( data->onnx_ScatterND(data, indices, updates, output, reduction) != YNX_OK ) {
                yannx_panic("API: ScatterND  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto reduction = fetch_optional_string(stack, "none");

            auto updates = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_ScatterND(data, indices, updates, output, reduction) != YNX_OK ) {
                yannx_panic("API: ScatterND  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ScatterND)
    };


    struct GatherElements : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto axis = fetch_optional_int(stack, 0);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_GatherElements_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(data);
                infer_.new_input(indices);

                auto f = query_inference_function("GatherElements");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( data->onnx_GatherElements(data, indices, output, axis) != YNX_OK ) {
                yannx_panic("API: GatherElements  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, 0);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_GatherElements(data, indices, output, axis) != YNX_OK ) {
                yannx_panic("API: GatherElements  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(GatherElements)
    };


    struct IsInf : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto detect_positive = fetch_optional_int(stack, 1);
            auto detect_negative = fetch_optional_int(stack, 1);

            auto X = fetch_tensor(stack);


            if ( X->onnx_IsInf_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("detect_positive", detect_positive);
                infer_.new_attr("detect_negative", detect_negative);

                infer_.new_input(X);

                auto f = query_inference_function("IsInf");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_IsInf(X, Y, detect_negative, detect_positive) != YNX_OK ) {
                yannx_panic("API: IsInf  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto detect_positive = fetch_optional_int(stack, 1);
            auto detect_negative = fetch_optional_int(stack, 1);

            auto X = fetch_tensor(stack);


            if ( X->onnx_IsInf(X, Y, detect_negative, detect_positive) != YNX_OK ) {
                yannx_panic("API: IsInf  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(IsInf)
    };


    struct Split : NativeWord<TensorType> {
        std::vector<tensor_t> outputs;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            outputs.resize( fetch_int(stack) );
            for(size_t i = 0; i < outputs.size(); i++) {
                outputs[i] = rt.create_undefined_user_tensor();
            }

            auto axis = fetch_optional_int(stack, 0);

            auto split = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Split_typing() == false ) {
                std::vector<size_t> outputs_;
                for ( size_t i = 0; i < outputs.size(); i++) {
                    outputs_.push_back(i);
                }
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(input);
                infer_.new_input(split);

                auto f = query_inference_function("Split");
                infer_.do_inference(f);
                for ( size_t i = 0; i < outputs.size(); i++) {
                    infer_.check_output(i, outputs[i]);
                }


            }

            if ( input->onnx_Split(input, split, outputs, axis) != YNX_OK ) {
                yannx_panic("API: Split  return error!");
            }

            put_tensors(stack, outputs);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, 0);

            auto split = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Split(input, split, outputs, axis) != YNX_OK ) {
                yannx_panic("API: Split  return error!");
            }

            put_tensors(stack, outputs);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Split)
    };


    struct Cast : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto to = fetch_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Cast_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("to", to);

                infer_.new_input(input);

                auto f = query_inference_function("Cast");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Cast(input, output, to) != YNX_OK ) {
                yannx_panic("API: Cast  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto to = fetch_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_Cast(input, output, to) != YNX_OK ) {
                yannx_panic("API: Cast  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Cast)
    };


    struct Pad : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto mode = fetch_optional_string(stack, "constant");

            auto constant_value = fetch_optional_tensor(stack);
            auto pads = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Pad_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("mode", mode);

                infer_.new_input(data);
                infer_.new_input(pads);
                infer_.new_input(constant_value);

                auto f = query_inference_function("Pad");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( data->onnx_Pad(data, pads, constant_value, output, mode) != YNX_OK ) {
                yannx_panic("API: Pad  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto mode = fetch_optional_string(stack, "constant");

            auto constant_value = fetch_optional_tensor(stack);
            auto pads = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Pad(data, pads, constant_value, output, mode) != YNX_OK ) {
                yannx_panic("API: Pad  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Pad)
    };


    struct NonZero : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_NonZero_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("NonZero");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_NonZero(X, Y) != YNX_OK ) {
                yannx_panic("API: NonZero  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_NonZero(X, Y) != YNX_OK ) {
                yannx_panic("API: NonZero  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(NonZero)
    };


    struct Concat : NativeWord<TensorType> {
        tensor_t concat_result;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            concat_result = rt.create_undefined_user_tensor();

            auto axis = fetch_int(stack);

            auto inputs = fetch_tensors(stack);


            if ( concat_result->onnx_Concat_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(inputs);

                auto f = query_inference_function("Concat");
                infer_.do_inference(f);
                infer_.check_output(0, concat_result);


            }

            if ( concat_result->onnx_Concat(inputs, concat_result, axis) != YNX_OK ) {
                yannx_panic("API: Concat  return error!");
            }

            put_tensor(stack, concat_result);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_int(stack);

            auto inputs = fetch_tensors(stack);


            if ( concat_result->onnx_Concat(inputs, concat_result, axis) != YNX_OK ) {
                yannx_panic("API: Concat  return error!");
            }

            put_tensor(stack, concat_result);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Concat)
    };


    struct Unique : NativeWord<TensorType> {
        std::variant<void *, tensor_t> counts;
        std::variant<void *, tensor_t> inverse_indices;
        std::variant<void *, tensor_t> indices;
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            if ( fetch_bool(stack) == true) {
                counts = rt.create_undefined_user_tensor();
            }
            if ( fetch_bool(stack) == true) {
                inverse_indices = rt.create_undefined_user_tensor();
            }
            if ( fetch_bool(stack) == true) {
                indices = rt.create_undefined_user_tensor();
            }
            Y = rt.create_undefined_user_tensor();

            auto sorted = fetch_optional_int(stack, 1);
            auto axis = fetch_optional_int(stack, 0);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Unique_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                if ( indices.index() != 0) {
                    outputs_.push_back(1);
                }
                if ( inverse_indices.index() != 0) {
                    outputs_.push_back(2);
                }
                if ( counts.index() != 0) {
                    outputs_.push_back(3);
                }
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("sorted", sorted);
                infer_.new_attr("axis", axis);

                infer_.new_input(X);

                auto f = query_inference_function("Unique");
                infer_.do_inference(f);
                infer_.check_output(0, Y);
                if ( indices.index() != 0) {
                    infer_.check_output(1, std::get<1>(indices));
                }
                if ( inverse_indices.index() != 0) {
                    infer_.check_output(2, std::get<1>(inverse_indices));
                }
                if ( counts.index() != 0) {
                    infer_.check_output(3, std::get<1>(counts));
                }


            }

            if ( X->onnx_Unique(X, Y, indices, inverse_indices, counts, axis, sorted) != YNX_OK ) {
                yannx_panic("API: Unique  return error!");
            }

            put_tensor(stack, Y);
            if ( indices.index() != 0) {
                put_optional_tensor(stack, indices);
            }
            if ( inverse_indices.index() != 0) {
                put_optional_tensor(stack, inverse_indices);
            }
            if ( counts.index() != 0) {
                put_optional_tensor(stack, counts);
            }

        }
        virtual void run(ValueStack<TensorType>& stack) {

            fetch_bool(stack);
            fetch_bool(stack);
            fetch_bool(stack);

            auto sorted = fetch_optional_int(stack, 1);
            auto axis = fetch_optional_int(stack, 0);

            auto X = fetch_tensor(stack);


            if ( X->onnx_Unique(X, Y, indices, inverse_indices, counts, axis, sorted) != YNX_OK ) {
                yannx_panic("API: Unique  return error!");
            }

            put_tensor(stack, Y);
            if ( indices.index() != 0) {
                put_optional_tensor(stack, indices);
            }
            if ( inverse_indices.index() != 0) {
                put_optional_tensor(stack, inverse_indices);
            }
            if ( counts.index() != 0) {
                put_optional_tensor(stack, counts);
            }

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Unique)
    };


    struct Slice : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto steps = fetch_optional_tensor(stack);
            auto axes = fetch_optional_tensor(stack);
            auto ends = fetch_tensor(stack);
            auto starts = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Slice_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(data);
                infer_.new_input(starts);
                infer_.new_input(ends);
                infer_.new_input(axes);
                infer_.new_input(steps);

                auto f = query_inference_function("Slice");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( data->onnx_Slice(data, starts, ends, axes, steps, output) != YNX_OK ) {
                yannx_panic("API: Slice  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto steps = fetch_optional_tensor(stack);
            auto axes = fetch_optional_tensor(stack);
            auto ends = fetch_tensor(stack);
            auto starts = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Slice(data, starts, ends, axes, steps, output) != YNX_OK ) {
                yannx_panic("API: Slice  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Slice)
    };


    struct ReverseSequence : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto time_axis = fetch_optional_int(stack, 0);
            auto batch_axis = fetch_optional_int(stack, 1);

            auto sequence_lens = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_ReverseSequence_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("time_axis", time_axis);
                infer_.new_attr("batch_axis", batch_axis);

                infer_.new_input(input);
                infer_.new_input(sequence_lens);

                auto f = query_inference_function("ReverseSequence");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( input->onnx_ReverseSequence(input, sequence_lens, Y, batch_axis, time_axis) != YNX_OK ) {
                yannx_panic("API: ReverseSequence  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto time_axis = fetch_optional_int(stack, 0);
            auto batch_axis = fetch_optional_int(stack, 1);

            auto sequence_lens = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_ReverseSequence(input, sequence_lens, Y, batch_axis, time_axis) != YNX_OK ) {
                yannx_panic("API: ReverseSequence  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ReverseSequence)
    };


    struct OneHot : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto axis = fetch_optional_int(stack, -1);

            auto values = fetch_tensor(stack);
            auto depth = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);


            if ( indices->onnx_OneHot_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(indices);
                infer_.new_input(depth);
                infer_.new_input(values);

                auto f = query_inference_function("OneHot");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( indices->onnx_OneHot(indices, depth, values, output, axis) != YNX_OK ) {
                yannx_panic("API: OneHot  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, -1);

            auto values = fetch_tensor(stack);
            auto depth = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);


            if ( indices->onnx_OneHot(indices, depth, values, output, axis) != YNX_OK ) {
                yannx_panic("API: OneHot  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(OneHot)
    };


    struct Gather : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto axis = fetch_optional_int(stack, 0);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Gather_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(data);
                infer_.new_input(indices);

                auto f = query_inference_function("Gather");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( data->onnx_Gather(data, indices, output, axis) != YNX_OK ) {
                yannx_panic("API: Gather  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, 0);

            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Gather(data, indices, output, axis) != YNX_OK ) {
                yannx_panic("API: Gather  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Gather)
    };


    struct IsNaN : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();


            auto X = fetch_tensor(stack);


            if ( X->onnx_IsNaN_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(X);

                auto f = query_inference_function("IsNaN");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_IsNaN(X, Y) != YNX_OK ) {
                yannx_panic("API: IsNaN  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto X = fetch_tensor(stack);


            if ( X->onnx_IsNaN(X, Y) != YNX_OK ) {
                yannx_panic("API: IsNaN  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(IsNaN)
    };


    struct DepthToSpace : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto mode = fetch_optional_string(stack, "DCR");
            auto blocksize = fetch_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_DepthToSpace_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("mode", mode);
                infer_.new_attr("blocksize", blocksize);

                infer_.new_input(input);

                auto f = query_inference_function("DepthToSpace");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_DepthToSpace(input, output, blocksize, mode) != YNX_OK ) {
                yannx_panic("API: DepthToSpace  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto mode = fetch_optional_string(stack, "DCR");
            auto blocksize = fetch_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_DepthToSpace(input, output, blocksize, mode) != YNX_OK ) {
                yannx_panic("API: DepthToSpace  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(DepthToSpace)
    };


    struct ScatterElements : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto reduction = fetch_optional_string(stack, "none");
            auto axis = fetch_optional_int(stack, 0);

            auto updates = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_ScatterElements_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("reduction", reduction);
                infer_.new_attr("axis", axis);

                infer_.new_input(data);
                infer_.new_input(indices);
                infer_.new_input(updates);

                auto f = query_inference_function("ScatterElements");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( data->onnx_ScatterElements(data, indices, updates, output, axis, reduction) != YNX_OK ) {
                yannx_panic("API: ScatterElements  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto reduction = fetch_optional_string(stack, "none");
            auto axis = fetch_optional_int(stack, 0);

            auto updates = fetch_tensor(stack);
            auto indices = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_ScatterElements(data, indices, updates, output, axis, reduction) != YNX_OK ) {
                yannx_panic("API: ScatterElements  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(ScatterElements)
    };


    struct Reshape : NativeWord<TensorType> {
        tensor_t reshaped;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            reshaped = rt.create_undefined_user_tensor();

            auto allowzero = fetch_optional_int(stack, 0);

            auto shape = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Reshape_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("allowzero", allowzero);

                infer_.new_input(data);
                infer_.new_input(shape);

                auto f = query_inference_function("Reshape");
                infer_.do_inference(f);
                infer_.check_output(0, reshaped);


            }

            if ( data->onnx_Reshape(data, shape, reshaped, allowzero) != YNX_OK ) {
                yannx_panic("API: Reshape  return error!");
            }

            put_tensor(stack, reshaped);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto allowzero = fetch_optional_int(stack, 0);

            auto shape = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Reshape(data, shape, reshaped, allowzero) != YNX_OK ) {
                yannx_panic("API: Reshape  return error!");
            }

            put_tensor(stack, reshaped);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Reshape)
    };


    struct Shape : NativeWord<TensorType> {
        tensor_t shape;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            shape = rt.create_undefined_user_tensor();

            auto start = fetch_optional_int(stack, 0);
            auto end = fetch_optional_int(stack, 0);

            auto data = fetch_tensor(stack);


            if ( data->onnx_Shape_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("start", start);
                infer_.new_attr("end", end);

                infer_.new_input(data);

                auto f = query_inference_function("Shape");
                infer_.do_inference(f);
                infer_.check_output(0, shape);


            }

            if ( data->onnx_Shape(data, shape, end, start) != YNX_OK ) {
                yannx_panic("API: Shape  return error!");
            }

            put_tensor(stack, shape);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto start = fetch_optional_int(stack, 0);
            auto end = fetch_optional_int(stack, 0);

            auto data = fetch_tensor(stack);


            if ( data->onnx_Shape(data, shape, end, start) != YNX_OK ) {
                yannx_panic("API: Shape  return error!");
            }

            put_tensor(stack, shape);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Shape)
    };


    struct Size : NativeWord<TensorType> {
        tensor_t size;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            size = rt.create_undefined_user_tensor();


            auto data = fetch_tensor(stack);


            if ( data->onnx_Size_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(data);

                auto f = query_inference_function("Size");
                infer_.do_inference(f);
                infer_.check_output(0, size);


            }

            if ( data->onnx_Size(data, size) != YNX_OK ) {
                yannx_panic("API: Size  return error!");
            }

            put_tensor(stack, size);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto data = fetch_tensor(stack);


            if ( data->onnx_Size(data, size) != YNX_OK ) {
                yannx_panic("API: Size  return error!");
            }

            put_tensor(stack, size);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Size)
    };


    struct SpaceToDepth : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto blocksize = fetch_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_SpaceToDepth_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("blocksize", blocksize);

                infer_.new_input(input);

                auto f = query_inference_function("SpaceToDepth");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_SpaceToDepth(input, output, blocksize) != YNX_OK ) {
                yannx_panic("API: SpaceToDepth  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto blocksize = fetch_int(stack);

            auto input = fetch_tensor(stack);


            if ( input->onnx_SpaceToDepth(input, output, blocksize) != YNX_OK ) {
                yannx_panic("API: SpaceToDepth  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(SpaceToDepth)
    };


    struct Identity : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto input = fetch_tensor(stack);


            if ( input->onnx_Identity_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);

                auto f = query_inference_function("Identity");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Identity(input, output) != YNX_OK ) {
                yannx_panic("API: Identity  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto input = fetch_tensor(stack);


            if ( input->onnx_Identity(input, output) != YNX_OK ) {
                yannx_panic("API: Identity  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Identity)
    };


    struct Squeeze : NativeWord<TensorType> {
        tensor_t squeezed;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            squeezed = rt.create_undefined_user_tensor();


            auto axes = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Squeeze_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(data);
                infer_.new_input(axes);

                auto f = query_inference_function("Squeeze");
                infer_.do_inference(f);
                infer_.check_output(0, squeezed);


            }

            if ( data->onnx_Squeeze(data, axes, squeezed) != YNX_OK ) {
                yannx_panic("API: Squeeze  return error!");
            }

            put_tensor(stack, squeezed);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto axes = fetch_optional_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Squeeze(data, axes, squeezed) != YNX_OK ) {
                yannx_panic("API: Squeeze  return error!");
            }

            put_tensor(stack, squeezed);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Squeeze)
    };


    struct Tile : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto repeats = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Tile_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(input);
                infer_.new_input(repeats);

                auto f = query_inference_function("Tile");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Tile(input, repeats, output) != YNX_OK ) {
                yannx_panic("API: Tile  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto repeats = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Tile(input, repeats, output) != YNX_OK ) {
                yannx_panic("API: Tile  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Tile)
    };


    struct Unsqueeze : NativeWord<TensorType> {
        tensor_t expanded;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            expanded = rt.create_undefined_user_tensor();


            auto axes = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Unsqueeze_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(data);
                infer_.new_input(axes);

                auto f = query_inference_function("Unsqueeze");
                infer_.do_inference(f);
                infer_.check_output(0, expanded);


            }

            if ( data->onnx_Unsqueeze(data, axes, expanded) != YNX_OK ) {
                yannx_panic("API: Unsqueeze  return error!");
            }

            put_tensor(stack, expanded);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto axes = fetch_tensor(stack);
            auto data = fetch_tensor(stack);


            if ( data->onnx_Unsqueeze(data, axes, expanded) != YNX_OK ) {
                yannx_panic("API: Unsqueeze  return error!");
            }

            put_tensor(stack, expanded);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Unsqueeze)
    };


    struct Transpose : NativeWord<TensorType> {
        tensor_t transposed;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            transposed = rt.create_undefined_user_tensor();

            auto perm = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_Transpose_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("perm", perm);

                infer_.new_input(data);

                auto f = query_inference_function("Transpose");
                infer_.do_inference(f);
                infer_.check_output(0, transposed);


            }

            if ( data->onnx_Transpose(data, transposed, perm) != YNX_OK ) {
                yannx_panic("API: Transpose  return error!");
            }

            put_tensor(stack, transposed);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto perm = fetch_optional_ints(stack, {});

            auto data = fetch_tensor(stack);


            if ( data->onnx_Transpose(data, transposed, perm) != YNX_OK ) {
                yannx_panic("API: Transpose  return error!");
            }

            put_tensor(stack, transposed);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Transpose)
    };


    struct Compress : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto axis = fetch_optional_int(stack, 0);

            auto condition = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Compress_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("axis", axis);

                infer_.new_input(input);
                infer_.new_input(condition);

                auto f = query_inference_function("Compress");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Compress(input, condition, output, axis) != YNX_OK ) {
                yannx_panic("API: Compress  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto axis = fetch_optional_int(stack, 0);

            auto condition = fetch_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Compress(input, condition, output, axis) != YNX_OK ) {
                yannx_panic("API: Compress  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Compress)
    };


    struct Trilu : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();

            auto upper = fetch_optional_int(stack, 1);

            auto k = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Trilu_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("upper", upper);

                infer_.new_input(input);
                infer_.new_input(k);

                auto f = query_inference_function("Trilu");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( input->onnx_Trilu(input, k, output, upper) != YNX_OK ) {
                yannx_panic("API: Trilu  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto upper = fetch_optional_int(stack, 1);

            auto k = fetch_optional_tensor(stack);
            auto input = fetch_tensor(stack);


            if ( input->onnx_Trilu(input, k, output, upper) != YNX_OK ) {
                yannx_panic("API: Trilu  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Trilu)
    };


    struct Where : NativeWord<TensorType> {
        tensor_t output;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            output = rt.create_undefined_user_tensor();


            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);
            auto condition = fetch_tensor(stack);


            if ( condition->onnx_Where_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);


                infer_.new_input(condition);
                infer_.new_input(X);
                infer_.new_input(Y);

                auto f = query_inference_function("Where");
                infer_.do_inference(f);
                infer_.check_output(0, output);


            }

            if ( condition->onnx_Where(condition, X, Y, output) != YNX_OK ) {
                yannx_panic("API: Where  return error!");
            }

            put_tensor(stack, output);

        }
        virtual void run(ValueStack<TensorType>& stack) {



            auto Y = fetch_tensor(stack);
            auto X = fetch_tensor(stack);
            auto condition = fetch_tensor(stack);


            if ( condition->onnx_Where(condition, X, Y, output) != YNX_OK ) {
                yannx_panic("API: Where  return error!");
            }

            put_tensor(stack, output);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Where)
    };


    struct Resize : NativeWord<TensorType> {
        tensor_t Y;

        virtual void boot(Runtime<TensorType>& rt, WordHash<TensorType>& hash) {
            ValueStack<TensorType>& stack = rt;

            Y = rt.create_undefined_user_tensor();

            auto nearest_mode = fetch_optional_string(stack, "round_prefer_floor");
            auto mode = fetch_optional_string(stack, "nearest");
            auto extrapolation_value = fetch_optional_float(stack, 0);
            auto exclude_outside = fetch_optional_int(stack, 0);
            auto cubic_coeff_a = fetch_optional_float(stack, -0.75);
            auto coordinate_transformation_mode = fetch_optional_string(stack, "half_pixel");

            auto sizes = fetch_optional_tensor(stack);
            auto scales = fetch_optional_tensor(stack);
            auto roi = fetch_optional_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_Resize_typing() == false ) {
                std::vector<size_t> outputs_;
                outputs_.push_back(0);
                YNXInferenceContextImpl infer_(outputs_);

                infer_.new_attr("nearest_mode", nearest_mode);
                infer_.new_attr("mode", mode);
                infer_.new_attr("extrapolation_value", extrapolation_value);
                infer_.new_attr("exclude_outside", exclude_outside);
                infer_.new_attr("cubic_coeff_a", cubic_coeff_a);
                infer_.new_attr("coordinate_transformation_mode", coordinate_transformation_mode);

                infer_.new_input(X);
                infer_.new_input(roi);
                infer_.new_input(scales);
                infer_.new_input(sizes);

                auto f = query_inference_function("Resize");
                infer_.do_inference(f);
                infer_.check_output(0, Y);


            }

            if ( X->onnx_Resize(X, roi, scales, sizes, Y, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode) != YNX_OK ) {
                yannx_panic("API: Resize  return error!");
            }

            put_tensor(stack, Y);

        }
        virtual void run(ValueStack<TensorType>& stack) {


            auto nearest_mode = fetch_optional_string(stack, "round_prefer_floor");
            auto mode = fetch_optional_string(stack, "nearest");
            auto extrapolation_value = fetch_optional_float(stack, 0);
            auto exclude_outside = fetch_optional_int(stack, 0);
            auto cubic_coeff_a = fetch_optional_float(stack, -0.75);
            auto coordinate_transformation_mode = fetch_optional_string(stack, "half_pixel");

            auto sizes = fetch_optional_tensor(stack);
            auto scales = fetch_optional_tensor(stack);
            auto roi = fetch_optional_tensor(stack);
            auto X = fetch_tensor(stack);


            if ( X->onnx_Resize(X, roi, scales, sizes, Y, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode) != YNX_OK ) {
                yannx_panic("API: Resize  return error!");
            }

            put_tensor(stack, Y);

        }
        NWORD_CREATOR_DEFINE_TENSORTYPE(Resize)
    };

}
