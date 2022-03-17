# YaNNX
Yet Another Neural Network Exchange (YaNNX) is an open source format for AI models. 

YaNNX uses a stack expression to create a DAG of neural network some likes Forth language, so it is a semi-dynamic DSL for deep learning.
YaNNX supports user defined functions to descripting sub DAG, so YaNNX's model has a hierarchical structure and is easy readable for users, 
and these hierarchical structure is important for subsequent optimization in diffrent platfrom.

The data type and the operator's schemas are all inherited from ONNX project, the type and shape infrence functions are also used in YaNNX's tools.
YaNNX's runtime includes a global and local hash map (for every UDF) to storing or naming tensor，which give us more flexibility to descripting complex network.
The stack expression or language of YaNNX is easily to been implemented, it just is a few stack operators like 'pop', 'swap', 'rot' etc, and hash operator 'set' and 'get'.

This is a simple YaNNX model files, the constant weight is spelated with a msgpack format file.

```
;
;  A simple MLP example
;

3.14 "VERSION" set
"Lupu Mucic" "AUTHOR" set
"f32"   "DEFAULT_TYPE" set

[1 32] ("DEFAULT_TYPE" @) new_tensor "x" set       ; created a 2D tensor named with 'x' in global

"x" @ 128 fullconnect       ; first layer fullconnect with [1 128] output 
dup                         ; dup output from first layer in stack
64 fullconnect              ; [1 128] as input, output [1 64]
1 concat2                   ; there is two tensor in stack, do concat so output is [1 192]
logsoftmax                  ; just do logosftmax 

"y" set                     ; set output to "y" 

```

## The base syntax of YaNNX model file

The syntax of YaNNX is a stack expression, the base item of expresstion is `value` and `word`. 
Value is a number, a string, a number tuple, a tensor or a tensor tuple. 
Every 'value' will be pushed into stack as the order of code, `word` is a function (built-in or user defined) consumes values, pops them from stack, and push output into stack.
YaNNX provides two built-in word 'set' and 'get' to handle hashmap of global or local in user defined word, hashmap's key is a constant string and value is the 'value'.
The '@' is just a short-cut for 'get' word. Parentheses in code is meaningless, is just separator for readable. 
The quare brackets is used for create tuple value, such as number tuple, is only used in code parsing phase.

We provided a REPL ( Read, Evaluate, Print and Loop ) tool for just debug or testing YaNNX model files.


## Tensor tyoe in YaNNX

The tensor's data type is followed with ONNX, and high level type is variable tensor, constant tensor, parameter tensor. The constant tensor's data is stored in a msgpack file. 
The parameter tensor is used for training in the future.



