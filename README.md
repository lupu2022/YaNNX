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
def FullConnect
    "out_size" set
    "in_size"  set

    "out_size" @ "in_size" @ 2 .new_constant~
    "out_size" @ 1 .new_constant~ 
    
    ; now stack = x, w, b
    1.0 1.0 onnx.gemm
end

def MLP
    128 FullConnect
    0.1 onnx.LeakyRelu
    128 64 FullConnect
    0.1 onnx.LeakyRelu
    64 32 FullConnect
end

def Network
    80 MLP
    swap 
    40 MLP
    1 2 onnx.concat 
end

; define input
[1 40] .new_tensor~ 
dup "x1" set

[1 80] .new_tensor~
dup "x2" set

; run network
swap Network 

; define output
"y" set
```


## The base syntax of YaNNX model file

The syntax of YaNNX is a stack expression, the base item of expresstion is `value` and `word`. 
Value is a number, a string, a number tuple, a tensor or a tensor tuple. 
Every 'value' will be pushed into stack as the order of code, `word` is a function (built-in or user defined) consumes values, pops them from stack, and push output into stack.
YaNNX provides two built-in word 'set' and 'get' to handle hashmap of global or local in user defined word, hashmap's key is a constant string and value is the 'value'.
The '@' is just a short-cut for 'get' word. Parentheses in code is meaningless, is just separator for readable. 
The quare brackets is used for create tuple value, such as number tuple, is only used in code parsing phase.

We provided a REPL ( Read, Evaluate, Print and Loop ) tool for just debug or testing YaNNX model files.

The tensor's data type is followed with ONNX, and high level type is variable tensor, constant tensor, parameter tensor. The constant tensor's data is stored in a msgpack file. 
The built-in operator creates constant tensors when they are first executed, the created constant tensor will registered in runtime.
The list of all constant tensors can be accessed by runtime's API, the order in this list is same as executing code. 
The parameter tensor is used for training in the future.

We removed built-in control operators, such as `if`, `for loop` etc, so why we called YaNNX is semi-dynammic, it is not full programed. 
We also don't provided basic operator for number, string or tuple, YaNNX is not a language is just a exprssion for nerual network. 

## User define wrod/function

YaNNX support user defined word, every word appreas in the code will has it's own hashmap, which give us a closure mechanism. 
Like the example, we can define a `FullConnect` user defined word based on basic operator 'onnx.gemm', the constants were created when `FullConnect` is first invoked.
All the word name end with '~' means this operator only run once, and from second invoke it just pop input and push last output.

## The runtime of YaNNX

We provided public API in runtime of YaNNX, these API is the external interfaces for outside world of YaNNX. 
Thess API includs: adding a new built-in operator, accessing the global hashmap or list of constant tensor.
`yannx.hpp` is all-in-one c++ head file for extenstion in your project, is a already parser, checker and base c++ runtime classes for writing your own backend or tools.

We provides a REPL with dummy backend, a onnx convert and others, these tools are developed all based on `yaonnx.hpp` certainly. We provides a YaNNX model zoo also.


