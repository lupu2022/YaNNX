# YaNNX
Yet Another Neural Network Exchange (YaNNX) is an open source format for AI models. 

## 1.Intrduction

YaNNX uses a stack expression to create DAG of neural network some likes Forth language, it is a semi-dynamic DSL for describing nerual network.
YaNNX supports user defined word (UDW for short, just like functions) to expressing sub DAG, so YaNNX's model has a hierarchical or modular structure and is easy readable for users.
Keeping these hierarchical structure is important for subsequent optimization in diffrent platfrom. 

Most morden neural network models have modular structure , for example, typical ResNet has layer-1 to layer-3 which shared same structure. 
The ONNX's solution is based on a long tiled node list which losts the modular structure, and it is difficult resotre the original structure also. 
So we create a new solution YaNNX based on stack machine which don't need naming every link between nodes, and introduce user defined word which will give us the ablilty of keep modular structure.

The data type and the operator schemas are all inherited from ONNX project, the type and shape infrence functions are also used in the YaNNX's tools.
The YaNNX's runtime includes a global and local hash map (for every UDW) to storing or naming tensor，which give us more flexibility to descripting complex network.
The stack expression or language of YaNNX is easily to been implemented, it just is a few stack operators like `pop`, `swap`, `rot` etc, and hash operator `set` and `get`.

There is a simple YaNNX model file, the constant weights is another msgpack format file which not show here.

```
def FullConnect
    "out_size" set
    "in_size"  set

    ("out_size" @ "in_size" @ 2 tuple) .new_constant~
    ("out_size" @ 1 tuple) .new_constant~ 
    
    ; now stack = x, w, b
    1.0 1.0 onnx.Gemm           ; call built-in onnx operator
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

### The base syntax of YaNNX model file

The syntax of YaNNX is a stack expression, the base item of expresstion is `value` and `word`. 
`Value` is a number , a string, a number tuple, a tensor or a tensor tuple. 
Every `value` will be pushed into stack as the order of code, `word` is a function (built-in or user defined) consumes values, pops them from stack, and push output into stack back.
YaNNX provides two built-in words `set` and `get` to write and read hashmap ( global or local in user defined word), the hashmaps' key is a constant string and value is the `value`.
The word `@` is just a short-cut for 'get' word. Parentheses in code is meaningless, is just a separator for more readable. 
The square brackets is used for creating tuple value, such as number tuple, is only used in code parsing phase.

We provided a REPL (Read, Evaluate, Print and Loop) tool for debug or test YaNNX model files.

The tensor's data type is followed with ONNX, and high level classes is variable tensor, constant tensor, parameter tensor. The constant tensor's data is stored in a isolate msgpack file. 
The built-in operators (such as `.new_tensor~` `.new_constant~` ) create tensors when they are first executed, the created constant tensor will registered in YaNNX's runtime.
The list of all constant tensors can be accessed by the runtime's API, the order in this list is same as created or excuted.
The parameter tensor is used for training in the future.

We removed built-in control operators (which are included in Forth language), such as `if`, `for loop` etc, so why we called YaNNX is semi-dynammic, it is not fully programable. 
We only provided limited operators for `value`, YaNNX is not a language is just a exprssion for nerual network. 
But adding these function into YaNNX is easy, you can do it in your project's enviroment.

### User define word

YaNNX supports user defined word ( by using `def` and `end`), every word appreas in the code will has it's own hashmap, which give us a closure mechanism to hidden some local tensors .
In a morden deep learning framework, we creates a neural network by python code, hierarchical structure and repeated modules is obviously, keep this structure is exchange mode is also important.
High level hierarchical structure will help subsequent optimization , and help generating structured python code for diffirent framwork.

ONNX model don't provides these hierarchical structure, and the DAG is just a tiled list for which analysis is difficult, and it is not readable of course.

Like the example, we can define a `FullConnect` user defined word based on basic operator `onnx.gemm`, the constants were created when `FullConnect` is first invoked.
All the word name end with '~' means this operator only run once, and from second call it just pop input and push last output.

### The runtime of YaNNX

We provided public c++ API in runtime class of YaNNX, these API is the external interfaces for outside world of YaNNX. 
Thess API includs: adding a new built-in operator, accessing the global hashmap or listing constant tensors, etc.
`yannx.hpp` is a only-one c++ head file included in your project, is a already parser, checker and base c++ framework for writing your own backend or tools. 

We provides a REPL tool with a dummy backend, a onnx convert tool and others, these tools are developed all based on `yaonnx.hpp` certainly. 
At last, we provides a YaNNX model zoo also, all the models in zoo are readable and well structured.

## 2. Building YaNNX's tools
