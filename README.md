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

There is a Resnet34 YaNNX model file, the whole architecture is very clear. 

```
;
;   basic operators definement
;
def conv1x1
    "stride" !
    "out_size" !
    "in_size" !

    ; W
    ["out_size" @ "in_size" @ 3 3 ] "YNX_FLOAT" ynx.NewTensor 
    ; B
    none

    ; auto_pad, dilation, group, kernel, pads, strides
    "NOTSET" [1 1] 1 [1 1] [0 0 0 0] ["stride" @ dup]             
    onnx.Conv 
end

def conv3x3
    "stride" !
    "out_size" !
    "in_size" !

    ; W
    ["out_size" @ "in_size" @ 3 3 ] "YNX_FLOAT" ynx.NewTensor 
    ; B
    none

    ; auto_pad, dilation, group, kernel, pads, strides
    "NOTSET" [1 1] 1 [3 3] [1 1 1 1] ["stride" @ dup]             
    onnx.Conv 
end

def conv7x7
    ; W
    [64 3 7 7 ] "YNX_FLOAT" ynx.NewTensor 
    ; B
    none

    ; auto_pad, dilation, group, kernel, pads, strides
    "NOTSET" [1 1] 1 [7 7] [3 3 3 3] [2 2]             
    onnx.Conv 
end

def batch_norm
    "out_feature" !

    ["out_feature" @] "YNX_FLOAT" ynx.NewTensor     ; scale
    ["out_feature" @] "YNX_FLOAT" ynx.NewTensor     ; B
    ["out_feature" @] "YNX_FLOAT" ynx.NewTensor     ; input_mean
    ["out_feature" @] "YNX_FLOAT" ynx.NewTensor     ; input_var
    none none none false false
    
    onnx.BatchNormalization
end

def max_pool2d
    "NOTSET" 0 [1 1]            ; auto_pad, ceil_Mode, dilation
    [3 3]                       ; kernel_shape
    [0 0 0 0]                   ; pads
    0                           ; storage_order
    [2 2]                       ; strides
    false                       ; we don't want Indices
    
    onnx.MaxPool
end

def avg_pool2d
    "NOTSET" 0 0                ; auto_pad, ceil_Mode, count_include_pad
    [7 7]                       ; kernel_shape
    [0 0 0 0]                   ; pads
    [1 1]                       ; strides
     
    onnx.AveragePool
end

def linear
    "out_size" !
    "in_size" !
    
    ; W
    ["in_size" @ "out_size" @ ] "YNX_FLOAT" ynx.NewTensor 
    onnx.MatMul

    ; B
    ["out_size" @] "YNX_FLOAT" ynx.NewTensor                                      
    onnx.Add
end

;
; the components fro ResNet
;
def BasicBlock_DownSample
    "outplanes" !
    "inplanes" !

    dup
    "inplanes" @ "outplanes" @ 2 conv1x1
    "outplanes" @ batch_norm
   
    swap
    "inplanes" @ "outplanes" @ 2 conv3x3
    "outplanes" @ batch_norm
    onnx.Relu
    "outplanes" @ dup 1 conv3x3
    "outplanes" @ batch_norm

    onnx.Add
    onnx.Relu
end

def BasicBlock
    "outplanes" !
    "inplanes" !

    "inplanes" @ "outplanes" @ 1 conv3x3
    "outplanes" @ batch_norm
    onnx.Relu
    "outplanes" @ dup 1 conv3x3
    "outplanes" @ batch_norm
end

def ResLayer0
    conv7x7
    64 batch_norm
    onnx.Relu
    max_pool2d
end

;
; completed network
;
def Resnet34
    ResLayer0                       ; layer 0
   
    { 64 64 BasicBlock } %3         ; layer 1

    64 128 BasicBlock_DownSample    ; layer 2
    { 128 128 BasicBlock } %3

    128 256 BasicBlock_DownSample   ; layer 3
    { 256 256 BasicBlock } %5       

    256 512 BasicBlock_DownSample   ; layer 4
    { 512 512 BasicBlock } %2

    ; output
    avg_pool2d
    1 onnx.Flatten
    512 1000 linear   
end

```

### The base syntax of YaNNX model file

The syntax of YaNNX is a stack expression, the base item of expresstion is `value` and `word`. 
`Value` is a number , a string, a number tuple, a tensor or a tensor tuple. 
Every `value` will be pushed into stack as the order of code, `word` is a function (built-in or user defined) consumes values, pops them from stack, and push output into stack back.
YaNNX provides two built-in words `set` and `get` to write and read hashmap ( global or local in user defined word), the hashmaps' key is a constant string and value is the `value`.
The word `@` is just a short-cut for 'get' word. Parentheses in code is meaningless, is just a separator for more readable. 
The square brackets is used for creating tuple value, such as number tuple, the tuple is used for tensor shape , padding list etc.

We provided a REPL (Read, Evaluate, Print and Loop) tool for debug or test YaNNX model files.

The tensor's data type is followed with ONNX, and high level classes is variable tensor, constant tensor, parameter tensor. The constant tensor's data is stored in a isolate msgpack file. 
The built-in operators (such as `ynx.NewTensor` `ynx.NewConstant` ) create tensors when they are first executed, the created constant tensor will registered in YaNNX's runtime.
The list of all constant tensors can be accessed by the runtime's API, the order in this list is same as created or excuted.
The parameter tensor is used for training in the future.

We removed built-in control operators (which are included in Forth language), such as `if`, `for loop` etc, so why we called YaNNX is semi-dynammic, it is not fully programable. We also provide a loop macro `{...} %loop`, it is only a macro repeated text in `{}`, see previous examples.

We only provided limited operators for `value`, YaNNX is not a language is just a exprssion for nerual network. 
But adding these function into YaNNX is easy, you can do it in your project's enviroment.

### User define word

YaNNX supports user defined word ( by using `def` and `end`), every word appreas in the code will has it's own hashmap, which give us a closure mechanism to hidden some local tensors .
In a morden deep learning framework, we creates a neural network by python code, hierarchical structure and repeated modules is obviously, keep this structure is exchange mode is also important.
High level hierarchical structure will help subsequent optimization , and help generating structured python code for diffirent framwork.

ONNX model don't provides these hierarchical structure, and the DAG is just a tiled list for which analysis is difficult, and it is not readable of course.

Like the example, we can define a `linear` user defined word based on basic operator `onnx.MatMul`, the constants were created when `linear` is first invoked.
All the word name end with '~' means this operator only run once, and from second call it just pop input and push last output.

### The runtime of YaNNX

We provided public c++ API in runtime class of YaNNX, these API is the external interfaces for outside world of YaNNX. 
Thess API includs: adding a new built-in operator, accessing the global hashmap or listing constant tensors, etc.
`yannx.hpp` is a only-one c++ head file included in your project, is a already parser, checker and base c++ framework for writing your own backend or tools. 

We provides a REPL tool with a dummy backend, a onnx convert tool and others, these tools are developed all based on `yaonnx.hpp` certainly. 
At last, we provides a YaNNX model zoo also, all the models in zoo are readable and well structured.

## 2. Building YaNNX's tools

`setup.sh` provides basic third party libraries' path info, including protobuf and onnx. 
The `tensortype.hpp` is auto generated from `onnx~`, we have implemented a basic frameowork for your real tensor.
The apis of `tensortype.hpp` are from onnx's definment, the tensor caculation api is exactly following onnx.

`yannx.hpp` provides basic Yannx core, it is a simple Forth like language's parser and compiler. 
The stack structure is very effective, it is easy extended with diffirent object.

### The repl tool of YaNNX

The repl tool is based on onnx's internal stuff, which supports tensor type&shape inference only and don't include real tensor implementation.
Yannx core provides two API , `boot` and `forward` , `boot` is used first time, which provides Yannx's internal hash, but when call `forward`, there is only internal stack which has more efficiency.
Following is the example of REPL.

```
./repl ../../examples/resnet.yannx
>> b test
boostrap: test
>> b ??
boostrap: ??
--- stack top ------
0:      <tensor>:YNX_FLOAT:[4 1000]
---- bottom --------
>> b test
boostrap: test
>> f
time: 3
>> f
time: 3
>> f
time: 3
>> b ??
boostrap: ??
--- stack top ------
0:      <tensor>:YNX_FLOAT:[4 1000]
1:      <tensor>:YNX_FLOAT:[4 1000]
2:      <tensor>:YNX_FLOAT:[4 1000]
3:      <tensor>:YNX_FLOAT:[4 1000]
4:      <tensor>:YNX_FLOAT:[4 1000]
---- bottom --------
```

