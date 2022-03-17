# YaNNX
Yet Another Neural Network Exchange (YaNNX) is an open source format for AI models. 

YaNNX uses a stack expression to create a DAG of neural network some likes Forth language, so it is a semi-dynamic DSL for deep learning.
YaNNX supports user defined functions to descripting sub DAG, so YaNNX's model has a hierarchical structure and is easy readable for users.
The data type and operator's schemas are all inherited from ONNX project, the type and shape infrence functions are also used in YaNNX. 
YaNNX also includes a global and local hash map (for UDF) to storing or naming tensor，which give us more flexibility to descripting complex network.

The stack machine of YaNNX is easily to been implement, it has a few built-in operators like 'pop', 'swap', 'rot' etc.
