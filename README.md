# YaNNX
Yet Another Neural Network Exchange (YaNNX) is an open source format for AI models. 

YaNNX uses a stack expression to create a DAG of neural network some likes Forth language, so it is a semi-dynamic DSL for deep learning.
YaNNX supports user defined functions to descripting sub DAG, so YaNNX's model has a hierarchical structure and is easy readable for users, 
and these hierarchical structure is important for subsequent optimization in diffrent platfrom.

The data type and the operator's schemas are all inherited from ONNX project, the type and shape infrence functions are also used in YaNNX's tools.
YaNNX's runtime includes a global and local hash map (for every UDF) to storing or naming tensor，which give us more flexibility to descripting complex network.
The stack expression or language of YaNNX is easily to been implemented, it just is a few stack operators like 'pop', 'swap', 'rot' etc, and hash operator 'set' and 'get'.



