#!/bin/bash

function env {
   export ONNX_PATH=/home/teaonly/opt/onnx
   export PROTOBUF_PATH=/home/teaonly/opt/protobuf

   export DNNL_PATH=/home/teaonly/opt/dnnl
   export CUDA_PATH=/usr/local/cuda
   export CUDNN_PATH=/usr/local/cuda
}

function install {
    prefix=/home/teaonly/opt/yannx
    export YANNX_PATH=$prefix
    rm -rf $prefix
    
    mkdir -p $prefix
    mkdir -p $prefix/dnnl
    mkdir -p $prefix/autogen
    
    cp ./dnnl/* $prefix/dnnl/
    cp ./autogen/*.inc $prefix/autogen/
    cp yannx.hpp tensortype.hpp opwords.hpp $prefix/ 
}

$1
