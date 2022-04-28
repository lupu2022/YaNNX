#!/bin/bash

function env {
   export ONNX_PATH=/home/teaonly/opt/onnx
   export PROTOBUF_PATH=/home/teaonly/opt/protobuf

   export DNNL_PATH=/home/teaonly/opt/dnnl
   export CUDA_PATH=/usr/local/cuda
   export CUDNN_PATH=/usr/local/cuda
}

$1
