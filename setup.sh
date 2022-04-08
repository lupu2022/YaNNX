#!/bin/bash

function env {
   export ONNX_PATH=/home/teaonly/opt/onnx
   export PROTOBUF_PATH=/home/teaonly/opt/protobuf
}

$1
