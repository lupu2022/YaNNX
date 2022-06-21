.PHONY: all

TARGET = repl

FLAGS = -std=c++17 -Wall -g -DONNX_NAMESPACE=onnx -DONNX_ML=0 -D_USE_DNNL_CPU_ 
INC = -Imsgpack -I$(YANNX_PATH) -I${ONNX_PATH}/include -I$(PROTOBUF_PATH)/include -I$(DNNL_PATH)/include 
LINK = -L${ONNX_PATH}/lib  -L${PROTOBUF_PATH}/lib -L$(DNNL_PATH)/lib -lonnx -lonnx_proto -lprotobuf -ldnnl -lpthread 

all: $(TARGET)

$(TARGET): repl.cpp 
	g++ $(FLAGS) $^ $(INC) $(LINK) -o $@  

run: $(TARGET)
	LD_LIBRARY_PATH=$(ONNX_PATH)/lib:$(PROTOBUF_PATH)/lib:$(DNNL_PATH)/lib:$LD_LIBRARY_PATH gdb ./$(TARGET) 

clean:
	rm -f $(TARGET)
