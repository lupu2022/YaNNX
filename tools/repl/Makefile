.PHONY: all

TARGET = repl

FLAGS = -std=c++17 -Wall -O3 -DONNX_NAMESPACE=onnx -DONNX_ML=0 
INC = -I../.. -I${ONNX_PATH}/include -I$(PROTOBUF_PATH)/include
LINK = -L${ONNX_PATH}/lib  -L${PROTOBUF_PATH}/lib -lonnx -lonnx_proto -lprotobuf -lpthread 

all: $(TARGET)

$(TARGET): repl.cpp ../../yannx.hpp ../../tensortype.hpp
	g++ $(FLAGS) -o $@ $^ $(INC) $(LINK) 

run: $(TARGET)
	LD_LIBRARY_PATH=$(ONNX_PATH)/lib:$(PROTOBUF_PATH)/lib:$LD_LIBRARY_PATH ./$(TARGET) ../../examples/alexnet.yannx

clean:
	rm -f $(TARGET)
