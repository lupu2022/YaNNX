.PHONY: all

TARGET = autogen

FLAGS = -std=c++17 -Wall -O3 -DONNX_NAMESPACE=onnx -DONNX_ML=0 
INC = -I. -I${ONNX_PATH}/include -I$(PROTOBUF_PATH)/include
LINK = -L${ONNX_PATH}/lib  -L${PROTOBUF_PATH}/lib -lonnx -lonnx_proto -lprotobuf -lpthread 

all: $(TARGET)

$(TARGET): autogen.cpp
	g++ $(FLAGS) -o $@ $^ $(INC) $(LINK) 

install: $(TARGET)
	LD_LIBRARY_PATH=$(ONNX_PATH)/lib:$(PROTOBUF_PATH)/lib:$LD_LIBRARY_PATH ./autogen
	mv tensortype.hpp ../	
	
clean:
	rm -f $(TARGET)
	rm -f tensortype.hpp
