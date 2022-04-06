.PHONY: all

TARGET = repl

FLAGS = -std=c++14 -Wall -g
INC = -I../..
LINK = 

all: $(TARGET) 

$(TARGET): repl.cpp
	g++ $(FLAGS) $(INC) -o $@ $^ $(LINK) 

clean:
	rm -f repl
