.PHONY: all

TARGET = repl

FLAGS = -std=c++17 -Wall -g
INC = -I../..
LINK = 

all: $(TARGET) 

$(TARGET): repl.cpp
	g++ $(FLAGS) $(INC) -o $@ $^ $(LINK) 

clean:
	rm -f repl
