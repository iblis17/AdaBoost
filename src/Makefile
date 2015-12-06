CC = clang++
CFLAGS = -std=c++11 -lOpenCL -g -Wall -DDEBUG


all: build_dir Add WeakLeaner

build_dir:
	mkdir -p ./build

%: %.cpp
	$(CC) $(CFLAGS) -o build/$@ $<

clean:
	rm -fv *.o
	rm -fv build/*
