CC = clang++
CFLAGS = -std=c++11 -lOpenCL -g


all: build_dir Add

build_dir:
	mkdir -p ./build

%: %.cpp
	$(CC) $(CFLAGS) -o build/$@ $<

clean:
	rm -fv *.o
	rm -fv build/*
