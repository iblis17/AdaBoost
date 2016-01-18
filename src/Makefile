CC = clang++
CFLAGS = -std=c++11 -lOpenCL -g -Wall -DDEBUG -I/usr/local/include/

BUILDDIR = build

all: build_dir Add WeakLeaner

build_dir:
	mkdir -p ./build

%: %.cpp
	$(CC) $(CFLAGS) -o $(BUILDDIR)/$@ $<

clean:
	rm -fv *.o
	rm -fv build/*

test-add:
	$(BUILDDIR)/Add

test-weakleaner:
	$(BUILDDIR)/WeakLeaner

test: test-add test-weakleaner
	@:
