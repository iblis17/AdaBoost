include user.mk

CC ?= clang++
CFLAGS += -std=c++11 -lOpenCL -Wall -I/usr/local/include/

BUILDDIR ?= ./build

all: build_dir Add WeakLeaner

build_dir:
	mkdir -p ${BUILDDIR}

%: %.cpp
	$(CC) $(CFLAGS) -o $(BUILDDIR)/$@ $<

clean:
	rm -fv *.o
	rm -fv ${BUILDDIR}/*

test-add:
	$(BUILDDIR)/Add

test-weakleaner:
	$(BUILDDIR)/WeakLeaner

test: test-add test-weakleaner
	@:
