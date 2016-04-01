include user.mk

CFLAGS += -std=c++11 -lOpenCL -Wall -I/usr/local/include/

BUILDDIR ?= ./build

all: build_dir Add WeakLeaner JointLeaner

.PHONY: build_dir
build_dir:
	mkdir -p ${BUILDDIR}

%: %.cpp
	$(CXX) $(CFLAGS) -o $(BUILDDIR)/$@ $<

.PHONY: clean
clean:
	rm -fv *.o
	rm -fv ${BUILDDIR}/*

.PHONY: test-add
test-add:
	$(BUILDDIR)/Add

.PHONY: test-weakleaner
test-weakleaner:
	$(BUILDDIR)/WeakLeaner

.PHONY: test-jointleaner
test-jointleaner:
	$(BUILDDIR)/JointLeaner

.PHONY: test
test: test-add test-weakleaner
	@:
