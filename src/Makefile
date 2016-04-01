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
test-add: Add
	$(BUILDDIR)/Add

.PHONY: test-weakleaner
test-weakleaner: WeakLeaner
	$(BUILDDIR)/WeakLeaner

.PHONY: test-jointleaner
test-jointleaner: JointLeaner
	$(BUILDDIR)/JointLeaner

.PHONY: test
test: test-add test-weakleaner
	@:
