# Makefile to build scipy_fastmin extensions and build + execute gtest tests.

# package name
PKG_NAME       = scipy_fastmin
# directory for libgtest test runner code
GTEST_DIR      = gtest
# C and C++ compilers, of course
CC             = gcc
CXX            = g++
# C extension module dependencies
XDEPS          = \
	$(wildcard $(PKG_NAME)/utils/*.c) $(wildcard $(PKG_NAME)/solvers/*.c) \
	$(wildcard $(PKG_NAME)/include/*.h)
# Python source (directories will be searched with vpath)
PYDEPS         = \
	$(wildcard $(PKG_NAME)/*.py) $(wildcard $(PKG_NAME)/utils/*.py) \
	$(wildcard $(PKG_NAME)/solvers/*.py)
# dependencies for test running code
GTEST_DEPS     = $(wildcard $(GTEST_DIR)/*.cc)
# set python; on docker specify PYTHON value externally using absolute path
PYTHON        ?= python3
# flags to pass to setup.py build
BUILD_FLAGS    =
# directory to save distributions to; use absolute path on docker
DIST_FLAGS    ?= --dist-dir ./dist
# python compiler and linker flags for use when linking python into external
# C/C++ code; can be externally specified. gcc/g++ requires -fPIE.
PY_CFLAGS     ?= -fPIE $(shell python3-config --cflags)
# ubuntu needs --embed, else -lpythonx.y is omitted by --ldflags, which is a
# linker error. libpython3.8 is in /usr/lib/x86_64-linux-gnu for me.
PY_LDFLAGS    ?= $(shell python3-config --embed --ldflags)
# g++ compile flags for gtest runner. my libgtest.so is in /usr/local/lib.
GTEST_CFLAGS   = $(PY_CFLAGS) -I$(PKG_NAME)/include
# g++ linker flags for compiling gtest runner
GTEST_LDFLAGS  = $(PY_LDFLAGS) -lgtest -lgtest_main
# flags to pass to the gtest test runner
RUNNER_FLAGS   =

# phony targets
.PHONY: clean dummy dist

# triggered if no target is provided
dummy:
	@echo "Please specify a target to build."

# removes local build, dist, egg-info
clean:
	@rm -vf *~
	@rm -vrf build
	@rm -vrf $(PKG_NAME).egg-info
	@rm -vrf dist

# build extension module locally in ./build from source files with setup.py
# triggers when any of the files that are required are touched/modified.
build: $(XDEPS) $(PYDEPS)
	@$(PYTHON) setup.py build $(BUILD_FLAGS)

# build extension modules in-place with build_ext --inplace. in-place means
# the shared objects will be in the same directory as the respective sources.
inplace: $(XDEPS)
	@$(PYTHON) setup.py build_ext --inplace $(BUILD_FLAGS)

# build test runner and run gtest unit tests. show flags passed to g++
check: $(GTEST_DEPS) inplace
	$(CXX) $(GTEST_CFLAGS) -o runner $(GTEST_DEPS) $(GTEST_LDFLAGS)
	@./runner $(RUNNER_FLAGS)

# make source and wheel
dist: build
	@$(PYTHON) setup.py sdist bdist_wheel $(DIST_FLAGS)