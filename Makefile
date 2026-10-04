CC ?= cc
CXX ?= c++
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wshadow
SPEC_VECTORS ?= ../tandem-spec/vectors.json
LDLIBS ?= -lm
CORE_INCLUDE ?= ../tandem-cuda/include

.PHONY: all test vectors cross bench clean

all: libtandem.a

tandem.o: tandem.c tandem.h
	$(CC) $(CFLAGS) -c -o $@ tandem.c

libtandem.a: tandem.o
	$(AR) rcs $@ $^

tests/test_vectors: tests/test_vectors.c tests/vectors.h tandem.c tandem.h
	$(CC) $(CFLAGS) -o $@ tests/test_vectors.c tandem.c $(LDLIBS)

tests/test_stream: tests/test_stream.c tandem.c tandem.h
	$(CC) $(CFLAGS) -o $@ tests/test_stream.c tandem.c $(LDLIBS)

tests/test_api: tests/test_api.c tests/cross_below.h tests/cross_normal.h tandem.c tandem.h
	$(CC) $(CFLAGS) -o $@ tests/test_api.c tandem.c $(LDLIBS)

tests/test_cpp: tests/test_cpp.cpp tandem.hpp tandem.o
	$(CXX) $(CXXFLAGS) -o $@ tests/test_cpp.cpp tandem.o $(LDLIBS)

test: tests/test_vectors tests/test_stream tests/test_api tests/test_cpp
	./tests/test_vectors
	./tests/test_stream tests/data
	./tests/test_api
	./tests/test_cpp

tools/bench: tools/bench.c tandem.c tandem.h
	$(CC) $(CFLAGS) -o $@ tools/bench.c tandem.c $(LDLIBS)

tools/bench_std: tools/bench_std.cpp tandem.hpp tandem.o
	$(CXX) $(CXXFLAGS) -o $@ tools/bench_std.cpp tandem.o $(LDLIBS)

bench: tools/bench tools/bench_std
	./tools/bench
	./tools/bench_std

# Regenerate the vector header from a checkout of https://github.com/tandem-rng/spec.
vectors:
	python3 tools/gen_vectors.py $(SPEC_VECTORS) > tests/vectors.h

# Regenerate the cross-check fixtures from a checkout of https://github.com/tandem-rng/tandem-cuda.
tools/gen_cross: tools/gen_cross.cpp
	$(CXX) $(CXXFLAGS) -I$(CORE_INCLUDE) -o $@ tools/gen_cross.cpp

cross: tools/gen_cross
	./tools/gen_cross below > tests/cross_below.h
	./tools/gen_cross normal > tests/cross_normal.h

clean:
	rm -f tandem.o libtandem.a tests/test_vectors tests/test_stream tests/test_api tests/test_cpp tools/bench tools/bench_std tools/gen_cross
