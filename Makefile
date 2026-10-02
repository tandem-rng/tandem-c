CC ?= cc
CXX ?= c++
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wshadow
SPEC_VECTORS ?= ../tandem-spec/vectors.json

.PHONY: all test vectors bench clean

all: libtandem.a

tandem.o: tandem.c tandem.h
	$(CC) $(CFLAGS) -c -o $@ tandem.c

libtandem.a: tandem.o
	$(AR) rcs $@ $^

tests/test_vectors: tests/test_vectors.c tests/vectors.h tandem.c tandem.h
	$(CC) $(CFLAGS) -o $@ tests/test_vectors.c tandem.c

tests/test_stream: tests/test_stream.c tandem.c tandem.h
	$(CC) $(CFLAGS) -o $@ tests/test_stream.c tandem.c

tests/test_cpp: tests/test_cpp.cpp tandem.hpp tandem.o
	$(CXX) $(CXXFLAGS) -o $@ tests/test_cpp.cpp tandem.o

test: tests/test_vectors tests/test_stream tests/test_cpp
	./tests/test_vectors
	./tests/test_stream tests/data
	./tests/test_cpp

tools/bench: tools/bench.c tandem.c tandem.h
	$(CC) $(CFLAGS) -o $@ tools/bench.c tandem.c

tools/bench_std: tools/bench_std.cpp tandem.hpp tandem.o
	$(CXX) $(CXXFLAGS) -o $@ tools/bench_std.cpp tandem.o

bench: tools/bench tools/bench_std
	./tools/bench
	./tools/bench_std

# Regenerate the vector header from a checkout of https://github.com/tandem-rng/spec.
vectors:
	python3 tools/gen_vectors.py $(SPEC_VECTORS) > tests/vectors.h

clean:
	rm -f tandem.o libtandem.a tests/test_vectors tests/test_stream tests/test_cpp tools/bench tools/bench_std
