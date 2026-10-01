CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
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

test: tests/test_vectors tests/test_stream
	./tests/test_vectors
	./tests/test_stream tests/data

tools/bench: tools/bench.c tandem.c tandem.h
	$(CC) $(CFLAGS) -o $@ tools/bench.c tandem.c

bench: tools/bench
	./tools/bench

# Regenerate the vector header from a checkout of https://github.com/tandem-rng/spec.
vectors:
	python3 tools/gen_vectors.py $(SPEC_VECTORS) > tests/vectors.h

clean:
	rm -f tandem.o libtandem.a tests/test_vectors tests/test_stream tools/bench
