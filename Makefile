# Clang is the primary compiler. make predefines CC as cc, so ?= would never apply.
ifeq ($(origin CC),default)
CC = clang
endif
ifeq ($(origin CXX),default)
CXX = clang++
endif
# -ffp-contract=off keeps every expression outside the explicit fused multiply-adds unfused, so
# all compilers produce the same normals. x86 needs no -m flags: tandem.c picks its AVX2 and FMA
# copy at run time.
CFLAGS ?= -std=c23 -O2 -ffp-contract=off -Wall -Wextra -Wpedantic -Wconversion -Wshadow
CXXFLAGS ?= -std=c++23 -O2 -Wall -Wextra -Wpedantic -Wshadow
SPEC_VECTORS ?= ../tandem-spec/vectors.json
SPEC_TABLES ?= ../tandem-spec/tables/normal_f64_zig1024.json
LDLIBS ?= -lm
CORE_INCLUDE ?= ../tandem-cuda/include
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
VERSION ?= 0.1.0

.PHONY: all test stats vectors tables cross bench accuracy test-target bench-target install clean

all: libtandem.a

tandem.o: tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) -c -o $@ tandem.c

libtandem.a: tandem.o
	$(AR) rcs $@ $^

tests/test_vectors: tests/test_vectors.c tests/vectors.h tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) -o $@ tests/test_vectors.c tandem.c $(LDLIBS)

tests/test_stream: tests/test_stream.c tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) -o $@ tests/test_stream.c tandem.c $(LDLIBS)

tests/test_api: tests/test_api.c tests/cross_below.h tests/cross_exponential.h tests/cross_fill_below.h tests/cross_normal.h tests/cuda_fill_below.h tests/cuda_fill_normal.h tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) -o $@ tests/test_api.c tandem.c $(LDLIBS)

tests/test_r123: tests/test_r123.c tandem123.h tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) -o $@ tests/test_r123.c tandem.c $(LDLIBS)

# OpenMP target offload, see tandem_target.c. OMP_FLAGS selects the compiler's offload flags:
#   clang: -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80
#   nvc:   -mp=gpu -gpu=cc80, with CC=nvc and CFLAGS="-std=c11 -O2"
# The default runs the target regions on the host, with host memory.
OMP_FLAGS ?= -fopenmp -DTANDEM_HOST_MEMORY

tests/test_target: tests/test_target.c tandem_target.c tandem123.h tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) -DTANDEM_OPENMP_TARGET -o $@ tests/test_target.c tandem_target.c tandem.c $(LDLIBS)

tools/bench_target: tools/bench_target.c tandem_target.c tandem123.h tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) -DTANDEM_OPENMP_TARGET -o $@ tools/bench_target.c tandem_target.c tandem.c $(LDLIBS)

test-target: tests/test_target
	./tests/test_target

bench-target: tools/bench_target
	./tools/bench_target

tests/test_normal_bits: tests/test_normal_bits.c tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) -o $@ tests/test_normal_bits.c tandem.c $(LDLIBS)

tests/test_exponential_bits: tests/test_exponential_bits.c tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) -o $@ tests/test_exponential_bits.c tandem.c $(LDLIBS)

tests/test_cpp: tests/test_cpp.cpp tandem.hpp tandem.o
	$(CXX) $(CXXFLAGS) -o $@ tests/test_cpp.cpp tandem.o $(LDLIBS)

test: tests/test_vectors tests/test_stream tests/test_api tests/test_r123 tests/test_normal_bits tests/test_exponential_bits tests/test_cpp
	./tests/test_vectors
	./tests/test_stream tests/data
	./tests/test_api
	./tests/test_r123 tests/data
	./tests/test_normal_bits
	./tests/test_exponential_bits
	./tests/test_cpp

# The distribution of 10^8 f64 normals. It takes about half a minute, so it runs apart from test.
tests/test_normal_stats: tests/test_normal_stats.c tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) -o $@ tests/test_normal_stats.c tandem.c $(LDLIBS)

stats: tests/test_normal_stats
	./tests/test_normal_stats

tools/bench:tools/bench.c tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) -o $@ tools/bench.c tandem.c $(LDLIBS)

tools/bench_std: tools/bench_std.cpp tandem.hpp tandem.o
	$(CXX) $(CXXFLAGS) -o $@ tools/bench_std.cpp tandem.o $(LDLIBS)

tools/normal_accuracy: tools/normal_accuracy.c tandem.c tandem.h tandem_normal_tables.h
	$(CC) $(CFLAGS) -o $@ tools/normal_accuracy.c tandem.c $(LDLIBS)

# Random123 sits on the system include path so its own warnings stay quiet.
tools/bench_philox: tools/bench_philox.c
	$(CC) $(CFLAGS) -isystem tools/random123 -o $@ tools/bench_philox.c $(LDLIBS)

bench: tools/bench tools/bench_std tools/bench_philox
	./tools/bench
	./tools/bench_std
	./tools/bench_philox

# Regenerate the vector header from a checkout of https://github.com/tandem-rng/spec.
vectors:
	python3 tools/gen_vectors.py $(SPEC_VECTORS) > tests/vectors.h

# Regenerate the ziggurat tables of the f64 normals from the same checkout.
tables:
	python3 tools/gen_zig_tables.py $(SPEC_TABLES) > tandem_normal_tables.h

accuracy: tools/normal_accuracy
	./tools/normal_accuracy

# Regenerate the cross-check fixtures from a checkout of https://github.com/tandem-rng/tandem-cuda.
tools/gen_cross: tools/gen_cross.cpp tandem_normal_tables.h tandem.o
	$(CXX) $(CXXFLAGS) -I$(CORE_INCLUDE) -o $@ tools/gen_cross.cpp tandem.o $(LDLIBS)

cross: tools/gen_cross
	./tools/gen_cross below > tests/cross_below.h
	./tools/gen_cross fill_below > tests/cross_fill_below.h
	./tools/gen_cross normal > tests/cross_normal.h
	./tools/gen_cross exponential > tests/cross_exponential.h
	cp $(CORE_INCLUDE)/../tests/cross_fill_below.h tests/cuda_fill_below.h
	cp $(CORE_INCLUDE)/../tests/cross_fill_normal.h tests/cuda_fill_normal.h

# DESTDIR stages the files for packagers. The paths inside tandem.pc ignore it.
install: libtandem.a
	install -d $(DESTDIR)$(LIBDIR)/pkgconfig $(DESTDIR)$(INCLUDEDIR)
	install -m 644 libtandem.a $(DESTDIR)$(LIBDIR)
	install -m 644 tandem.h tandem.hpp $(DESTDIR)$(INCLUDEDIR)
	sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@LIBDIR@|$(LIBDIR)|' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|' \
	    -e 's|@VERSION@|$(VERSION)|' packaging/tandem.pc.in > $(DESTDIR)$(LIBDIR)/pkgconfig/tandem.pc

clean:
	rm -f tandem.o libtandem.a tests/test_vectors tests/test_stream tests/test_api tests/test_r123 tests/test_normal_bits tests/test_exponential_bits tests/test_cpp tests/test_normal_stats tools/bench tools/bench_std tools/bench_philox tools/gen_cross tools/normal_accuracy tests/test_target tools/bench_target
