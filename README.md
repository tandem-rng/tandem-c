<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .c"></p>

# tandem-c

[![CI](https://github.com/tandem-rng/tandem-c/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-c/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tandem--rng.github.io-7fb3ee.svg)](https://tandem-rng.github.io/tandem-c/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Reference C and C++ implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a
noncryptographic pseudorandom number generator. It produces the stream the specification
defines, bit for bit, with SIMD fills on CPUs and OpenMP target fills on GPUs.

Build with `make`, or compile `tandem.c` into your project with `-ffp-contract=off` and no fast-math. The library needs C11, C++17 and libm.

```sh
make                              # libtandem.a, built with clang
make install PREFIX=<prefix>      # libtandem.a, tandem.h, tandem.hpp, tandem.pc
make test
```

```c
#include "tandem.h"

tandem_rng rng = tandem_seed(42, 0, 0);          /* 128-bit seed as two halves, default K */
uint32_t words[1024];
tandem_fill_u32(&rng, words, 1024);
tandem_rng worker = tandem_split(&rng, 7);        /* by index, from the key alone */
double z = tandem_normal_f64(&worker);            /* ziggurat; link with -lm */
```

- Weighted choice by an integer alias table, one draw per index, bit exact across ports.
- Inline 32- and 64-bit scalar draws, as fast as xoshiro256++ in a loop on an Apple M4.
- A plain `make` on x86-64 runs the AVX2 and FMA copy on every CPU that has them, picked at
  run time. Without them, the base copy rounds each fused multiply-add by an exact emulation,
  with the same bits and no library call.

See [API](docs/api.md) for the C++ `<random>` engine, MPI and OpenMP offload, and
[design](docs/design.md), [tests](docs/tests.md) and [speed](docs/speed.md) for the rest.

Portions of the code were generated with the assistance of LLMs.

[Documentation](https://tandem-rng.github.io/tandem-c/) · [Apache 2.0 license](LICENSE)
