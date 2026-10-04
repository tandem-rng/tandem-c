<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .c"></p>

# tandem-c

[![CI](https://github.com/tandem-rng/tandem-c/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-c/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Reference C and C++ implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a
noncryptographic pseudorandom number generator. It produces the stream the specification
defines, bit for bit, with SIMD fills on CPUs and OpenMP target fills on GPUs.

Build with `make`, or compile `tandem.c` into your project. The library needs C11, C++17 and libm.

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
double z = tandem_normal_f64(&worker);            /* Box-Muller; link with -lm */
```

See [API](docs/api.md) for the C++ `<random>` engine, MPI and OpenMP offload, and
[design](docs/design.md), [tests](docs/tests.md) and [speed](docs/speed.md) for the rest.

Portions of the code were generated with the assistance of LLMs.

[Documentation](docs/index.md) · [Apache 2.0 license](LICENSE)
