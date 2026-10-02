<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .c"></p>

# tandem-c

Reference C implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a
noncryptographic pseudorandom number generator built to be fast on CPUs and GPUs alike. It
produces the stream the specification defines, bit for bit, for every type it supports.

- C99, no dependencies, two files: `tandem.h` and `tandem.c`. `tandem.hpp` adds a C++17
  value type that satisfies `std::uniform_random_bit_generator`, so it drives every
  `<random>` distribution.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`) plus
  a cache of the current 1024-bit row. Copy it by value.
- Every type in the specification: `bool`, 8 to 128-bit unsigned integers, `float`,
  `double`, binary16 as bit patterns, Unicode scalars, and complex pairs. Signed integers are
  the unsigned draws reinterpreted. Random access without advancing. Split by index, fork at
  the current block, sub by purpose.
- The eight chunks of a row step together in registers. With GCC 12+ or clang the step is
  written with vector extensions and compiles to NEON or SSE/AVX. Define `TANDEM_NO_SIMD`
  for the scalar version. After alignment every integer fill is one byte stream, so one
  routine serves all widths and floats convert in place.

## Use

```c
#include "tandem.h"

tandem_rng rng = tandem_seed(42, 0, 0);          /* 128-bit seed as two halves, default K */
double x = tandem_next_f64(&rng);
uint32_t words[1024];
tandem_fill_u32(&rng, words, 1024);
tandem_rng worker = tandem_split(&rng, 7);        /* by index, from the key alone */
tandem_rng kids[4];
tandem_fork(&rng, kids, 4);                        /* from the current block, parent moves on */
```

Build with `make`, which produces `libtandem.a`, or compile `tandem.c` into your project.

From C++:

```cpp
#include "tandem.hpp"

tandem::rng g(42);
std::normal_distribution<double> gauss;
double z = gauss(g);                               /* any <random> distribution */
double u = g.next<double>();                       /* the spec's own draws */
std::vector<float> xs = g.fill<float>(1 << 20);
tandem::rng worker = g.split(7);
std::vector<tandem::rng> kids = g.fork(4);
```

## Tests

```sh
make test
```

`tests/test_vectors.c` checks every vector of the specification. `tests/vectors.h` is
generated from the spec repository's `vectors.json` by `tools/gen_vectors.py`, and CI fails
when it is out of date. `tests/test_stream.c` compares long fills, scalar draws, and random
access against reference stream dumps in `tests/data`, written by `tools/dump_streams.jl`.
`tests/test_cpp.cpp` checks that the C++ wrapper agrees with the C API and runs `<random>`.

## Speed

Apple M4, one thread, `make bench` (clang, `-O2`), minimum of seven runs of 2^24 elements
after a warm-up:

| | GiB/s | with `TANDEM_NO_SIMD` |
|---|---|---|
| `tandem_fill_u32` | 17.1 | 12.1 |
| `tandem_fill_u64` | 17.7 | 12.7 |
| `tandem_fill_f32` | 13.1 | 10.1 |
| `tandem_fill_f64` | 13.6 | 10.3 |
| `tandem_next_f64` chain, ns per draw | 1.52 | 1.9 |

The row loop keeps the eight lane states in registers and stores each row by a vector
transpose, which is where the throughput comes from. Float fills pay a second pass for the
conversion.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
