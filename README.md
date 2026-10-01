# tandem-c

Reference C implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a
noncryptographic pseudorandom number generator designed for GPUs first. The Julia reference
is [TandemRNG.jl](https://github.com/tandem-rng/TandemRNG.jl). This implementation produces
the same stream, bit for bit, for every type it supports.

- C99, no dependencies, two files: `tandem.h` and `tandem.c`.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`) plus
  a cache of the current 1024-bit row. Copy it by value.
- Every type in the specification: `bool`, 8 to 128-bit unsigned integers, `float`,
  `double`, binary16 as bit patterns, Unicode scalars, and complex pairs. Signed integers are
  the unsigned draws reinterpreted. Random access without advancing. Split by index, fork at
  the current block, sub by purpose.
- The eight chunks of a row step together. With GCC or clang the step is written with vector
  extensions and compiles to NEON or SSE/AVX. Define `TANDEM_NO_SIMD` for the scalar step.
  Fills copy whole 128-byte rows out of the cache.

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

## Tests

```sh
make test
```

`tests/test_vectors.c` checks every vector of the specification. `tests/vectors.h` is
generated from the spec repository's `vectors.json` by `tools/gen_vectors.py`, and CI fails
when it is out of date. `tests/test_stream.c` compares long fills, scalar draws, and random
access against dumps written by TandemRNG.jl with `tools/dump_streams.jl`.

## Speed

Apple M4, one thread, `make bench` (clang, `-O2`), minimum of seven runs of 2^24 elements:

| | GiB/s | with `TANDEM_NO_SIMD` |
|---|---|---|
| `tandem_fill_u32` | 8.6 | 4.0 |
| `tandem_fill_u64` | 8.5 | 2.4 |
| `tandem_fill_f32` | 10.0 | 1.6 |
| `tandem_fill_f64` | 9.4 | 2.2 |
| `tandem_next_f64` chain | 3.7 | 2.2 |

The scalar-fallback column predates the whole-row fill path and is shown for the row step
alone. TandemRNG.jl reaches 14 to 18 GiB/s on the same machine with its hand-shuffled
`Lane8` core, so this C is a correct and reasonably fast reference, not the speed ceiling.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
