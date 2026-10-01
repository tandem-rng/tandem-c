# tandem-c

Reference C implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a
noncryptographic pseudorandom number generator designed for GPUs first. The Julia reference
is [TandemRNG.jl](https://github.com/tandem-rng/TandemRNG.jl). This implementation produces
the same stream, bit for bit, for every type it supports.

- C99, no dependencies, two files: `tandem.h` and `tandem.c`.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`) plus
  a cache of the current 1024-bit row. Copy it by value.
- Scalar draws and fills for `bool`, 8 to 64-bit unsigned integers, `float`, and `double`.
  Random access without advancing. Split by index, fork at the current block, sub by purpose.
- Scalar code, written for clarity. Sequential fills cost one `T` per 16 bytes once a row is
  cached. A SIMD row kernel is future work.

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

## Not implemented

`Float16`, 128-bit integers, `Char`, and complex draws from the specification's type table.
Signed integers are the unsigned draws reinterpreted.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
