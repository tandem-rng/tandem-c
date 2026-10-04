<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .c"></p>

# tandem-c

Reference C implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a
noncryptographic pseudorandom number generator. It produces the stream the specification
defines, bit for bit. It is fast on CPUs and GPUs alike.

## Install

```sh
make                              # libtandem.a, built with clang
make install PREFIX=<prefix>      # libtandem.a, tandem.h, tandem.hpp, tandem.pc
```

Or compile `tandem.c` into your project. The library needs C11 and C++17 and libm. `make`
builds C23 and C++23. `packaging/` holds a Spack recipe and a conda-forge style recipe.

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
uint32_t die = tandem_u32_below(&rng, 6);          /* uniform on [0, 6), Lemire's method */
double z = tandem_normal_f64(&rng);                /* Box-Muller; link with -lm */
double e = tandem_exponential_f64(&rng);           /* -ln(1 - u), Exp(1) */
```

```cpp
#include "tandem.hpp"

tandem::rng g(42);
std::normal_distribution<double> gauss;
double z = gauss(g);                               /* any <random> distribution */
std::vector<float> xs = g.fill<float>(1 << 20);
tandem::rng worker = g.split(7);
```

## What it provides

Packaging, design, test and speed detail: [docs/notes.md](docs/notes.md).

- `tandem_rng`: a copyable value, 128-bit key, 64-bit bit position, chunk length `K`.
- Every specification type: `bool`, 8 to 128-bit unsigned integers, `float`, `double`,
  binary16 bit patterns, Unicode scalars, complex pairs. Scalar draws and `tandem_fill_*`.
- `tandem_set_position`, `tandem_split`, `tandem_fork`, `tandem_sub`, `tandem_from_key`.
- Bounded integers `tandem_u32_below`, `tandem_u64_below` and `tandem_fill_u32_below`,
  `tandem_fill_u64_below`. A fill cut anywhere equals the whole fill.
- Normals `tandem_normal_f64`, `tandem_normal_f32`, `tandem_normal2_*`, `tandem_fill_normal_*`.
  The f64 and f32 normals match the device core of tandem-cuda to 1e-12 relative and a few ulps.
- Exponentials `tandem_exponential_*` and `tandem_fill_exponential_*`, bit exact on every
  compiler, target and device.
- `tandem::rng` in `tandem.hpp`: a `std::uniform_random_bit_generator` with `seed`,
  constant-time `discard`, `at`, `below`, `normal`, `exponential`, `split` and `fork`.
- `tandem123.h`: a Random123-style counter and key function `tandem4x32` for C, C++, CUDA and
  HIP device code.
- Parallel use: element `i` of a fill is draw `i`, so a rank that starts at its first
  element writes its part of one global fill. `examples/mpi` shows MPI and OpenMP threads.
  See [Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative).
- OpenMP target offload: `tandem_fill_u32_target`, `_u64_`, `_f32_` and `_f64_` from
  `tandem_target.c`. Define `TANDEM_OPENMP_TARGET` before `tandem.h`.
- x86-64 builds pick an AVX2 and FMA copy of the fill loops at run time. Define
  `TANDEM_NO_AVX2` or `TANDEM_NO_SIMD` to turn off the AVX2 copy or all SIMD. The bits stay
  the same.

## Tests

```sh
make test
```

- The specification vectors in `tests/vectors.h`, generated from the spec's `vectors.json`.
- Long fills, scalar draws and random access against the stream dumps in `tests/data`.
- Bounded integers, normals and exponentials against fixtures from the tandem-cuda core,
  with hashes of 10^7 normals and 10^6 exponentials. `make cross` regenerates them.
- `tandem123.h` against the fills, and the C++ wrapper against the C API.
- `make test-target` checks the OpenMP target fills against the host fills.

## Speed

Apple M4 and AMD EPYC 7702P, one thread, `make bench`, clang `-O2`, minimum of seven runs of
2^24 elements, in GiB/s.

| | M4 | M4 with `TANDEM_NO_SIMD` | EPYC 7702P |
|---|---|---|---|
| `tandem_fill_u32` | 20.4 | 12.3 | 8.4 |
| `tandem_fill_u64` | 20.2 | 12.3 | 8.7 |
| `tandem_fill_f32` | 17.4 | 11.1 | 7.9 |
| `tandem_fill_f64` | 17.5 | 11.1 | 7.0 |
| `tandem_fill_normal_f64` | 5.0 | 4.3 | 2.4 |
| `tandem_fill_normal_f32` | 5.5 | 4.6 | 3.3 |
| `tandem_fill_exponential_f64` | 6.1 | 5.1 | 3.4 |
| `tandem_fill_exponential_f32` | 6.7 | 5.5 | 4.5 |
| `tandem_next_f64` chain, ns per draw | 1.40 | 1.77 | 4.05 |

The C++ wrapper against the standard generators on the M4, Apple clang 21, libc++, in GiB/s.

| | GiB/s |
|---|---|
| `tandem::rng::fill<uint32_t>` | 20.5 |
| `tandem::rng::fill<uint64_t>` | 20.3 |
| `tandem::rng::fill<float>` | 18.4 |
| `tandem::rng::fill<double>` | 17.7 |
| `tandem::rng::next<double>` chain | 5.4 |
| `std::uniform_real_distribution<double>` on `tandem::rng` | 5.3 |
| `std::mt19937`, `uint32_t` | 3.0 |
| `std::mt19937_64`, `uint64_t` | 5.5 |
| `std::mt19937_64` with `std::generate_canonical<double, 53>` | 5.5 |
| `arc4random_buf` | 4.9 |

OpenMP target offload on an A100 40 GB PCIe with nvc 25.3: 2^28 elements at about 570 GiB/s
median, `make bench-target`. Build with `OMP_FLAGS="-mp=gpu -gpu=cc80" CC=nvc`.

## AI assistance

This implementation was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
