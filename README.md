<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .c"></p>

# tandem-c

Reference C implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a
noncryptographic pseudorandom number generator built to be fast on CPUs and GPUs alike. It
produces the stream the specification defines, bit for bit, for every type it supports.

- No dependencies besides libm, two files: `tandem.h` and `tandem.c`. `tandem.hpp` adds a
  C++ value type that satisfies `std::uniform_random_bit_generator`, so it drives every
  `<random>` distribution. The library needs C11 and C++17 and nothing newer, so projects
  that vendor it can keep their own flags. `make` builds it as C23 and C++23 with clang, the
  primary compiler. CI builds with the latest clang and gcc, and a separate job compiles with
  strict `-std=c11 -pedantic-errors` and `-std=c++17` to keep the older standards honest.
- `tandem::rng` is a C++ random number engine: it has `seed`, `discard` in constant time,
  `==`, and stream operators, so `std::shuffle` and every `<random>` distribution take it.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`) plus
  a cache of the current 1024-bit row. Copy it by value.
- Every type in the specification: `bool`, 8 to 128-bit unsigned integers, `float`,
  `double`, binary16 as bit patterns, Unicode scalars, and complex pairs. Signed integers are
  the unsigned draws reinterpreted. Random access without advancing. Split by index, fork at
  the current block, sub by purpose. Bounded integers `tandem_u32_below` and
  `tandem_u64_below` (Lemire) and normals `tandem_normal_f64` and `tandem_normal_f32`
  (Box-Muller, which needs `-lm`) go beyond the specification and return the same values as
  tandem-cuda. The scalar bounded draws loop on rejection. The bounded fills use the parallel
  rule of tandem-cuda instead: element i maps draw i of the plain fill, which keeps the SIMD
  speed, and a rejected draw retries on a fallback generator, so a fill consumes exactly `len`
  draws. A Box-Muller step gives two normals: `tandem_normal2_f64` and `_f32` return both
  (cos half first), `tandem_normal_*` returns the cos half, and `tandem_fill_normal_*` writes
  the flattened pairs, so an odd count still consumes `2 * ceil(n / 2)` uniforms. The f32
  normals draw two f32 uniforms and compute the radius in float, so they agree across ports to
  a few ulps because float libm functions differ. Uniforms are bit exact and f64 normals agree
  to about 1e-12 relative.
  The normal fills and draws share one loop without libm calls: polynomials for the logarithm
  of the exponent-split argument and for the sine and cosine of the angle after an exact
  quarter-turn reduction, which the compiler vectorizes. `make accuracy` compares the fills with
  libm on 5e7 pairs. The f64 normals agree to 1e-15 relative and the f32 normals to 3.3 ulps.
  `make bench` reports the normal fills too.
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
tandem_set_position(&rng, 0);                      /* rewind; false if the position is >= 2^63 */
uint32_t die = tandem_u32_below(&rng, 6);          /* uniform on [0, 6), Lemire's method */
uint64_t idx[100];
tandem_fill_u64_below(&rng, idx, 100, 1000000000000u);
double z = tandem_normal_f64(&rng);                /* Box-Muller; link with -lm */
float zs[1000];
tandem_fill_normal_f32(&rng, zs, 1000);
```

Build with `make`, which uses clang and produces `libtandem.a`, or compile `tandem.c` into
your project. Set `CC` and `CXX` for another compiler.

From C++:

```cpp
#include "tandem.hpp"

tandem::rng g(42);
std::normal_distribution<double> gauss;
double z = gauss(g);                               /* any <random> distribution */
double u = g.next<double>();                       /* the spec's own draws */
std::vector<float> xs = g.fill<float>(1 << 20);
auto c = g.next<std::complex<double>>();           /* also uint128, char32_t, float16_bits */
double u3 = g.at<double>(3);                       /* random access, no advance */
uint32_t die = g.below<uint32_t>(6);               /* Lemire */
double n = g.normal();                             /* Box-Muller, cos half */
auto pair = g.normal2();                           /* both halves of one step */
g.set_position(0);
tandem::rng worker = g.split(7);
std::vector<tandem::rng> kids = g.fork(4);
```

## Counter-based interface

`tandem123.h` is a header-only function of a counter and a key in the shape of the
[Random123](https://github.com/DEShawResearch/random123) interface, for C, C++, CUDA and HIP
device code (`TANDEM_CUDA_DEVICE` marks the functions `__host__ __device__`):

```c
#include "tandem123.h"

tandem4x32_key_t key = tandem4x32_key_from_seed(42, 0);   /* or the four key words directly */
tandem4x32_ctr_t ctr = {{block, 0, 0, 0}};                 /* see the mapping below */
tandem4x32_ctr_t out = tandem4x32(ctr, key);               /* four 32-bit words */
```

It differs from `philox4x32` in that the key has four words, not two, because Tandem's key is
128 bits, and the counter is not a free space. `ctr.v[0] + 2^32 ctr.v[1]` is the 128-bit
block, which is stream position `128 block`. `ctr.v[2]` is the chunk length `K`, a power of two
up to 65536, and 0 means 32. `ctr.v[3]` is reserved. The result is the specification's block at
that position: the words `tandem_fill_u32` writes as elements `4 block` to `4 block + 3` for a
generator with that key and `K` at position 0. The specification limits positions to 2^63 bits,
so `ctr.v[1]` stays below 2^24. `tests/test_r123.c` checks this against `tandem_fill_u32`,
`tandem_block`, and the reference stream dumps.

## Parallel use

Element `i` of a fill is draw `i`, at bit `64 i` for doubles, so a rank or thread that starts
its generator at the position of its first element writes its part of one global fill.
`tandem_split` gives one stream per task from the key alone, and `tandem_sub` one per purpose.
Results then do not depend on the number of ranks or threads.
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification gives the patterns.

```c
tandem_rng mine = tandem_from_key(key, 64 * first, K);   /* key and K of the global generator */
tandem_fill_f64(&mine, x + first, count);                /* x[first .. first + count) */
tandem_rng task = tandem_split(&noise, task_index);      /* by task, not by rank */
```

`examples/mpi` fills a field of 2^24 doubles across MPI ranks or OpenMP threads, draws a batch
of normals per block from `tandem_split(block)`, and prints a hash of the result.
`pixi run -e mpi check` in that directory builds it with MPICH and checks that 1, 2 and 4 ranks
and 1, 4 and 14 threads print the hash of a serial run. CI runs the same check.

## Install

`make install PREFIX=<prefix>` installs `libtandem.a`, `tandem.h`, `tandem.hpp` and a `tandem.pc` file for
`pkg-config`. `DESTDIR` stages the files. The `packaging/` directory holds a Spack recipe
(`spack/package.py`) and a conda-forge style recipe (`conda/recipe.yaml`). Neither is submitted to
Spack or conda-forge yet, and both build from the `main` branch.

## OpenMP target offload

`tandem_target.c` provides `tandem_fill_u32_target`, `_u64_`, `_f32_` and `_f64_`, which fill
device memory from a `#pragma omp target teams distribute parallel for` region, one thread per
chunk, with the same values and the same final position as the host fills. Declare them by
defining `TANDEM_OPENMP_TARGET` before `tandem.h`, and compile `tandem_target.c` with the
offload flags of your compiler:

```sh
make test-target OMP_FLAGS="-mp=gpu -gpu=cc80" CC=nvc CFLAGS="-std=c11 -O2"        # nvc
make test-target OMP_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80"
```

The plain build does not see any of it. The default `OMP_FLAGS` run the target regions on the
host with host memory, which is what CI does. `tests/test_target.c` compares the four fills
bit for bit with the host fills over four chunk lengths, aligned and unaligned starts, and
sizes from 0 to 2^20.

On an A100 40 GB PCIe (driver 570, GPU idle before each run) with nvc 25.3, `make bench-target`
writes 2^28 elements into device memory at 480 to 850 GiB/s over ten runs, median about 570,
the same for all four types, against 1386 GiB/s for the CUDA kernels of tandem-cuda. The
conda-forge clang 19 of tandem-cuda's pixi environment ships no `libomptarget` device
runtime, so it cannot build the offload target there. nvc can.

## Tests

```sh
make test
```

`tests/test_vectors.c` checks every vector of the specification. `tests/vectors.h` is
generated from the spec repository's `vectors.json` by `tools/gen_vectors.py`, and CI fails
when it is out of date. `tests/test_stream.c` compares long fills, scalar draws, and random
access against reference stream dumps in `tests/data`, written by `tools/dump_streams.jl`.
`tests/test_api.c` checks the functions that are not part of the specification's draws. The
bounded integers (scalar and fill) and normals are compared, values and stream position,
with fixtures that `tools/gen_cross.cpp` computes from the shared core of
[tandem-cuda](https://github.com/tandem-rng/tandem-cuda). f32 normals match within 16 ulps. The device-derived fill fixtures of tandem-cuda are copied to `tests/cuda_fill_*.h`.
`make cross` regenerates the fixtures.
`tests/test_r123.c` checks `tandem123.h` against the fills and the dumps.
`tests/test_target.c` checks the OpenMP target fills against the host fills.
`tests/test_cpp.cpp` checks that the C++ wrapper, including `at`, `below`, `normal`,
`set_position` and the extra draw types, agrees with the C API and runs `<random>`. Compiled
as C++20 it also checks `std::uniform_random_bit_generator`.

## Speed

Apple M4, one thread, `make bench` (clang, `-O2`; all figures here are clang figures unless
they say gcc), minimum of seven runs of 2^24 elements
after a warm-up:

| | GiB/s | with `TANDEM_NO_SIMD` |
|---|---|---|
| `tandem_fill_u32` | 20.4 | 12.3 |
| `tandem_fill_u64` | 20.2 | 12.3 |
| `tandem_fill_f32` | 17.4 | 11.1 |
| `tandem_fill_f64` | 17.5 | 11.1 |
| `tandem_fill_normal_f64` | 5.0 | 4.3 |
| `tandem_fill_normal_f32` | 5.5 | 4.6 |
| `tandem_next_f64` chain, ns per draw | 1.40 | 1.77 |

The normal rows count the bytes written. They run the vectorized Box-Muller loop described
above after the float fill, so they do not depend on `TANDEM_NO_SIMD` except through the
uniforms.

The row loop keeps the eight lane states in registers and stores each row by a vector
transpose, which is where the throughput comes from. Float fills map the words to floats in
the same loop, before the store. On AArch64 the fixed-point `ucvtf` does the shift and the
scale in one instruction. The 32x32 to 64-bit products are NEON `umull` and SSE2 `pmuludq`
through intrinsics: from the portable spelling, a widened 64-bit vector multiply, GCC emits
scalar multiplies on AArch64 and three `pmuludq` per product on x86. Clang gives the
figures above. GCC 16 at `-O2` reaches 14.5 GiB/s for `tandem_fill_u32` on the same machine
and 16.1 at `-O3`, because it keeps the lane states in memory for part of the row loop. The
scalar fallback is one straight-line step per lane in a loop over the eight lanes. Clang
vectorizes that loop, GCC 16 does not at `-O2` and reaches 5.6 GiB/s.

`make bench` also runs `tools/bench_std.cpp`, the C++ wrapper against the generators of the
C and C++ standard libraries on the same machine (Apple clang 21, libc++):

| | GiB/s |
|---|---|
| `tandem::rng::fill<uint32_t>` | 20.5 |
| `tandem::rng::fill<uint64_t>` | 20.3 |
| `tandem::rng::fill<float>` | 18.4 |
| `tandem::rng::fill<double>` | 17.7 |
| `tandem::rng::next<double>` chain | 5.4 |
| `std::uniform_real_distribution<double>` on `tandem::rng` | 5.3 |
| `rand()`, 31 bits per call into `uint32_t` | 1.0 |
| `random()`, 31 bits per call into `uint32_t` | 2.7 |
| `arc4random_buf` | 4.9 |
| `std::mt19937`, `uint32_t` | 3.0 |
| `std::mt19937_64`, `uint64_t` | 5.5 |
| `std::mt19937` with `std::uniform_real_distribution<float>` | 1.6 |
| `std::mt19937_64` with `std::uniform_real_distribution<double>` | 3.1 |
| `std::mt19937_64` with `std::generate_canonical<double, 53>` | 5.5 |

The standard generators have no fill interface, so each writes one value per call.

## AI assistance

This implementation was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
