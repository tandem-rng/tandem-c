# tandem-c notes

Detail moved out of the README. Sections follow the README headings.

## Install

The `packaging/` directory holds a Spack recipe (`spack/package.py`) and a conda-forge style
recipe (`conda/recipe.yaml`). Neither is submitted to Spack or conda-forge yet, and both build
from the `main` branch. `DESTDIR` stages the installed files.

The library needs C11 and C++17 and nothing newer, so projects that vendor it can keep their
own flags. `make` builds it as C23 and C++23 with clang, the primary compiler. Set `CC` and
`CXX` for another compiler. A separate CI job compiles with strict `-std=c11 -pedantic-errors`
and `-std=c++17`.

## What it provides

- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`) plus
  a cache of the current 1024-bit row. Copy it by value.
- `tandem::rng` is a C++ random number engine: it has `seed`, `discard` in constant time,
  `==`, and stream operators, so `std::shuffle` and every `<random>` distribution take it.
- Signed integers are the unsigned draws reinterpreted. Random access does not advance.
- Bounded integers `tandem_u32_below` and `tandem_u64_below` (Lemire) and normals
  `tandem_normal_f64` and `tandem_normal_f32` (Box-Muller, which needs `-lm`) go beyond the
  specification and return the same values as tandem-cuda. The scalar bounded draws loop on
  rejection. The bounded fills use the parallel rule of tandem-cuda instead: element i maps
  draw i of the plain fill, which keeps the SIMD speed, and a rejected draw retries on a
  fallback generator keyed by the global draw index, so a fill consumes exactly `len` draws and
  a fill cut anywhere equals the whole fill.
- A Box-Muller step gives two normals: `tandem_normal2_f64` and `_f32` return both (cos half
  first), `tandem_normal_*` returns the cos half, and `tandem_fill_normal_*` writes the
  flattened pairs, so an odd count still consumes `2 * ceil(n / 2)` uniforms. The f32 normals
  draw two f32 uniforms and compute the radius in float, so they agree across ports to a few
  ulps because float libm functions differ. Uniforms are bit exact and f64 normals agree to
  about 1e-12 relative.
- The normal fills and draws share one loop without libm calls: polynomials for the logarithm
  of the exponent-split argument and for the sine and cosine of the angle after an exact
  quarter-turn reduction, which the compiler vectorizes. `make accuracy` compares the fills
  with libm on 5e7 pairs. The f64 normals agree to 1e-15 relative and the f32 normals to 3.3
  ulps. `make bench` reports the normal fills too.
- Every multiply-add in that loop is an explicit `fma`, and the loop is built with floating
  point contraction off, so every compiler and target produces the same bits:
  `tests/test_normal_bits.c` checks a hash of 10^7 normals against the value from the M4.
- Exponentials `tandem_exponential_f64` and `_f32` and the fills `tandem_fill_exponential_f64`
  and `_f32` return `-ln(1 - u)` from one uniform `u` each, as Appendix A of the specification
  describes: element i of a fill comes from uniform i, so a fill equals the scalar draws and a
  fill cut anywhere equals the whole fill. They use the logarithm of the normal loop, with no
  libm call, and tandem-cuda runs the same arithmetic on the device, so exponentials are bit
  exact across compilers, targets and devices. The maximum error is 1.1e-15 relative for f64
  and 2.8e-7 for f32, checked over all 2^24 f32 inputs.
- On x86-64 with GCC 12+ or clang, `tandem.c` compiles the row loop and the normal and
  exponential loops a second time for AVX2 and FMA and picks that copy at run time. A plain
  `-O2` build with no `-m` flags so gets 256-bit rows and hardware fused multiply-adds on
  Haswell, Zen and newer, and still runs on older CPUs, where `fma` is a library call that is
  correct but about seven times slower. With `-mavx2 -mfma` only the AVX2 copy remains.
  Define `TANDEM_NO_AVX2` to leave it out. Both copies give the same bits.
- The eight chunks of a row step together in registers. With GCC 12+ or clang the step is
  written with vector extensions and compiles to NEON, SSE2 or, in the AVX2 copy, one 256-bit
  vector per word. Define `TANDEM_NO_SIMD` for the scalar version. After alignment every
  integer fill is one byte stream, so one routine serves all widths and floats convert in
  place.

### Counter-based interface

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

### Parallel use

Element `i` of a fill is draw `i`, at bit `64 i` for doubles, so a rank or thread that starts
its generator at the position of its first element writes its part of one global fill.
`tandem_split` gives one stream per task from the key alone, and `tandem_sub` one per purpose.
Results then do not depend on the number of ranks or threads.

```c
tandem_rng mine = tandem_from_key(key, 64 * first, K);   /* key and K of the global generator */
tandem_fill_f64(&mine, x + first, count);                /* x[first .. first + count) */
tandem_rng task = tandem_split(&noise, task_index);      /* by task, not by rank */
```

`examples/mpi` fills a field of 2^24 doubles across MPI ranks or OpenMP threads, draws a batch
of normals per block from `tandem_split(block)`, and prints a hash of the result.
`pixi run -e mpi check` in that directory builds it with MPICH and checks that 1, 2 and 4 ranks
and 1, 4 and 14 threads print the hash of a serial run.

### OpenMP target offload

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
host with host memory. `tests/test_target.c` compares the four fills bit for bit with the host
fills over four chunk lengths, aligned and unaligned starts, and sizes from 0 to 2^20.

On an A100 40 GB PCIe (driver 570, GPU idle before each run) with nvc 25.3, `make bench-target`
writes 2^28 elements into device memory at 480 to 850 GiB/s over ten runs, median about 570,
the same for all four types, against 1386 GiB/s for the CUDA kernels of tandem-cuda. The
conda-forge clang 19 of tandem-cuda's pixi environment ships no `libomptarget` device
runtime, so it cannot build the offload target there. nvc can.

## Tests

`tests/test_vectors.c` checks every vector of the specification. `tests/vectors.h` is
generated from the spec repository's `vectors.json` by `tools/gen_vectors.py`.
`tests/test_stream.c` compares long fills, scalar draws, and random access against reference
stream dumps in `tests/data`, written by `tools/dump_streams.jl`. `tests/test_api.c` checks the
functions that are not part of the specification's draws. The bounded integers (scalar and
fill) and normals are compared, values and stream position, with fixtures that
`tools/gen_cross.cpp` computes from the shared core of
[tandem-cuda](https://github.com/tandem-rng/tandem-cuda). The normals match bit for bit,
because the host core runs the same explicit-fma loop. The device-derived fill fixtures of
tandem-cuda are copied to `tests/cuda_fill_*.h`, where device normals match within 1e-12
relative for f64 and 16 ulps for f32. `tests/cross_exponential.h` holds exponentials of the
same core from five start positions, unaligned ones included, and the fills and scalar draws
must match it bit for bit. `make cross` regenerates the fixtures.

`tests/test_api.c` also cuts exponential fills at several elements and compares them with the
whole fill and the scalar draws, and checks 10^7 f64 and 10^7 f32 exponentials against Exp(1):
the first four raw moments within five standard errors and a Kolmogorov-Smirnov statistic
below the 0.1 % point. `tests/test_exponential_bits.c` checks the FNV-1a hash
`47f8f98297d94ee2` of 10^6 f64 and 10^6 f32 exponentials from each of five positions.
`tools/dump_exponentials.c` writes the same bytes, whose SHA-256 is
`5c035a4ef1368231d25a9c2f9201be2df3224e28a14549a50625d0db3770ef4e`.

`tests/test_cpp.cpp` checks that the C++ wrapper, including `at`, `below`, `normal`,
`exponential`, `set_position` and the extra draw types, agrees with the C API and runs
`<random>`. Compiled as C++20 it also checks `std::uniform_random_bit_generator`.

On x86-64 CI runs the suite three times: plain `-O2`, which takes the AVX2 copy on the
runner, `-DTANDEM_NO_AVX2` for the base copy, and `-mavx2 -mfma`.

## Speed

On the EPYC the base copy, which `TANDEM_NO_AVX2` selects, reaches 6.2 GiB/s for
`tandem_fill_u32` and 0.24 for `tandem_fill_normal_f64`, where `fma` is a library call. The
EPYC build is clang 20 at plain `-O2` with no `-m` flags, so it runs the AVX2 copy picked at
run time. GCC 12 at `-O2` reaches 9.1 GiB/s for `tandem_fill_u32` but only 0.83 for
`tandem_fill_normal_f64`. GCC honours `-fno-math-errno` only on the command line, not as a
function attribute, so `sqrt` keeps its errno branch and the normal loop stays scalar, with
hardware fused multiply-adds.

The normal and exponential rows count the bytes written. They run the vectorized loops
after the float fill, so they do not depend on `TANDEM_NO_SIMD` except through the uniforms.

The row loop keeps the eight lane states in registers and stores each row by a vector
transpose, which is where the throughput comes from. Float fills map the words to floats in
the same loop, before the store. On AArch64 the fixed-point `ucvtf` does the shift and the
scale in one instruction. The 32x32 to 64-bit products are NEON `umull` and SSE2 `pmuludq`
through intrinsics: from the portable spelling, a widened 64-bit vector multiply, GCC emits
scalar multiplies on AArch64 and three `pmuludq` per product on x86. Clang gives the
figures in the README. GCC 16 at `-O2` reaches 14.5 GiB/s for `tandem_fill_u32` on the same
machine and 16.1 at `-O3`, because it keeps the lane states in memory for part of the row
loop. The scalar fallback is one straight-line step per lane in a loop over the eight lanes.
Clang vectorizes that loop, GCC 16 does not at `-O2` and reaches 5.6 GiB/s.

`make bench` also runs `tools/bench_std.cpp`, the C++ wrapper against the generators of the
C and C++ standard libraries on the same machine (Apple clang 21, libc++). The standard
generators have no fill interface, so each writes one value per call. Rows left out of the
README table:

| | GiB/s |
|---|---|
| `rand()`, 31 bits per call into `uint32_t` | 1.0 |
| `random()`, 31 bits per call into `uint32_t` | 2.7 |
| `std::mt19937` with `std::uniform_real_distribution<float>` | 1.6 |
| `std::mt19937_64` with `std::uniform_real_distribution<double>` | 3.1 |

A100 offload range: 480 to 850 GiB/s over ten runs, see the offload section above.
