# API

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
double z = tandem_normal_f64(&rng);                /* ziggurat; link with -lm */
double e = tandem_exponential_f64(&rng);           /* -ln(1 - u), Exp(1) */
```

## Reference

- `tandem_rng`: a copyable value, 128-bit key, 64-bit bit position, chunk length `K`. A
  generator is this transport form plus a cache of the current 1024-bit row. Copy it by value.
- Every specification type: `bool`, 8 to 128-bit unsigned integers, `float`, `double`,
  binary16 bit patterns, Unicode scalars, complex pairs. Scalar draws and `tandem_fill_*`.
  Signed integers are the unsigned draws reinterpreted. Random access does not advance.
- `tandem_set_position`, `tandem_split`, `tandem_fork`, `tandem_sub`, `tandem_from_key`.
- Bounded integers `tandem_u32_below`, `tandem_u64_below` and `tandem_fill_u32_below`,
  `tandem_fill_u64_below`. A fill cut anywhere equals the whole fill.
- Normals `tandem_normal_f64`, `tandem_normal_f32`, `tandem_normal2_f32`,
  `tandem_fill_normal_*`. f64 normals are the ziggurat of Appendix A, one draw per element, bit
  exact on every compiler and target and with every port that copies its tables and logarithm.
  An f64 fill cut anywhere equals the whole fill. f32 normals are Box-Muller pairs, bit exact
  with tandem-cuda on the host.
- Exponentials `tandem_exponential_*` and `tandem_fill_exponential_*`, bit exact on every
  compiler, target and device.
- `tandem.c` includes `tandem_normal_tables.h`. Copy both when you vendor the library.
- x86-64 builds pick an AVX2 and FMA copy of the fill loops at run time. Define
  `TANDEM_NO_AVX2` or `TANDEM_NO_SIMD` to turn off the AVX2 copy or all SIMD. The bits stay
  the same.

## C++

```cpp
#include "tandem.hpp"

tandem::rng g(42);
std::normal_distribution<double> gauss;
double z = gauss(g);                               /* any <random> distribution */
std::vector<float> xs = g.fill<float>(1 << 20);
tandem::rng worker = g.split(7);
```

`tandem::rng` is a `std::uniform_random_bit_generator` and a C++ random number engine with
`seed`, constant-time `discard`, `==`, stream operators, `at`, `below`, `normal`,
`exponential`, `split` and `fork`. `std::shuffle` and every `<random>` distribution take it.

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
host with host memory. The conda-forge clang 19 of tandem-cuda's pixi environment ships no
`libomptarget` device runtime, so it cannot build the offload target there. nvc can.

## Parallel use

Element `i` of a fill is draw `i`, at bit `64 i` for doubles, so a rank or thread that starts
its generator at the position of its first element writes its part of one global fill.
`tandem_split` gives one stream per task from the key alone, and `tandem_sub` one per purpose.
Results then do not depend on the number of ranks or threads. See
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification.

```c
tandem_rng mine = tandem_from_key(key, 64 * first, K);   /* key and K of the global generator */
tandem_fill_f64(&mine, x + first, count);                /* x[first .. first + count) */
tandem_rng task = tandem_split(&noise, task_index);      /* by task, not by rank */
```

`examples/mpi` fills a field of 2^24 doubles across MPI ranks or OpenMP threads, draws a batch
of normals per block from `tandem_split(block)`, and prints a hash of the result.
`pixi run -e mpi check` in that directory builds it with MPICH and checks that 1, 2 and 4 ranks
and 1, 4 and 14 threads print the hash of a serial run.
