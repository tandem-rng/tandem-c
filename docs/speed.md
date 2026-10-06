# Speed

`make bench` produces the CPU figures, Philox included, and `make bench-target` the GPU figures.

## CPU

Apple M4 and AMD EPYC 7702P, one thread, `make bench`, clang `-O2`, minimum of seven runs of
2^24 elements, in GiB/s. Each machine's columns come from one session and give the median of
three passes. The EPYC session ran on one pinned core.

| | M4 | M4 with `TANDEM_NO_SIMD` | mt19937, M4 | Philox4x32-10, M4 |
|---|---|---|---|---|
| `tandem_fill_u32` | 19.0 | 11.8 | 2.7 | 2.5 |
| `tandem_fill_u64` | 19.1 | 11.7 | 5.0 | 2.4 |
| `tandem_fill_f32` | 16.4 | 10.7 | 2.8 | 2.4 |
| `tandem_fill_f64` | 16.5 | 10.6 | 5.1 | 2.4 |
| `tandem_fill_normal_f64` | 7.6 | 5.8 | 0.90 | - |
| `tandem_fill_normal_f32` | 5.5 | 4.6 | 0.67 | - |
| `tandem_fill_exponential_f64` | 6.1 | 5.0 | 1.1 | - |
| `tandem_fill_exponential_f32` | 6.7 | 5.4 | 1.6 | - |
| `tandem_next_f64` chain | 5.2 | 4.1 | 5.5 | 2.2 |

| | EPYC 7702P | mt19937, EPYC | Philox4x32-10, EPYC |
|---|---|---|---|
| `tandem_fill_u32` | 8.6 | 1.2 | 1.1 |
| `tandem_fill_u64` | 8.6 | 2.3 | 1.1 |
| `tandem_fill_f32` | 7.7 | 1.1 | 1.0 |
| `tandem_fill_f64` | 6.9 | 2.1 | 0.97 |
| `tandem_fill_normal_f64` | 3.6 | 0.42 | - |
| `tandem_fill_normal_f32` | 3.2 | 0.30 | - |
| `tandem_fill_exponential_f64` | 3.4 | 0.65 | - |
| `tandem_fill_exponential_f32` | 4.5 | 0.48 | - |
| `tandem_next_f64` chain | 1.9 | 1.9 | 0.96 |

The mt19937 column is `std::mt19937` for u32 and f32 and `std::mt19937_64` for u64 and f64,
with `std::uniform_real_distribution`, `std::normal_distribution` and
`std::exponential_distribution` of libc++ for the floats, from `tools/bench_std.cpp`. On the EPYC
the same file built against libstdc++ ran about twice as fast as against libc++ on every row, so
that column is libstdc++. Random123's Box-Muller on Philox wrote fewer normals in the same session. The Philox column is
`tools/bench_philox.c`: Random123 1.14.0 `philox4x32` with 10 rounds, one block per call,
written to the buffer with the bit-to-float maps of `tandem.c`. Clang does not vectorize its
32x32-bit products, so it runs scalar. The chain row sums 2^24 scalar draws, 8 bytes each, and
Philox draws two f64 from each block. `tandem_next_f64` is a call into `tandem.c` per draw,
while the mt19937_64 draw is inlined, which is why mt19937_64 leads that row.

The EPYC build is conda-forge clang 23.1 at plain `-O2` with no `-m` flags, so it runs the AVX2
copy picked at run time. The base copy, which `TANDEM_NO_AVX2` selects, ran at the same speed in
the same session, 8.7 GiB/s for `tandem_fill_u32` and 3.6 for `tandem_fill_normal_f64`, whose
library `fma` calls stay in the rare slow path.
GCC 12 at `-O2` reaches 9.1 GiB/s for `tandem_fill_u32` and 3.2 for `tandem_fill_normal_f64`,
but only 0.55 for `tandem_fill_normal_f32`. GCC honours `-fno-math-errno` only on the command
line, not as a function attribute, so `sqrt` keeps its errno branch and the Box-Muller loop
stays scalar.

The normal and exponential rows count the bytes written. The f32 normal and the exponential
fills run their vectorized loops after the float fill, so they depend on `TANDEM_NO_SIMD`
only through the uniforms. The f64 normal fill runs a table pass after the u64 fill, which
`TANDEM_NO_SIMD` slows. Before the ziggurat, the Box-Muller f64 normal fill ran at 5.0 GiB/s
on the M4 and 2.4 on the EPYC.

The row loop keeps the eight lane states in registers and stores each row by a vector
transpose, which is where the throughput comes from. Float fills map the words to floats in
the same loop, before the store. On AArch64 the fixed-point `ucvtf` does the shift and the
scale in one instruction. The 32x32 to 64-bit products are NEON `umull` and SSE2 `pmuludq`
through intrinsics: from the portable spelling, a widened 64-bit vector multiply, GCC emits
scalar multiplies on AArch64 and three `pmuludq` per product on x86. Clang gives the
figures in the table. GCC 16 at `-O2` reaches 14.5 GiB/s for `tandem_fill_u32` on the same
machine and 16.1 at `-O3`, because it keeps the lane states in memory for part of the row
loop. The scalar fallback is one straight-line step per lane in a loop over the eight lanes.
Clang vectorizes that loop, GCC 16 does not at `-O2` and reaches 5.6 GiB/s.

## GPU

OpenMP target offload on an A100 40 GB PCIe (driver 570, GPU idle before each run) with nvc
25.3: `make bench-target` writes 2^28 elements into device memory at 480 to 850 GiB/s over ten runs, median about 570,
the same for all four types, against 1386 GiB/s for the CUDA kernels of tandem-cuda. Build
with `OMP_FLAGS="-mp=gpu -gpu=cc80" CC=nvc`.

## Other generators

`make bench` also runs `tools/bench_std.cpp`, the C++ wrapper against the generators of the
C and C++ standard libraries on the M4 (Apple clang 21, libc++), in GiB/s, from the same
session as the CPU table. The standard generators have no fill interface, so each writes one
value per call. Each loop has its own generator. The mt19937 figures move with code layout: in a
program with only that loop, `std::mt19937_64` with `std::uniform_real_distribution<double>`
reaches 8.2 GiB/s.

| | GiB/s |
|---|---|
| `tandem::rng::fill<uint32_t>` | 19.2 |
| `tandem::rng::fill<uint64_t>` | 19.0 |
| `tandem::rng::fill<float>` | 16.5 |
| `tandem::rng::fill<double>` | 16.5 |
| `tandem::rng::next<double>` chain | 5.1 |
| `std::uniform_real_distribution<double>` on `tandem::rng` | 5.1 |
| `std::mt19937`, `uint32_t` | 2.7 |
| `std::mt19937_64`, `uint64_t` | 5.0 |
| `std::mt19937_64` with `std::generate_canonical<double, 53>` | 5.1 |
| `std::mt19937_64` with `std::uniform_real_distribution<double>`, chain | 5.5 |
| `arc4random_buf` | 4.5 |
| `rand()`, 31 bits per call into `uint32_t` | 0.92 |
| `random()`, 31 bits per call into `uint32_t` | 2.5 |
| `std::mt19937` with `std::uniform_real_distribution<float>` | 2.8 |
| `std::mt19937_64` with `std::uniform_real_distribution<double>` | 5.1 |
| `std::mt19937_64` with `std::normal_distribution<double>` | 0.90 |
| `std::mt19937` with `std::normal_distribution<float>` | 0.67 |
| `std::mt19937_64` with `std::exponential_distribution<double>` | 1.1 |
| `std::mt19937` with `std::exponential_distribution<float>` | 1.6 |
