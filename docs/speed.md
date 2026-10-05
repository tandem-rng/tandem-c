# Speed

`make bench` produces the CPU figures, Philox included, and `make bench-target` the GPU figures.

## CPU

Apple M4 and AMD EPYC 7702P, one thread, `make bench`, clang `-O2`, minimum of seven runs of
2^24 elements, in GiB/s. The Philox and mt19937 columns run on the M4 and give the median of
three passes.

| | M4 | M4 with `TANDEM_NO_SIMD` | EPYC 7702P | Philox4x32-10 | mt19937 |
|---|---|---|---|---|---|
| `tandem_fill_u32` | 20.4 | 12.3 | 8.4 | 2.5 | 2.8 |
| `tandem_fill_u64` | 20.2 | 12.3 | 8.7 | 2.4 | 5.0 |
| `tandem_fill_f32` | 17.4 | 11.1 | 7.9 | 2.4 | 1.4 |
| `tandem_fill_f64` | 17.5 | 11.1 | 7.0 | 2.4 | 2.8 |
| `tandem_fill_normal_f64` | 7.7 | 5.9 | 3.6 | - | - |
| `tandem_fill_normal_f32` | 5.5 | 4.6 | 3.3 | - | - |
| `tandem_fill_exponential_f64` | 6.1 | 5.1 | 3.4 | - | - |
| `tandem_fill_exponential_f32` | 6.7 | 5.5 | 4.5 | - | - |
| `tandem_next_f64` chain, ns per draw | 1.40 | 1.77 | 4.05 | - | - |

The Philox column is `tools/bench_philox.c`: Random123 1.14.0 `philox4x32` with 10 rounds,
one block per call, written to the buffer with the bit-to-float maps of `tandem.c`. Clang does
not vectorize its 32x32-bit products, so it runs scalar. In the same passes `tandem_fill_u32`
ran at 19.1 GiB/s and `tandem_fill_f64` at 16.4. The mt19937 column is `std::mt19937` for u32
and f32 and `std::mt19937_64` for u64 and f64, with `std::uniform_real_distribution` for the
floats, from `tools/bench_std.cpp`.

On the EPYC the base copy, which `TANDEM_NO_AVX2` selects, reaches 6.2 GiB/s for
`tandem_fill_u32` and 2.9 for `tandem_fill_normal_f64`, whose library `fma` calls stay in the
rare slow path. The EPYC build is clang 20 at plain `-O2` with no `-m` flags, so it runs the
AVX2 copy picked at run time. With `-mavx2 -mfma` the f64 normal fill runs at the same speed.
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
C and C++ standard libraries on the M4 (Apple clang 21, libc++), in GiB/s. The standard
generators have no fill interface, so each writes one value per call.

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
| `rand()`, 31 bits per call into `uint32_t` | 1.0 |
| `random()`, 31 bits per call into `uint32_t` | 2.7 |
| `std::mt19937` with `std::uniform_real_distribution<float>` | 1.6 |
| `std::mt19937_64` with `std::uniform_real_distribution<double>` | 3.1 |
