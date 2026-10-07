# Tests

```sh
make test
make stats                # the distribution of 10^8 f64 normals, about 15 s
make test-target          # the OpenMP target fills against the host fills
```

## Suite

- The specification vectors in `tests/vectors.h`, generated from the spec's `vectors.json`.
- Long fills, scalar draws and random access against the stream dumps in `tests/data`.
- Bounded integers, f32 normals and exponentials against fixtures from the tandem-cuda core.
  f64 normals against this library's published fixture and a Python implementation of the
  spec. Hashes of normals and exponentials.
- `tandem123.h` against the fills, and the C++ wrapper against the C API.
- `make test-target` checks the OpenMP target fills against the host fills.

`tests/test_vectors.c` checks every vector of the specification. `tests/test_stream.c`
compares long fills, scalar draws, and random access against reference stream dumps in
`tests/data`. `tests/test_api.c` checks the functions that are not part of the specification's
draws. The bounded integers (scalar and fill) and f32 normals are compared, values and stream
position, with fixtures from the shared core of tandem-cuda. The f32 normals match bit for bit,
because the host core runs the same explicit-fma loop. Device f32 normals from the
device-derived fill fixtures of tandem-cuda match within 16 ulps, and their f64 ziggurat rows
match bit for bit. Exponential fills and
scalar draws must match the same core bit for bit from five start positions, unaligned ones
included.

`tests/test_api.c` also checks that an f64 fill equals the scalar draws and that fills cut at
odd elements and at a missed element equal the whole fill, over 40000 elements, and that an
empty fill only aligns the position. `tests/test_normal_bits.c` checks the FNV-1a hash
`a61cfa844c85f7c1` of 10^6 f64 normals from each of five positions. It also checks 2 x 10^5
f64 normals from two positions against a Python implementation written from the text of
Appendix A, and the FNV-1a hash `aa1ea656ce73a4fb` of the f32 normals from the five positions.
`tests/test_normal_stats.c` checks 10^8 f64 normals against the standard normal: raw moments 1
to 6 and the counts beyond 3, 3.5, 4, 4.5 and 5 within four standard errors, and
Kolmogorov-Smirnov and Anderson-Darling p-values above 0.001.

`tests/test_api.c` also cuts exponential fills at several elements and compares them with the
whole fill and the scalar draws, and checks 10^7 f64 and 10^7 f32 exponentials against Exp(1):
the first four raw moments within five standard errors and a Kolmogorov-Smirnov statistic
below the 0.1 % point. `tests/test_exponential_bits.c` checks the FNV-1a hash
`1c761a2d471073c2` of 10^6 f64 and 10^6 f32 exponentials from each of five positions.

`tests/test_cpp.cpp` checks that the C++ wrapper, including `at`, `below`, `normal`, `normal2`,
`exponential`, `set_position` and the extra draw types, agrees with the C API and runs
`<random>`. Compiled as C++20 it also checks `std::uniform_random_bit_generator`.

`tests/test_target.c` compares the four OpenMP target fills bit for bit with the host fills
over four chunk lengths, aligned and unaligned starts, and sizes from 0 to 2^20.

## Fixtures

`tests/vectors.h` is generated from the spec repository's `vectors.json` by
`tools/gen_vectors.py`. The stream dumps in `tests/data` are written by
`tools/dump_streams.jl`.

`tools/gen_cross.cpp` computes the bounded integer, f32 normal and exponential fixtures from
the shared core of [tandem-cuda](https://github.com/tandem-rng/tandem-cuda).
`tests/cross_exponential.h` holds exponentials of that core from five start positions,
unaligned ones included. The device-derived fill fixtures of tandem-cuda are copied to
`tests/cuda_fill_*.h`. `make cross` regenerates the fixtures.

This library is the reference of the f64 normals. `tools/gen_cross.cpp` writes their rows of
`tests/cross_normal.h` from it: fills of 64 elements from the key of seed 42 at bits 0, 1 and
12345, and at three more unaligned starts that put a wedge accept, a wedge reject and a tail
draw at element 20. Ports check against this file, whose SHA-256 is
`3cd7c8f9178711255718288eb712eaccb33a1726d2a185f412f13590398ad3ac`.

This library is also the reference of weighted choice, Appendix C. `tests/cross_choice.h` holds
fills of 64 indices from six weight tables, zero, subnormal and near-overflow weights and 100
weights among them, from the key of seed 42 at bits 0, 1 and 12345. Its SHA-256 is
`73c0badade569b913eb01a2883ae4ae995e90f5b04e8bb7adaa3f42a957eb0bc`. The spec's choice vectors,
which `tests/test_vectors.c` checks, were computed apart from this library from the spec text and
the stream dump. `tests/test_api.c` runs a chi-square test on 10^7 indices.

`tools/dump_normals.c` writes the bytes that `tests/test_normal_bits.c` hashes, whose SHA-256
is `700ec4d2f4d6b82aaa56c6eff18a4e5919585fdbd093988773383d580ea610d1`.
`tools/dump_exponentials.c` writes the bytes that `tests/test_exponential_bits.c` hashes, whose
SHA-256 is `7b12b7c36baf14ab42f7736a5a67925c3c7b1bfafc8d50f078ba1af927044dd5`.
`tools/dump_derived.c` maps the derived draws back to uniform bits for PractRand and TestU01.
The results are in `docs/statistics.md` of the spec repository.

## CI

- Ubuntu with the latest clang, the primary compiler, and the latest gcc, as a compatibility
  check. macOS with Apple clang. Each runs `make test` as C23 and C++23 with warnings as errors.
- Ubuntu with the latest clang under AddressSanitizer and UndefinedBehaviorSanitizer.
- A separate job compiles with strict `-std=c11 -pedantic-errors` and `-std=c++17`, with clang
  and with gcc.
- On x86-64 CI runs the suite three times: plain `-O2`, which takes the AVX2 copy on the
  runner, `-DTANDEM_NO_AVX2` for the base copy, and `-mavx2 -mfma`.
- One job runs `tests/test_normal_stats.c`. CI checks that `make vectors tables` reproduces
  `tests/vectors.h` and `tandem_normal_tables.h` from the spec.
- The OpenMP target job runs `make test-target` on the host with the latest clang and
  attempts the nvptx offload compile.
- The parallel example job checks that 1, 2 and 4 MPI ranks and 1, 4 and 14 OpenMP threads
  print the hash of a serial run.
