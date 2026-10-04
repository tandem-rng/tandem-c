# Tests

```sh
make test
```

- The specification vectors in `tests/vectors.h`, generated from the spec's `vectors.json`.
- Long fills, scalar draws and random access against the stream dumps in `tests/data`.
- Bounded integers, normals and exponentials against fixtures from the tandem-cuda core,
  with hashes of 10^7 normals and 10^6 exponentials. `make cross` regenerates them.
- `tandem123.h` against the fills, and the C++ wrapper against the C API.
- `make test-target` checks the OpenMP target fills against the host fills.

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

`tests/test_target.c` compares the four OpenMP target fills bit for bit with the host fills
over four chunk lengths, aligned and unaligned starts, and sizes from 0 to 2^20.
