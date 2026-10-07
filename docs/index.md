# tandem-c

Reference C and C++ implementation of Tandem8x32. It produces the stream of the
[specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) bit for bit, with SIMD
fills on CPUs and OpenMP target fills on GPUs.

- [API](api.md): the C and C++ interface, the counter-based header, OpenMP target offload and
  parallel use.
- [Design](design.md): how the fills, bounded integers, normals and exponentials work.
- [Tests](tests.md): what the suite checks, where the fixtures come from, and what CI runs.
- [Speed](speed.md): CPU and offload figures, and the standard generators.

## Install

```sh
make                              # libtandem.a, built with clang
make install PREFIX=<prefix>      # libtandem.a, tandem.h, tandem.hpp, tandem.pc
```

Or compile `tandem.c` into your project, with `tandem_normal_tables.h` next to it. `DESTDIR`
stages the installed files.

The library needs C11 and C++17 and libm, and nothing newer, so projects that vendor it can
keep their own flags, with one exception: the floating-point model must stay strict. The
Float32 exponential carries its logarithm as a high and a low part, and a compiler that
reassociates or contracts those sums changes the values. Build `tandem.c` with
`-ffp-contract=off` and without fast-math. The Intel compiler `icx` defaults to
`-fp-model=fast`, so pass `-fp-model=precise -ffp-contract=off` there. `make` builds it as C23
and C++23 with clang, the primary compiler. Set `CC` and `CXX` for another compiler.

The `packaging/` directory holds a Spack recipe (`spack/package.py`) and a conda-forge style
recipe (`conda/recipe.yaml`). Neither is submitted to Spack or conda-forge yet, and both build
from the `main` branch.

## AI assistance

This implementation was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
