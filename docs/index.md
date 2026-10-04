# tandem-c documentation

- [API](api.md): the C and C++ interface, the counter-based header, parallel use and OpenMP
  target offload.
- [Design](design.md): how the fills, bounded integers, normals and exponentials work.
- [Tests](tests.md): what the suite checks and how to regenerate the fixtures.
- [Speed](speed.md): CPU and offload figures.

## Install

```sh
make                              # libtandem.a, built with clang
make install PREFIX=<prefix>      # libtandem.a, tandem.h, tandem.hpp, tandem.pc
```

Or compile `tandem.c` into your project. `DESTDIR` stages the installed files.

The library needs C11 and C++17 and libm, and nothing newer, so projects that vendor it can
keep their own flags. `make` builds it as C23 and C++23 with clang, the primary compiler. Set
`CC` and `CXX` for another compiler. A separate CI job compiles with strict
`-std=c11 -pedantic-errors` and `-std=c++17`.

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
