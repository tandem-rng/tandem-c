#!/usr/bin/env bash
set -euo pipefail

# The library needs C11 only. The Makefile default asks for C23 and clang.
make libtandem.a CC="${CC}" CFLAGS="${CFLAGS:-} -std=c11 -O2 -fPIC"
make install PREFIX="${PREFIX}" VERSION="${PKG_VERSION}"
