# Recipe for a Spack package repository. It is not submitted to spack-packages yet.
from spack_repo.builtin.build_systems.makefile import MakefilePackage

from spack.package import *


class TandemC(MakefilePackage):
    """Reference C implementation of the Tandem8x32 random number generator."""

    homepage = "https://github.com/tandem-rng/tandem-c"
    git = "https://github.com/tandem-rng/tandem-c.git"

    license("Apache-2.0")

    # No releases exist. A release adds
    # version("X.Y.Z", sha256="...", url="https://github.com/tandem-rng/tandem-c/archive/refs/tags/vX.Y.Z.tar.gz")
    version("main", branch="main")

    depends_on("c", type="build")
    depends_on("pkgconfig", type="build")

    @property
    def build_targets(self):
        # The library needs C11 only. The default flags ask for C23, which gcc 13 lacks.
        return ["libtandem.a", f"CC={spack_cc}", "CFLAGS=-std=c11 -O2 -fPIC"]

    @property
    def install_targets(self):
        return ["install", f"PREFIX={self.prefix}", f"CC={spack_cc}"]

    def check(self):
        # The test suite needs the sibling tandem-spec and tandem-cuda checkouts.
        pass

    @run_after("install")
    def check_pkgconfig(self):
        pkgconf = which("pkg-config") or which("pkgconf")
        with working_dir(self.prefix):
            pkgconf("--exists", "tandem", env={"PKG_CONFIG_PATH": self.prefix.lib.pkgconfig})
