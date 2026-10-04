# Design

## Bounded integers

Bounded integers `tandem_u32_below` and `tandem_u64_below` (Lemire) and normals
`tandem_normal_f64` and `tandem_normal_f32` (Box-Muller, which needs `-lm`) go beyond the
specification and return the same values as tandem-cuda. The scalar bounded draws loop on
rejection. The bounded fills use the parallel rule of tandem-cuda instead: element i maps
draw i of the plain fill, which keeps the SIMD speed, and a rejected draw retries on a
fallback generator keyed by the global draw index, so a fill consumes exactly `len` draws and
a fill cut anywhere equals the whole fill.

## Normals

A Box-Muller step gives two normals: `tandem_normal2_f64` and `_f32` return both (cos half
first), `tandem_normal_*` returns the cos half, and `tandem_fill_normal_*` writes the
flattened pairs, so an odd count still consumes `2 * ceil(n / 2)` uniforms. The f32 normals
draw two f32 uniforms and compute the radius in float, so they agree across ports to a few
ulps because float libm functions differ. Uniforms are bit exact and f64 normals agree to
about 1e-12 relative.

The normal fills and draws share one loop without libm calls: polynomials for the logarithm
of the exponent-split argument and for the sine and cosine of the angle after an exact
quarter-turn reduction, which the compiler vectorizes. `make accuracy` compares the fills
with libm on 5e7 pairs. The f64 normals agree to 1e-15 relative and the f32 normals to 3.3
ulps. `make bench` reports the normal fills too.

Every multiply-add in that loop is an explicit `fma`, and the loop is built with floating
point contraction off, so every compiler and target produces the same bits:
`tests/test_normal_bits.c` checks a hash of 10^7 normals against the value from the M4.

## Exponentials

Exponentials `tandem_exponential_f64` and `_f32` and the fills `tandem_fill_exponential_f64`
and `_f32` return `-ln(1 - u)` from one uniform `u` each, as Appendix A of the specification
describes: element i of a fill comes from uniform i, so a fill equals the scalar draws and a
fill cut anywhere equals the whole fill. They use the logarithm of the normal loop, with no
libm call, and tandem-cuda runs the same arithmetic on the device, so exponentials are bit
exact across compilers, targets and devices. The maximum error is 1.1e-15 relative for f64
and 2.8e-7 for f32, checked over all 2^24 f32 inputs.

## SIMD

On x86-64 with GCC 12+ or clang, `tandem.c` compiles the row loop and the normal and
exponential loops a second time for AVX2 and FMA and picks that copy at run time. A plain
`-O2` build with no `-m` flags so gets 256-bit rows and hardware fused multiply-adds on
Haswell, Zen and newer, and still runs on older CPUs, where `fma` is a library call that is
correct but about seven times slower. With `-mavx2 -mfma` only the AVX2 copy remains.
Define `TANDEM_NO_AVX2` to leave it out. Both copies give the same bits.

The eight chunks of a row step together in registers. With GCC 12+ or clang the step is
written with vector extensions and compiles to NEON, SSE2 or, in the AVX2 copy, one 256-bit
vector per word. Define `TANDEM_NO_SIMD` for the scalar version. After alignment every
integer fill is one byte stream, so one routine serves all widths and floats convert in
place.
