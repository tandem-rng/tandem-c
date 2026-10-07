/* Scalar draws, one call per value, against xoshiro256++ inlined into the same loop. The u64
 * rows sum integers, the f64 rows sum doubles, whose add latency bounds them. Build: make bench. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../tandem.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* xoshiro256++ 1.0 by David Blackman and Sebastiano Vigna, public domain reference code from
 * https://prng.di.unimi.it/xoshiro256plusplus.c, the generator of Rust's SmallRng. */
static uint64_t xs[4] = {0x9e3779b97f4a7c15u, 0xbf58476d1ce4e5b9u, 0x94d049bb133111ebu,
                         0x2545f4914f6cb7a9u};
static inline uint64_t xrotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t xoshiro_next(uint64_t *s) {
    const uint64_t result = xrotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = xrotl(s[3], 45);
    return result;
}
static inline double xoshiro_f64(uint64_t *s) { return (double)(xoshiro_next(s) >> 11) * 0x1p-53; }

#define CHAIN(label, type, draw)                                                                   \
    do {                                                                                           \
        double best = 1e9;                                                                         \
        for (int rep = 0; rep < 7; rep++) {                                                        \
            double t0 = now();                                                                     \
            type acc = 0;                                                                          \
            for (size_t i = 0; i < n; i++) acc += draw;                                            \
            double t = now() - t0;                                                                 \
            sink += (double)acc;                                                                   \
            if (t < best) best = t;                                                                \
        }                                                                                          \
        printf("%-28s %8.2f GiB/s %7.3f ns\n", label, 8.0 * (double)n / best / 1073741824.0,      \
               1e9 * best / (double)n);                                                            \
    } while (0)

int main(void) {
    size_t n = (size_t)1 << 24;
    uint32_t *buf = malloc(n * sizeof *buf);
    tandem_rng rng = tandem_seed(42, 0, 0);
    uint64_t s[4];
    volatile double sink = 0;

    memcpy(s, xs, sizeof s);
    /* Run for half a second so the clock has ramped up. */
    for (double t0 = now(); now() - t0 < 0.5;) tandem_fill_u32(&rng, buf, n);

    /* Through a pointer to the exported function, as a call from another language makes it. */
    double (*volatile next_f64_call)(tandem_rng *) = tandem_next_f64;
    double (*f64_call)(tandem_rng *) = next_f64_call;

    CHAIN("chain next_u64", uint64_t, tandem_next_u64(&rng));
    CHAIN("chain next_f64", double, tandem_next_f64(&rng));
    CHAIN("chain next_f64, called", double, f64_call(&rng));
    CHAIN("chain xoshiro256++ u64", uint64_t, xoshiro_next(s));
    CHAIN("chain xoshiro256++ f64", double, xoshiro_f64(s));
    free(buf);
    return 0;
}
