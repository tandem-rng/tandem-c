/* Throughput of the sequential fills, the normal and exponential fills, and the scalar chain.
 * Build: make bench. */
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

#define BENCH(label, bytes, body)                                                                  \
    do {                                                                                           \
        double best = 1e9;                                                                         \
        for (int rep = 0; rep < 7; rep++) {                                                        \
            double t0 = now();                                                                     \
            body;                                                                                  \
            double t = now() - t0;                                                                 \
            if (t < best) best = t;                                                                \
        }                                                                                          \
        printf("%-28s %8.2f GiB/s\n", label, (double)(bytes) / best / (1024.0 * 1024.0 * 1024.0)); \
    } while (0)

int main(void) {
    size_t n = (size_t)1 << 24;
    uint32_t *u32 = malloc(n * sizeof *u32);
    uint64_t *u64 = malloc(n * sizeof *u64);
    double *f64 = malloc(n * sizeof *f64);
    float *f32 = malloc(n * sizeof *f32);
    tandem_rng rng = tandem_seed(42, 0, 0);
    volatile double sink = 0;

    /* Touch every buffer and run for half a second so the clock has ramped up. */
    memset(u32, 0, n * sizeof *u32);
    memset(u64, 0, n * sizeof *u64);
    memset(f64, 0, n * sizeof *f64);
    memset(f32, 0, n * sizeof *f32);
    for (double t0 = now(); now() - t0 < 0.5;) tandem_fill_u32(&rng, u32, n);

    BENCH("fill_u32", n * 4, tandem_fill_u32(&rng, u32, n));
    BENCH("fill_u64", n * 8, tandem_fill_u64(&rng, u64, n));
    BENCH("fill_f32", n * 4, tandem_fill_f32(&rng, f32, n));
    BENCH("fill_f64", n * 8, tandem_fill_f64(&rng, f64, n));
    BENCH("fill_normal_f64", n * 8, tandem_fill_normal_f64(&rng, f64, n));
    BENCH("fill_normal_f32", n * 4, tandem_fill_normal_f32(&rng, f32, n));
    BENCH("fill_exponential_f64", n * 8, tandem_fill_exponential_f64(&rng, f64, n));
    BENCH("fill_exponential_f32", n * 4, tandem_fill_exponential_f32(&rng, f32, n));
    BENCH("chain next_f64", n * 8, {
        double acc = 0;
        for (size_t i = 0; i < n; i++) acc += tandem_next_f64(&rng);
        sink = acc;
    });
    (void)sink;
    free(u32);
    free(u64);
    free(f64);
    free(f32);
    return 0;
}
