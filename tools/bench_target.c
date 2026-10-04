/* Throughput of the OpenMP target fills, output stays in device memory. Build: make bench-target
 * OMP_FLAGS="...". Run it on an idle device. */
#define _POSIX_C_SOURCE 199309L
#ifndef TANDEM_OPENMP_TARGET
#define TANDEM_OPENMP_TARGET
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if defined(_OPENMP) && !defined(TANDEM_HOST_MEMORY)
#define USE_OMP_DEVICE
#include <omp.h>
#endif

#include "../tandem.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

#define BENCH(label, bytes, call)                                                                     \
    do {                                                                                           \
        double best = 1e9;                                                                         \
        call; /* warm-up: first launch loads the kernel */                                         \
        for (int rep = 0; rep < 10; rep++) {                                                       \
            double t0 = now();                                                                     \
            call;                                                                                  \
            double t = now() - t0;                                                                 \
            if (t < best) best = t;                                                                \
        }                                                                                          \
        printf("%-22s %8.1f GiB/s\n", label, (double)(bytes) / best / (1024.0 * 1024.0 * 1024.0)); \
    } while (0)

int main(void) {
    size_t n = (size_t)1 << 28;
    int device = 0;
    tandem_rng rng = tandem_seed(42, 0, 0);
#ifdef USE_OMP_DEVICE
    device = omp_get_default_device();
    void *buf = omp_target_alloc(n * 8, device);
#else
    void *buf = malloc(n * 8);
#endif
    BENCH("fill_u32_target", n * 4, tandem_fill_u32_target(&rng, buf, n, device));
    BENCH("fill_u64_target", n * 4, tandem_fill_u64_target(&rng, buf, n / 2, device));
    BENCH("fill_f32_target", n * 4, tandem_fill_f32_target(&rng, buf, n, device));
    BENCH("fill_f64_target", n * 4, tandem_fill_f64_target(&rng, buf, n / 2, device));
    return 0;
}
