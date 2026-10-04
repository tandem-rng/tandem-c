/* The OpenMP target fills equal the host fills bit for bit, from aligned and unaligned
 * positions, small and large sizes, and several chunk lengths. Build with
 * make test-target OMP_FLAGS="...". Memory comes from omp_target_alloc unless
 * TANDEM_HOST_MEMORY is defined, which runs the target regions on the host. */
#ifndef TANDEM_OPENMP_TARGET
#define TANDEM_OPENMP_TARGET
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_OPENMP) && !defined(TANDEM_HOST_MEMORY)
#define USE_OMP_DEVICE
#include <omp.h>
#endif

#include "../tandem.h"

static int failures;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                \
        }                                                                                          \
    } while (0)

static int device;

static void *dev_alloc(size_t bytes) {
#ifdef USE_OMP_DEVICE
    return omp_target_alloc(bytes, device);
#else
    return malloc(bytes);
#endif
}
static void dev_free(void *p) {
#ifdef USE_OMP_DEVICE
    omp_target_free(p, device);
#else
    free(p);
#endif
}
static void dev_get(void *host, const void *dev, size_t bytes) {
#ifdef USE_OMP_DEVICE
    omp_target_memcpy(host, dev, bytes, 0, 0, omp_get_initial_device(), device);
#else
    memcpy(host, dev, bytes);
#endif
}

typedef void (*host_fill)(tandem_rng *, void *, size_t);
typedef void (*dev_fill)(tandem_rng *, void *, size_t, int);

static void host_u32(tandem_rng *g, void *o, size_t n) { tandem_fill_u32(g, o, n); }
static void host_u64(tandem_rng *g, void *o, size_t n) { tandem_fill_u64(g, o, n); }
static void host_f32(tandem_rng *g, void *o, size_t n) { tandem_fill_f32(g, o, n); }
static void host_f64(tandem_rng *g, void *o, size_t n) { tandem_fill_f64(g, o, n); }
static void dev_u32(tandem_rng *g, void *o, size_t n, int d) { tandem_fill_u32_target(g, o, n, d); }
static void dev_u64(tandem_rng *g, void *o, size_t n, int d) { tandem_fill_u64_target(g, o, n, d); }
static void dev_f32(tandem_rng *g, void *o, size_t n, int d) { tandem_fill_f32_target(g, o, n, d); }
static void dev_f64(tandem_rng *g, void *o, size_t n, int d) { tandem_fill_f64_target(g, o, n, d); }

static void compare(const char *name, host_fill hf, dev_fill df, size_t es) {
    uint32_t key[4] = {1, 2, 3, 4};
    uint32_t Ks[] = {0, 1, 8, 4096};
    uint64_t starts[] = {0, 1, 33, 64, 1000, 1024, 5000, 123457};
    size_t sizes[] = {0, 1, 2, 3, 31, 32, 33, 255, 256, 1000, 4097, 100003, 1u << 20};
    size_t i, j, k;

    for (i = 0; i < sizeof Ks / sizeof Ks[0]; i++)
        for (j = 0; j < sizeof starts / sizeof starts[0]; j++)
            for (k = 0; k < sizeof sizes / sizeof sizes[0]; k++) {
                tandem_rng a = tandem_from_key(key, starts[j], Ks[i]), b = a;
                size_t n = sizes[k];
                char *want = malloc(n * es + 1), *got = malloc(n * es + 1);
                void *d = dev_alloc(n * es + 1);
                hf(&a, want, n);
                df(&b, d, n, device);
                dev_get(got, d, n * es);
                if (memcmp(want, got, n * es) != 0 || tandem_position(&a) != tandem_position(&b)) {
                    printf("FAIL %s K=%u start=%llu n=%zu\n", name, Ks[i],
                           (unsigned long long)starts[j], n);
                    failures++;
                }
                dev_free(d);
                free(want);
                free(got);
            }
}

int main(void) {
#ifdef USE_OMP_DEVICE
    device = omp_get_default_device();
    printf("devices: %d, using %d\n", omp_get_num_devices(), device);
#endif
    compare("u32", host_u32, dev_u32, 4);
    compare("u64", host_u64, dev_u64, 8);
    compare("f32", host_f32, dev_f32, 4);
    compare("f64", host_f64, dev_f64, 8);
    if (failures) {
        printf("%d failures\n", failures);
        return 1;
    }
    puts("target: ok");
    return 0;
}
