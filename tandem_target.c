/* OpenMP target offload fills for Tandem8x32. Build with -DTANDEM_OPENMP_TARGET and the
 * offload flags of your compiler, for example
 *   clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80
 *   nvc -mp=gpu -gpu=cc80
 * Without -fopenmp the pragmas are ignored and the same code runs on the host.
 *
 * Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE. */
#ifdef TANDEM_OPENMP_TARGET
#include <stdint.h>
#include <string.h>

#include "tandem.h"

#pragma omp declare target
#include "tandem123.h"

/* The same maps as tandem.c: the top 24 or 53 bits of a draw, scaled. */
static inline double target_to_f64(uint64_t raw) { return (double)(raw >> 11) * 0x1p-53; }
static inline float target_to_f32(uint32_t raw) { return (float)(raw >> 8) * 0x1p-24f; }
#pragma omp end declare target

enum { MODE_RAW, MODE_F32, MODE_F64 };

/* Elements of w bits at the aligned position pos, a byte range of the stream. The unit of work
 * is one chunk: a thread seeds its chunk, then steps it through the rows it has to write. A
 * row is 128 bytes, the eight chunks of a group at one step, and chunk l owns bytes 16 l to
 * 16 l + 15 of it. Elements outside the range, in the first and last rows, are skipped. */
static void fill_target(tandem_rng *rng, void *out, size_t n, unsigned w, int mode, int device) {
    uint64_t p = (rng->pos + w - 1u) & ~((uint64_t)w - 1u);
    uint64_t start = p >> 3, end = start + (uint64_t)n * (w / 8u);
    uint32_t K = rng->K, shift = 0;
    tandem4x32_key_t key = {{rng->key[0], rng->key[1], rng->key[2], rng->key[3]}};
    uint64_t r0, r1, g0, ngroups, threads;
    unsigned es = w / 8u;

    (void)device; /* unused when the pragma is ignored */
    rng->pos = p + (uint64_t)n * w;
    if (n == 0) return;
    while ((K >> shift) > 1u) shift++;
    r0 = start >> 7;
    r1 = (end - 1u) >> 7;
    g0 = r0 >> shift;
    ngroups = (r1 >> shift) - g0 + 1u;
    threads = 8u * ngroups;

#pragma omp target teams distribute parallel for is_device_ptr(out) device(device)
    for (uint64_t t = 0; t < threads; t++) {
        uint64_t g = g0 + t / 8u, lane = t % 8u;
        uint32_t o[4], h[4];
        tandem123_chunk(key, 8u * g + lane, o, h);
        for (uint32_t j = 0; j < K; j++) {
            uint64_t row = (g << shift) + j, base;
            tandem123_T(o, h);
            if (row < r0) continue;
            if (row > r1) break;
            base = row * 128u + lane * 16u;
            /* Two 8-byte stores for a whole row of the chunk: a warp writes a row of eight chunks
               with a 16-byte stride, so narrower stores cost more memory transactions. */
            if (base >= start && base + 16u <= end &&
                (((uintptr_t)out + (base - start)) & 7u) == 0) {
                uint64_t d0, d1;
                uint64_t *dst = (uint64_t *)((char *)out + (base - start));
                if (mode == MODE_F32) {
                    float f[4];
                    uint32_t v[4];
                    for (unsigned e = 0; e < 4; e++) f[e] = target_to_f32(o[e]);
                    memcpy(v, f, 16);
                    d0 = v[0] | ((uint64_t)v[1] << 32);
                    d1 = v[2] | ((uint64_t)v[3] << 32);
                } else if (mode == MODE_F64) {
                    double f[2];
                    f[0] = target_to_f64(o[0] | ((uint64_t)o[1] << 32));
                    f[1] = target_to_f64(o[2] | ((uint64_t)o[3] << 32));
                    memcpy(&d0, &f[0], 8);
                    memcpy(&d1, &f[1], 8);
                } else {
                    d0 = o[0] | ((uint64_t)o[1] << 32);
                    d1 = o[2] | ((uint64_t)o[3] << 32);
                }
                dst[0] = d0;
                dst[1] = d1;
                continue;
            }
            for (unsigned e = 0; e < 16u / es; e++) {
                uint64_t eb = base + (uint64_t)es * e, idx = (eb - start) / es;
                if (eb < start || eb + es > end) continue;
                if (mode == MODE_RAW && es == 4u) {
                    ((uint32_t *)out)[idx] = o[e];
                } else if (mode == MODE_F32) {
                    ((float *)out)[idx] = target_to_f32(o[e]);
                } else {
                    uint64_t v = o[2u * e] | ((uint64_t)o[2u * e + 1u] << 32);
                    if (mode == MODE_RAW) ((uint64_t *)out)[idx] = v;
                    else ((double *)out)[idx] = target_to_f64(v);
                }
            }
        }
    }
}

void tandem_fill_u32_target(tandem_rng *rng, uint32_t *out, size_t n, int device) {
    fill_target(rng, out, n, 32, MODE_RAW, device);
}
void tandem_fill_u64_target(tandem_rng *rng, uint64_t *out, size_t n, int device) {
    fill_target(rng, out, n, 64, MODE_RAW, device);
}
void tandem_fill_f32_target(tandem_rng *rng, float *out, size_t n, int device) {
    fill_target(rng, out, n, 32, MODE_F32, device);
}
void tandem_fill_f64_target(tandem_rng *rng, double *out, size_t n, int device) {
    fill_target(rng, out, n, 64, MODE_F64, device);
}
#endif /* TANDEM_OPENMP_TARGET */
