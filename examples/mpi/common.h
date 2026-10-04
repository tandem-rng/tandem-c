/* The problem shared by serial.c, mpi.c and omp.c: a global field of N uniform doubles and one
 * batch of normals per block. Every program must print the same hash. */
#ifndef EXAMPLE_COMMON_H
#define EXAMPLE_COMMON_H

#include "tandem.h"
#include <stdint.h>
#include <string.h>

#define N ((uint64_t)1 << 24)
/* The block grid is part of the problem, not of the machine, so it stays fixed for any number
 * of ranks or threads. */
#define BLOCKS 64
#define BLOCK (N / BLOCKS)
#define NORMALS 4096

/* Purposes give each use of randomness its own stream, so adding one leaves the others alone. */
enum { FIELD = 1, NOISE = 2 };

static inline tandem_rng root(void) { return tandem_seed(2026, 0, 0); }

/* FNV-1a over 32-bit words: enough to show that two runs wrote the same bits. */
static inline uint32_t fnv(uint32_t h, const void *data, size_t words) {
    const unsigned char *p = data;
    for (size_t i = 0; i < words; i++) {
        uint32_t w;
        memcpy(&w, p + 4 * i, 4);
        h = (h ^ w) * 16777619u;
    }
    return h;
}

#define FNV_START 2166136261u

/* Hash of block b: its slice x of the field, then its normals, from split(b) of the NOISE
 * stream. */
static inline uint32_t block_hash(const tandem_rng *noise, uint64_t b, const double *x) {
    double z[NORMALS];
    tandem_rng r = tandem_split(noise, b);
    tandem_fill_normal_f64(&r, z, NORMALS);
    return fnv(fnv(FNV_START, x, 2 * BLOCK), z, 2 * NORMALS);
}

static inline uint32_t combine(const uint32_t h[BLOCKS]) { return fnv(FNV_START, h, BLOCKS); }

#endif
