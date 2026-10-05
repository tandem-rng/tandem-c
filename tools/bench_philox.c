/* Random123 Philox4x32-10 fills, the GPU baseline, with the same bit-to-float maps as tandem.c.
 * Build: make bench. */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <Random123/philox.h>

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

typedef struct {
    philox4x32_key_t key;
    uint64_t block;
} philox_rng;

static inline philox4x32_ctr_t block(philox_rng *g) {
    philox4x32_ctr_t c = {{(uint32_t)g->block, (uint32_t)(g->block >> 32), 0, 0}};
    g->block++;
    return philox4x32(c, g->key);
}

/* The buffers are freed unread, so without this clang deletes the fills. */
#define KEEP(p) __asm__ volatile("" : : "r"(p) : "memory")

/* n is a multiple of 4 in every call below. */
static void fill_u32(philox_rng *g, uint32_t *out, size_t n) {
    for (size_t i = 0; i < n; i += 4) {
        philox4x32_ctr_t r = block(g);
        memcpy(out + i, r.v, sizeof r.v);
    }
    KEEP(out);
}

static void fill_u64(philox_rng *g, uint64_t *out, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        philox4x32_ctr_t r = block(g);
        out[i] = r.v[0] | (uint64_t)r.v[1] << 32;
        out[i + 1] = r.v[2] | (uint64_t)r.v[3] << 32;
    }
    KEEP(out);
}

static void fill_f32(philox_rng *g, float *out, size_t n) {
    for (size_t i = 0; i < n; i += 4) {
        philox4x32_ctr_t r = block(g);
        for (int j = 0; j < 4; j++) out[i + (size_t)j] = (float)(r.v[j] >> 8) * 0x1p-24f;
    }
    KEEP(out);
}

static void fill_f64(philox_rng *g, double *out, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        philox4x32_ctr_t r = block(g);
        uint64_t a = r.v[0] | (uint64_t)r.v[1] << 32, b = r.v[2] | (uint64_t)r.v[3] << 32;
        out[i] = (double)(a >> 11) * 0x1p-53;
        out[i + 1] = (double)(b >> 11) * 0x1p-53;
    }
    KEEP(out);
}

/* A scalar stream: one block feeds two f64 draws. */
typedef struct {
    philox_rng g;
    philox4x32_ctr_t buf;
    int used;
} philox_stream;

static inline double next_f64(philox_stream *s) {
    if (s->used == 4) {
        s->buf = block(&s->g);
        s->used = 0;
    }
    uint64_t w = s->buf.v[s->used] | (uint64_t)s->buf.v[s->used + 1] << 32;
    s->used += 2;
    return (double)(w >> 11) * 0x1p-53;
}

int main(void) {
    size_t n = (size_t)1 << 24;
    uint32_t *u32 = malloc(n * sizeof *u32);
    uint64_t *u64 = malloc(n * sizeof *u64);
    double *f64 = malloc(n * sizeof *f64);
    float *f32 = malloc(n * sizeof *f32);
    philox_rng g = {{{42, 0}}, 0};

    memset(u32, 0, n * sizeof *u32);
    memset(u64, 0, n * sizeof *u64);
    memset(f64, 0, n * sizeof *f64);
    memset(f32, 0, n * sizeof *f32);
    for (double t0 = now(); now() - t0 < 0.5;) fill_u32(&g, u32, n);

    BENCH("philox fill_u32", n * 4, fill_u32(&g, u32, n));
    BENCH("philox fill_u64", n * 8, fill_u64(&g, u64, n));
    BENCH("philox fill_f32", n * 4, fill_f32(&g, f32, n));
    BENCH("philox fill_f64", n * 8, fill_f64(&g, f64, n));
    philox_stream s = {g, {{0, 0, 0, 0}}, 4};
    volatile double sink = 0;
    BENCH("philox chain next_f64", n * 8, {
        double acc = 0;
        for (size_t i = 0; i < n; i++) acc += next_f64(&s);
        sink = acc;
    });
    (void)sink;
    free(u32);
    free(u64);
    free(f64);
    free(f32);
    return 0;
}
