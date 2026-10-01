/* Tandem8x32 reference implementation. See tandem.h and the specification. */
#include "tandem.h"

#include <string.h>

#define CLOCK_WEYL 0x9e3779b9u
#define DOMAIN_STREAM 0x9e3779b9u
#define DOMAIN_SPLIT 0xbb67ae85u
#define DOMAIN_FORK 0xd2511f53u
#define DOMAIN_FOLD 0xcd9e8d57u
#define DOMAIN_SEED 0xa54ff53au
#define AUX_STREAM 0x94d049bbu

static const uint32_t RC[8] = {0xd17cc1b7u, 0xa7220a94u, 0xfe13abe8u, 0xfa9a6ee0u,
                               0xedb14accu, 0x9e21c820u, 0xff28b1d5u, 0xef5de2b0u};

static inline uint32_t rotl(uint32_t x, unsigned r) {
    return (x << r) | (x >> ((32u - r) & 31u));
}

void tandem_T(uint32_t o[4], uint32_t h[4]) {
    uint64_t p0 = (uint64_t)o[0] * (h[0] | 1u);
    uint64_t p1 = (uint64_t)o[2] * (h[1] | 1u);
    uint32_t lo0 = (uint32_t)p0, hi0 = (uint32_t)(p0 >> 32);
    uint32_t lo1 = (uint32_t)p1, hi1 = (uint32_t)(p1 >> 32);
    uint32_t n0 = o[1] ^ hi1 ^ lo1;
    uint32_t n1 = rotl(lo1, 16) ^ h[2];
    uint32_t n2 = o[3] ^ hi0 ^ lo0;
    uint32_t n3 = rotl(lo0, 16) ^ h[3];

    h[0] ^= rotl(h[1], 7);
    h[1] ^= rotl(h[2], 13);
    h[2] ^= rotl(h[3], 22);
    h[3] ^= rotl(h[0], 3);
    h[0] = (h[0] + CLOCK_WEYL) ^ n0;

    o[0] = n0;
    o[1] = n1;
    o[2] = n2;
    o[3] = n3;
}

void tandem_F(uint32_t o[4], uint32_t h[4]) {
    for (int r = 0; r < 8; r++) {
        uint32_t t[4];
        tandem_T(o, h);
        o[0] ^= RC[r];
        memcpy(t, o, sizeof t);
        memcpy(o, h, sizeof t);
        memcpy(h, t, sizeof t);
    }
}

void tandem_F_keyed(const uint32_t key[4], uint64_t counter, uint32_t domain, uint32_t aux,
                    uint32_t o[4], uint32_t h[4]) {
    o[0] = (uint32_t)counter;
    o[1] = (uint32_t)(counter >> 32);
    o[2] = domain;
    o[3] = aux;
    memcpy(h, key, 4 * sizeof *key);
    tandem_F(o, h);
}

void tandem_block(const uint32_t key[4], uint64_t c, uint32_t j, uint32_t out[4]) {
    uint32_t h[4];
    tandem_F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, out, h);
    for (uint32_t s = 0; s <= j; s++) tandem_T(out, h);
}

/* Chunk index and step of a row. K is a power of two, so these are shifts and masks. */
static inline uint64_t row_group(const tandem_rng *rng, uint64_t row) { return row / rng->K; }
static inline uint32_t row_step(const tandem_rng *rng, uint64_t row) {
    return (uint32_t)(row & (rng->K - 1u));
}

/* Make the cache hold the eight blocks of `row`. Stepping forward inside the cached group
 * costs one T per lane per row. Anything else reseeds the group. */
static void load_row(tandem_rng *rng, uint64_t row) {
    if (rng->cached && rng->row == row) return;
    if (rng->cached && row > rng->row && row_group(rng, row) == row_group(rng, rng->row)) {
        for (uint64_t r = rng->row; r < row; r++)
            for (int l = 0; l < 8; l++) tandem_T(rng->o[l], rng->h[l]);
    } else {
        uint64_t g = row_group(rng, row);
        uint32_t j = row_step(rng, row);
        for (int l = 0; l < 8; l++) {
            tandem_F_keyed(rng->key, 8u * g + (uint64_t)l, DOMAIN_STREAM, AUX_STREAM, rng->o[l],
                           rng->h[l]);
            for (uint32_t s = 0; s <= j; s++) tandem_T(rng->o[l], rng->h[l]);
        }
    }
    rng->row = row;
    rng->cached = 1u;
}

static inline uint64_t align_pos(uint64_t pos, unsigned w) {
    return (pos + w - 1u) & ~((uint64_t)w - 1u);
}

/* Read w bits (8 <= w <= 64, a power of two) at an aligned position. */
static uint64_t read_bits(tandem_rng *rng, uint64_t p, unsigned w) {
    const uint32_t *blk;
    unsigned byte, nbytes = w / 8u;
    uint64_t raw = 0;
    load_row(rng, p >> 10);
    blk = rng->o[(p >> 7) & 7u];
    byte = (unsigned)((p >> 3) & 15u);
    for (unsigned i = 0; i < nbytes; i++) {
        unsigned b = byte + i;
        raw |= (uint64_t)((blk[b >> 2] >> (8u * (b & 3u))) & 0xffu) << (8u * i);
    }
    return raw;
}

static bool read_bit(tandem_rng *rng, uint64_t p) {
    load_row(rng, p >> 10);
    return (rng->o[(p >> 7) & 7u][(p >> 5) & 3u] >> (p & 31u)) & 1u;
}

static inline double to_f64(uint64_t raw) { return (double)(raw >> 11) * 0x1p-53; }
static inline float to_f32(uint32_t raw) { return (float)(raw >> 8) * 0x1p-24f; }

tandem_rng tandem_from_key(const uint32_t key[4], uint64_t pos, uint32_t K) {
    tandem_rng rng;
    memset(&rng, 0, sizeof rng);
    memcpy(rng.key, key, sizeof rng.key);
    rng.pos = pos;
    rng.K = K ? K : TANDEM_DEFAULT_K;
    return rng;
}

tandem_rng tandem_seed(uint64_t seed_lo, uint64_t seed_hi, uint32_t K) {
    uint32_t o[4] = {0, 0, DOMAIN_SEED, 0};
    uint32_t h[4] = {(uint32_t)seed_lo, (uint32_t)(seed_lo >> 32), (uint32_t)seed_hi,
                     (uint32_t)(seed_hi >> 32)};
    tandem_F(o, h);
    return tandem_from_key(o, 0, K);
}

void tandem_key(const tandem_rng *rng, uint32_t key[4]) { memcpy(key, rng->key, 4 * sizeof *key); }
uint64_t tandem_position(const tandem_rng *rng) { return rng->pos; }
uint32_t tandem_chunk_length(const tandem_rng *rng) { return rng->K; }

bool tandem_next_bool(tandem_rng *rng) {
    uint64_t p = rng->pos;
    rng->pos = p + 1u;
    return read_bit(rng, p);
}

#define NEXT_INT(name, type, w)                                                                    \
    type name(tandem_rng *rng) {                                                                   \
        uint64_t p = align_pos(rng->pos, w);                                                       \
        rng->pos = p + w;                                                                          \
        return (type)read_bits(rng, p, w);                                                         \
    }
NEXT_INT(tandem_next_u8, uint8_t, 8u)
NEXT_INT(tandem_next_u16, uint16_t, 16u)
NEXT_INT(tandem_next_u32, uint32_t, 32u)
NEXT_INT(tandem_next_u64, uint64_t, 64u)
#undef NEXT_INT

float tandem_next_f32(tandem_rng *rng) { return to_f32(tandem_next_u32(rng)); }
double tandem_next_f64(tandem_rng *rng) { return to_f64(tandem_next_u64(rng)); }

void tandem_fill_bool(tandem_rng *rng, bool *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = tandem_next_bool(rng);
}

/* A fill reads whole words straight from the cached row while it stays inside one block,
 * which covers every element except the first and last partial ones. */
#define FILL_INT(name, type, w)                                                                    \
    void name(tandem_rng *rng, type *out, size_t n) {                                              \
        uint64_t p = align_pos(rng->pos, w);                                                       \
        for (size_t i = 0; i < n; i++, p += w) out[i] = (type)read_bits(rng, p, w);                \
        rng->pos = p;                                                                              \
    }
FILL_INT(tandem_fill_u8, uint8_t, 8u)
FILL_INT(tandem_fill_u16, uint16_t, 16u)
FILL_INT(tandem_fill_u64, uint64_t, 64u)
#undef FILL_INT

void tandem_fill_u32(tandem_rng *rng, uint32_t *out, size_t n) {
    uint64_t p = align_pos(rng->pos, 32u);
    size_t i = 0;
    while (i < n) {
        const uint32_t *blk;
        unsigned word = (unsigned)((p >> 5) & 3u);
        load_row(rng, p >> 10);
        blk = rng->o[(p >> 7) & 7u];
        for (; word < 4u && i < n; word++, i++, p += 32u) out[i] = blk[word];
    }
    rng->pos = p;
}

void tandem_fill_f32(tandem_rng *rng, float *out, size_t n) {
    uint64_t p = align_pos(rng->pos, 32u);
    for (size_t i = 0; i < n; i++, p += 32u) out[i] = to_f32((uint32_t)read_bits(rng, p, 32u));
    rng->pos = p;
}

void tandem_fill_f64(tandem_rng *rng, double *out, size_t n) {
    uint64_t p = align_pos(rng->pos, 64u);
    for (size_t i = 0; i < n; i++, p += 64u) out[i] = to_f64(read_bits(rng, p, 64u));
    rng->pos = p;
}

/* Random access works on a copy so the caller's generator and cache stay untouched. */
static uint64_t at_bits(const tandem_rng *rng, uint64_t i, unsigned w) {
    tandem_rng tmp = *rng;
    uint64_t p = align_pos(rng->pos, w) + i * w;
    return read_bits(&tmp, p, w);
}

uint32_t tandem_at_u32(const tandem_rng *rng, uint64_t i) { return (uint32_t)at_bits(rng, i, 32u); }
uint64_t tandem_at_u64(const tandem_rng *rng, uint64_t i) { return at_bits(rng, i, 64u); }
float tandem_at_f32(const tandem_rng *rng, uint64_t i) { return to_f32(tandem_at_u32(rng, i)); }
double tandem_at_f64(const tandem_rng *rng, uint64_t i) { return to_f64(tandem_at_u64(rng, i)); }

static tandem_rng child(const tandem_rng *rng, uint64_t counter, uint32_t domain, uint32_t aux,
                        unsigned half) {
    uint32_t o[4], h[4];
    tandem_F_keyed(rng->key, counter, domain, aux, o, h);
    return tandem_from_key(half ? h : o, 0, rng->K);
}

tandem_rng tandem_split(const tandem_rng *rng, uint64_t index) {
    return child(rng, index >> 1, DOMAIN_SPLIT, 0, (unsigned)(index & 1u));
}

tandem_rng tandem_sub(const tandem_rng *rng, uint64_t purpose) {
    return child(rng, purpose, DOMAIN_FOLD, 0, 0);
}

void tandem_fork(tandem_rng *parent, tandem_rng *children, uint64_t n) {
    uint64_t b = parent->pos >> 7;
    for (uint64_t i = 0; i < n; i++)
        children[i] = child(parent, b, DOMAIN_FORK, (uint32_t)(i >> 1), (unsigned)(i & 1u));
    parent->pos = (b + 1u) << 7;
}
