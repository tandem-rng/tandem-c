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

/* The row cache is word-major: o[w][lane]. One T over all eight lanes is then four
 * element-wise vector operations per word, which is how the GPU and the Julia Lane8 core
 * see it. With GCC or clang the step uses vector extensions, four lanes at a time, so it
 * compiles to NEON or SSE/AVX without intrinsics. */
#if defined(__GNUC__) && !defined(TANDEM_NO_SIMD)
typedef uint32_t u32x4 __attribute__((vector_size(16)));
typedef uint64_t u64x4 __attribute__((vector_size(32)));

static inline u32x4 vrotl(u32x4 x, unsigned r) { return (x << r) | (x >> (32u - r)); }

static inline void T4(uint32_t o[4][8], uint32_t h[4][8], unsigned half) {
    u32x4 o0, o1, o2, o3, h0, h1, h2, h3;
    u64x4 p0, p1;
    u32x4 lo0, hi0, lo1, hi1, n0, n1, n2, n3;
    unsigned l = 4u * half;
    memcpy(&o0, &o[0][l], 16);
    memcpy(&o1, &o[1][l], 16);
    memcpy(&o2, &o[2][l], 16);
    memcpy(&o3, &o[3][l], 16);
    memcpy(&h0, &h[0][l], 16);
    memcpy(&h1, &h[1][l], 16);
    memcpy(&h2, &h[2][l], 16);
    memcpy(&h3, &h[3][l], 16);

    p0 = __builtin_convertvector(o0, u64x4) * __builtin_convertvector(h0 | 1u, u64x4);
    p1 = __builtin_convertvector(o2, u64x4) * __builtin_convertvector(h1 | 1u, u64x4);
    lo0 = __builtin_convertvector(p0, u32x4);
    hi0 = __builtin_convertvector(p0 >> 32, u32x4);
    lo1 = __builtin_convertvector(p1, u32x4);
    hi1 = __builtin_convertvector(p1 >> 32, u32x4);
    n0 = o1 ^ hi1 ^ lo1;
    n1 = vrotl(lo1, 16) ^ h2;
    n2 = o3 ^ hi0 ^ lo0;
    n3 = vrotl(lo0, 16) ^ h3;

    h0 ^= vrotl(h1, 7);
    h1 ^= vrotl(h2, 13);
    h2 ^= vrotl(h3, 22);
    h3 ^= vrotl(h0, 3);
    h0 = (h0 + CLOCK_WEYL) ^ n0;

    memcpy(&o[0][l], &n0, 16);
    memcpy(&o[1][l], &n1, 16);
    memcpy(&o[2][l], &n2, 16);
    memcpy(&o[3][l], &n3, 16);
    memcpy(&h[0][l], &h0, 16);
    memcpy(&h[1][l], &h1, 16);
    memcpy(&h[2][l], &h2, 16);
    memcpy(&h[3][l], &h3, 16);
}

static void T8(uint32_t o[4][8], uint32_t h[4][8]) {
    T4(o, h, 0);
    T4(o, h, 1);
}
#else
static void T8(uint32_t o[4][8], uint32_t h[4][8]) {
    for (unsigned l = 0; l < 8; l++) {
        uint32_t ol[4] = {o[0][l], o[1][l], o[2][l], o[3][l]};
        uint32_t hl[4] = {h[0][l], h[1][l], h[2][l], h[3][l]};
        tandem_T(ol, hl);
        for (unsigned w = 0; w < 4; w++) {
            o[w][l] = ol[w];
            h[w][l] = hl[w];
        }
    }
}
#endif

static inline uint64_t row_group(const tandem_rng *rng, uint64_t row) { return row / rng->K; }
static inline uint32_t row_step(const tandem_rng *rng, uint64_t row) {
    return (uint32_t)(row & (rng->K - 1u));
}

/* Make the cache hold the eight blocks of `row`. Stepping forward inside the cached group
 * costs one T8 per row. Anything else reseeds the group. */
static void load_row(tandem_rng *rng, uint64_t row) {
    if (rng->cached && rng->row == row) return;
    if (rng->cached && row > rng->row && row_group(rng, row) == row_group(rng, rng->row)) {
        for (uint64_t r = rng->row; r < row; r++) T8(rng->o, rng->h);
    } else {
        uint64_t g = row_group(rng, row);
        uint32_t j = row_step(rng, row);
        for (unsigned l = 0; l < 8; l++) {
            uint32_t ol[4], hl[4];
            tandem_F_keyed(rng->key, 8u * g + l, DOMAIN_STREAM, AUX_STREAM, ol, hl);
            for (unsigned w = 0; w < 4; w++) {
                rng->o[w][l] = ol[w];
                rng->h[w][l] = hl[w];
            }
        }
        for (uint32_t s = 0; s <= j; s++) T8(rng->o, rng->h);
    }
    rng->row = row;
    rng->cached = 1u;
}

static inline uint64_t align_pos(uint64_t pos, unsigned w) {
    return (pos + w - 1u) & ~((uint64_t)w - 1u);
}

static inline uint32_t word_at(const tandem_rng *rng, uint64_t p) {
    return rng->o[(p >> 5) & 3u][(p >> 7) & 7u];
}

/* Read w bits (8 <= w <= 64, a power of two) at an aligned position. */
static uint64_t read_bits(tandem_rng *rng, uint64_t p, unsigned w) {
    uint64_t raw;
    load_row(rng, p >> 10);
    if (w <= 32u) {
        raw = word_at(rng, p) >> (p & 31u);
        if (w < 32u) raw &= (1u << w) - 1u;
    } else {
        raw = word_at(rng, p) | ((uint64_t)word_at(rng, p + 32u) << 32);
    }
    return raw;
}

static bool read_bit(tandem_rng *rng, uint64_t p) {
    load_row(rng, p >> 10);
    return (word_at(rng, p) >> (p & 31u)) & 1u;
}

static tandem_u128 read_u128(tandem_rng *rng, uint64_t p) {
    tandem_u128 v;
    load_row(rng, p >> 10);
    v.lo = word_at(rng, p) | ((uint64_t)word_at(rng, p + 32u) << 32);
    v.hi = word_at(rng, p + 64u) | ((uint64_t)word_at(rng, p + 96u) << 32);
    return v;
}

static inline double to_f64(uint64_t raw) { return (double)(raw >> 11) * 0x1p-53; }
static inline float to_f32(uint32_t raw) { return (float)(raw >> 8) * 0x1p-24f; }

/* (raw >> 5) * 2^-11 as binary16 bits. Every such value is zero or a normal half with the
 * 11-bit integer k = raw >> 5 as its significand, so the encoding is exact. */
static uint16_t to_f16_bits(uint16_t raw) {
    unsigned k = raw >> 5, m = 0;
    if (k == 0) return 0;
    while ((k >> m) > 1u) m++;
    return (uint16_t)(((m + 4u) << 10) | ((k << (10u - m)) & 0x3ffu));
}

/* floor(raw * 1112064 / 2^64) by the full product, then skip the surrogate range.
 * raw * m = a * m * 2^32 + b * m with both partial products below 2^53, so the high
 * 64 bits of the product are (a * m + (b * m >> 32)) >> 32. */
static uint32_t to_char(uint64_t raw) {
    uint64_t a = raw >> 32, b = raw & 0xffffffffu, m = 1112064u;
    uint64_t hi = (a * m + ((b * m) >> 32)) >> 32;
    return (uint32_t)(hi < 0xd800u ? hi : hi + 0x800u);
}

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

tandem_u128 tandem_next_u128(tandem_rng *rng) {
    uint64_t p = align_pos(rng->pos, 128u);
    rng->pos = p + 128u;
    return read_u128(rng, p);
}

float tandem_next_f32(tandem_rng *rng) { return to_f32(tandem_next_u32(rng)); }
double tandem_next_f64(tandem_rng *rng) { return to_f64(tandem_next_u64(rng)); }
uint16_t tandem_next_f16_bits(tandem_rng *rng) { return to_f16_bits(tandem_next_u16(rng)); }
uint32_t tandem_next_char(tandem_rng *rng) { return to_char(tandem_next_u64(rng)); }

void tandem_next_c32(tandem_rng *rng, float out[2]) {
    out[0] = tandem_next_f32(rng);
    out[1] = tandem_next_f32(rng);
}

void tandem_next_c64(tandem_rng *rng, double out[2]) {
    out[0] = tandem_next_f64(rng);
    out[1] = tandem_next_f64(rng);
}

void tandem_fill_bool(tandem_rng *rng, bool *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = tandem_next_bool(rng);
}

/* Copy the cached row out as 32 words in stream order (lane-major). */
static inline void row_words(const tandem_rng *rng, uint32_t row[32]) {
    for (unsigned l = 0; l < 8; l++)
        for (unsigned w = 0; w < 4; w++) row[4u * l + w] = rng->o[w][l];
}

/* Element e of width w (8 <= w <= 64) inside a row buffer. */
static inline uint64_t row_bits(const uint32_t row[32], unsigned e, unsigned w) {
    unsigned bit = e * w;
    if (w <= 32u) {
        uint64_t raw = row[bit >> 5] >> (bit & 31u);
        return w < 32u ? raw & ((1u << w) - 1u) : raw;
    }
    return row[bit >> 5] | ((uint64_t)row[(bit >> 5) + 1u] << 32);
}

/* Fills take whole rows while the position is row-aligned and at least a row of elements
 * remains, and fall back to single reads at the edges. */
#define FILL(name, type, w, convert)                                                               \
    void name(tandem_rng *rng, type *out, size_t n) {                                              \
        uint64_t p = align_pos(rng->pos, w);                                                       \
        size_t i = 0;                                                                              \
        while (i < n) {                                                                            \
            if ((p & 1023u) == 0 && (n - i) * (w) >= 1024u) {                                      \
                uint32_t row[32];                                                                  \
                load_row(rng, p >> 10);                                                            \
                row_words(rng, row);                                                               \
                for (unsigned e = 0; e < 1024u / (w); e++, i++) out[i] = convert(row_bits(row, e, w)); \
                p += 1024u;                                                                        \
            } else {                                                                               \
                out[i++] = convert(read_bits(rng, p, w));                                          \
                p += w;                                                                            \
            }                                                                                      \
        }                                                                                          \
        rng->pos = p;                                                                              \
    }
#define AS_U8(x) ((uint8_t)(x))
#define AS_U16(x) ((uint16_t)(x))
#define AS_U32(x) ((uint32_t)(x))
#define AS_U64(x) (x)
#define AS_F32(x) to_f32((uint32_t)(x))
#define AS_F16(x) to_f16_bits((uint16_t)(x))
FILL(tandem_fill_u8, uint8_t, 8u, AS_U8)
FILL(tandem_fill_u16, uint16_t, 16u, AS_U16)
FILL(tandem_fill_u32, uint32_t, 32u, AS_U32)
FILL(tandem_fill_u64, uint64_t, 64u, AS_U64)
FILL(tandem_fill_f32, float, 32u, AS_F32)
FILL(tandem_fill_f64, double, 64u, to_f64)
FILL(tandem_fill_f16_bits, uint16_t, 16u, AS_F16)
FILL(tandem_fill_char, uint32_t, 64u, to_char)
#undef FILL

void tandem_fill_u128(tandem_rng *rng, tandem_u128 *out, size_t n) {
    uint64_t p = align_pos(rng->pos, 128u);
    for (size_t i = 0; i < n; i++, p += 128u) out[i] = read_u128(rng, p);
    rng->pos = p;
}

void tandem_fill_c32(tandem_rng *rng, float *out, size_t n) { tandem_fill_f32(rng, out, 2u * n); }
void tandem_fill_c64(tandem_rng *rng, double *out, size_t n) { tandem_fill_f64(rng, out, 2u * n); }

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
