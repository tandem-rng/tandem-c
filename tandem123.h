/* Tandem8x32 in the shape of the Random123 interface: a counter-based generator, a pure function
 * of a 128-bit counter and a key, usable from C, C++, CUDA and HIP device code.
 *
 * Reference C implementation of https://github.com/tandem-rng/spec.
 * Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 *
 *     tandem4x32_ctr_t ctr = {{block, 0, 0, 0}};
 *     tandem4x32_key_t key = tandem4x32_key_from_seed(42, 0);
 *     tandem4x32_ctr_t r = tandem4x32(ctr, key);   // four 32-bit words
 *
 * Differences from Random123's philox4x32:
 *   - The key has four words, not two, because Tandem's key is 128 bits. tandem4x32_key_t is
 *     {uint32_t v[4]}, where philox4x32_key_t is {uint32_t v[2]}.
 *   - The counter is not a free 128-bit space. It names one 128-bit block of the stream and the
 *     chunk length of that stream, below.
 *   - There is no round-count variant. Tandem has one fixed step.
 *
 * Counter mapping. With ctr.v[0..3] and the key words k:
 *   block = ctr.v[0] + 2^32 ctr.v[1]   the 128-bit block, that is bits [128 block, 128 block + 128)
 *                                      of the stream, so stream position 128 block;
 *   K     = ctr.v[2]                   the chunk length, a power of two in [1, 65536], with 0
 *                                      meaning the default 32;
 *   ctr.v[3] is reserved and ignored.
 * The result is exactly the specification's block at that position: the four little-endian
 * 32-bit words that tandem_fill_u32 writes at elements 4 block .. 4 block + 3 of a generator
 * with key k, chunk length K, and position 0. In the notation of the specification the stream
 * row is r = block >> 3, the lane is block & 7, the chunk is 8 (r div K) + lane, the step is
 * r mod K, and the result is B(chunk, step). The specification limits stream positions to
 * 2^63 bits, so block < 2^56, which means ctr.v[1] < 2^24. Larger counters are defined by the
 * same formulas but are outside the specification.
 *
 * The functions are static inline and need no linked library, so a kernel can include this
 * file alone. TANDEM_CUDA_DEVICE marks them __host__ __device__ for nvcc and hipcc. */
#ifndef TANDEM123_H
#define TANDEM123_H

#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TANDEM_CUDA_DEVICE __host__ __device__
#else
#define TANDEM_CUDA_DEVICE
#endif

typedef struct {
    uint32_t v[4];
} tandem4x32_ctr_t;

typedef struct {
    uint32_t v[4];
} tandem4x32_key_t;

/* The step T and the seeding function F of the specification on the exposed half o and the
 * hidden half h. The names carry a prefix so that the header can sit next to tandem.h. */
TANDEM_CUDA_DEVICE static inline uint32_t tandem123_rotl(uint32_t x, unsigned r) {
    return (x << r) | (x >> (32u - r));
}

TANDEM_CUDA_DEVICE static inline void tandem123_T(uint32_t o[4], uint32_t h[4]) {
    uint64_t p0 = (uint64_t)o[0] * (h[0] | 1u);
    uint64_t p1 = (uint64_t)o[2] * (h[1] | 1u);
    uint32_t lo0 = (uint32_t)p0, hi0 = (uint32_t)(p0 >> 32);
    uint32_t lo1 = (uint32_t)p1, hi1 = (uint32_t)(p1 >> 32);
    uint32_t n0 = o[1] ^ hi1 ^ lo1;
    uint32_t n1 = tandem123_rotl(lo1, 16) ^ h[2];
    uint32_t n2 = o[3] ^ hi0 ^ lo0;
    uint32_t n3 = tandem123_rotl(lo0, 16) ^ h[3];

    h[0] ^= tandem123_rotl(h[1], 7);
    h[1] ^= tandem123_rotl(h[2], 13);
    h[2] ^= tandem123_rotl(h[3], 22);
    h[3] ^= tandem123_rotl(h[0], 3);
    h[0] = (h[0] + 0x9e3779b9u) ^ n0;

    o[0] = n0;
    o[1] = n1;
    o[2] = n2;
    o[3] = n3;
}

TANDEM_CUDA_DEVICE static inline void tandem123_F(uint32_t o[4], uint32_t h[4]) {
    const uint32_t rc[8] = {0xd17cc1b7u, 0xa7220a94u, 0xfe13abe8u, 0xfa9a6ee0u,
                            0xedb14accu, 0x9e21c820u, 0xff28b1d5u, 0xef5de2b0u};
    for (int r = 0; r < 8; r++) {
        uint32_t t[4];
        tandem123_T(o, h);
        o[0] ^= rc[r];
        for (int w = 0; w < 4; w++) {
            t[w] = o[w];
            o[w] = h[w];
            h[w] = t[w];
        }
    }
}

/* The chunk state after F: the exposed half o, hidden half h, of chunk c under the key. */
TANDEM_CUDA_DEVICE static inline void tandem123_chunk(tandem4x32_key_t key, uint64_t c,
                                                      uint32_t o[4], uint32_t h[4]) {
    o[0] = (uint32_t)c;
    o[1] = (uint32_t)(c >> 32);
    o[2] = 0x9e3779b9u; /* the stream domain */
    o[3] = 0x94d049bbu; /* and its auxiliary word */
    for (int w = 0; w < 4; w++) h[w] = key.v[w];
    tandem123_F(o, h);
}

/* The 128-bit key of a stream seeded with a 128-bit seed as two halves, as tandem_seed. */
TANDEM_CUDA_DEVICE static inline tandem4x32_key_t tandem4x32_key_from_seed(uint64_t seed_lo,
                                                                         uint64_t seed_hi) {
    uint32_t o[4] = {0, 0, 0xa54ff53au, 0};
    uint32_t h[4] = {(uint32_t)seed_lo, (uint32_t)(seed_lo >> 32), (uint32_t)seed_hi,
                     (uint32_t)(seed_hi >> 32)};
    tandem4x32_key_t k;
    tandem123_F(o, h);
    for (int w = 0; w < 4; w++) k.v[w] = o[w];
    return k;
}

/* The block of the counter, as described at the top of this file. */
TANDEM_CUDA_DEVICE static inline tandem4x32_ctr_t tandem4x32(tandem4x32_ctr_t ctr,
                                                             tandem4x32_key_t key) {
    uint64_t block = (uint64_t)ctr.v[0] | ((uint64_t)ctr.v[1] << 32);
    uint32_t K = ctr.v[2] ? ctr.v[2] : 32u;
    uint32_t log2k = 0, step, h[4], o[4];
    uint64_t row = block >> 3;
    tandem4x32_ctr_t out;

    while ((K >> log2k) > 1u) log2k++;
    step = (uint32_t)(row & (K - 1u));
    tandem123_chunk(key, 8u * (row >> log2k) + (block & 7u), o, h);
    for (uint32_t s = 0; s <= step; s++) tandem123_T(o, h);
    for (int w = 0; w < 4; w++) out.v[w] = o[w];
    return out;
}

#endif /* TANDEM123_H */
