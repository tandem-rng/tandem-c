/* Functions beyond the specification's draws: positioning, bounded integers, normals. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tandem.h"
#include "cross_below.h"
#include "cross_fill_below.h"
#include "cross_normal.h"
#include "cuda_fill_below.h"
/* The f32 literals of the copied header carry no f suffix. */
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif
#include "cuda_fill_normal.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

static int failures;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                \
        }                                                                                          \
    } while (0)

/* A rewound or jumped generator draws what a fresh one at that position draws, whatever the
 * cache held, and a rejected position changes nothing. */
static void test_set_position(void) {
    uint32_t key[4] = {1, 2, 3, 4};
    tandem_rng g = tandem_from_key(key, 0, 0), fresh;
    uint64_t pos[] = {0, 64, 1000, 1024 * 33 + 64, 4096, 17};
    size_t i;

    for (i = 0; i < 5000; i++) tandem_next_u64(&g);
    for (i = 0; i < sizeof pos / sizeof pos[0]; i++) {
        fresh = tandem_from_key(key, pos[i], 0);
        CHECK(tandem_set_position(&g, pos[i]));
        CHECK(tandem_position(&g) == pos[i]);
        CHECK(tandem_next_u64(&g) == tandem_next_u64(&fresh));
        CHECK(tandem_next_u32(&g) == tandem_next_u32(&fresh));
    }

    {
        tandem_rng before = g;
        CHECK(tandem_set_position(&g, ((uint64_t)1 << 63) - 1u));
        CHECK(tandem_set_position(&g, 0));
        g = before;
        CHECK(!tandem_set_position(&g, (uint64_t)1 << 63));
        CHECK(!tandem_set_position(&g, ~(uint64_t)0));
        CHECK(memcmp(&g, &before, sizeof g) == 0);
    }
}

/* Values and stream position of tandem-cuda's urand(range) and urand64(range). The position
 * pins the number of rejected draws as well as the values. */
static void test_below_cross(void) {
    size_t c, i;
    for (c = 0; c < sizeof CROSS_U32 / sizeof CROSS_U32[0]; c++) {
        tandem_rng g = tandem_seed(42, 0, 0);
        tandem_next_bool(&g);
        for (i = 0; i < CROSS_COUNT; i++) CHECK(tandem_u32_below(&g, CROSS_U32[c].n) == CROSS_U32[c].want[i]);
        CHECK(tandem_position(&g) == CROSS_U32[c].end_pos);
    }
    for (c = 0; c < sizeof CROSS_U64 / sizeof CROSS_U64[0]; c++) {
        tandem_rng g = tandem_seed(42, 0, 0);
        tandem_next_bool(&g);
        for (i = 0; i < CROSS_COUNT; i++) CHECK(tandem_u64_below(&g, CROSS_U64[c].n) == CROSS_U64[c].want[i]);
        CHECK(tandem_position(&g) == CROSS_U64[c].end_pos);
    }
}

/* The parallel fills of core.hpp, rejected draws included. The fills of 64 elements with the
 * large ranges reject often, so the fallback generator is exercised. A fill without a
 * rejection equals the scalar calls, which pins the plain path. */
static void test_fill_below(void) {
    size_t c, i, same = 0, total = 0;
    for (c = 0; c < sizeof CROSS_FILL_U32 / sizeof CROSS_FILL_U32[0]; c++) {
        tandem_rng g = tandem_seed(42, 0, 0), s;
        uint32_t out[CROSS_COUNT];
        tandem_set_position(&g, CROSS_FILL_U32[c].start);
        s = g;
        tandem_fill_u32_below(&g, out, CROSS_COUNT, CROSS_FILL_U32[c].n);
        CHECK(memcmp(out, CROSS_FILL_U32[c].want, sizeof out) == 0);
        CHECK(tandem_position(&g) == CROSS_FILL_U32[c].end_pos);
        for (i = 0; i < CROSS_COUNT; i++) same += out[i] == tandem_u32_below(&s, CROSS_FILL_U32[c].n), total++;
    }
    CHECK(same < total);
    for (c = 0; c < sizeof CROSS_FILL_U64 / sizeof CROSS_FILL_U64[0]; c++) {
        tandem_rng g = tandem_seed(42, 0, 0);
        uint64_t out[CROSS_COUNT];
        tandem_set_position(&g, CROSS_FILL_U64[c].start);
        tandem_fill_u64_below(&g, out, CROSS_COUNT, CROSS_FILL_U64[c].n);
        CHECK(memcmp(out, CROSS_FILL_U64[c].want, sizeof out) == 0);
        CHECK(tandem_position(&g) == CROSS_FILL_U64[c].end_pos);
    }
}

/* A fill cut at any element boundary equals the whole fill, rejected draws included, because
 * the fallback is keyed by the global draw index. Starts at nonzero, unaligned positions and
 * ranges with many rejections. */
static void test_fill_below_cut(void) {
    enum { N = 300 };
    uint64_t starts[] = {1, 12345, 100000};
    size_t cuts[] = {1, 7, 100, 299}, s, c, i;
    for (s = 0; s < sizeof starts / sizeof starts[0]; s++)
        for (c = 0; c < sizeof cuts / sizeof cuts[0]; c++) {
            tandem_rng whole = tandem_seed(5, 6, 0), part = whole;
            uint32_t w32[N], p32[N];
            uint64_t w64[N], p64[N];
            size_t cut = cuts[c];
            tandem_set_position(&whole, starts[s]);
            part = whole;
            tandem_fill_u32_below(&whole, w32, N, 0xc0000001u);
            tandem_fill_u32_below(&part, p32, cut, 0xc0000001u);
            tandem_fill_u32_below(&part, p32 + cut, N - cut, 0xc0000001u);
            CHECK(memcmp(w32, p32, sizeof w32) == 0 && tandem_position(&whole) == tandem_position(&part));
            tandem_set_position(&whole, starts[s]);
            part = whole;
            tandem_fill_u64_below(&whole, w64, N, 0xc000000000000001ull);
            tandem_fill_u64_below(&part, p64, cut, 0xc000000000000001ull);
            tandem_fill_u64_below(&part, p64 + cut, N - cut, 0xc000000000000001ull);
            CHECK(memcmp(w64, p64, sizeof w64) == 0 && tandem_position(&whole) == tandem_position(&part));
        }
    /* The same fill from the same key differs between starts that share elements: no shared fallback. */
    {
        tandem_rng a = tandem_seed(5, 6, 0), b = a;
        uint32_t x[N], y[N];
        tandem_set_position(&a, 0);
        tandem_set_position(&b, 32);
        tandem_fill_u32_below(&a, x, N, 0xc0000001u);
        tandem_fill_u32_below(&b, y, N - 1, 0xc0000001u);
        for (i = 0; i + 1 < N; i++) CHECK(x[i + 1] == y[i]);
    }
}

/* n = 1 and n = 0 return 0 and consume one draw, as core.hpp does. */
static void test_below_degenerate(void) {
    tandem_rng g = tandem_seed(1, 2, 0);
    CHECK(tandem_u32_below(&g, 0) == 0 && tandem_position(&g) == 32);
    CHECK(tandem_u64_below(&g, 0) == 0 && tandem_position(&g) == 128);
}

/* Box-Muller pairs of tandem-cuda's normal2 and normalf2. log, cos and sin may differ in the
 * last place between libms: 1e-12 relative for f64, 16 ulps plus an absolute floor near the zeros
 * of cos and sin for f32. The position pins two uniforms per pair. */
static void test_normal_cross(void) {
    tandem_rng g = tandem_seed(42, 0, 0);
    size_t i;
    tandem_next_bool(&g);
    for (i = 0; i < CROSS_NORMAL_COUNT; i++) {
        double z[2];
        tandem_normal2_f64(&g, z);
        CHECK(fabs(z[0] - CROSS_NORMAL[2 * i]) <= 1e-12 * fabs(CROSS_NORMAL[2 * i]));
        CHECK(fabs(z[1] - CROSS_NORMAL[2 * i + 1]) <= 1e-12 * fabs(CROSS_NORMAL[2 * i + 1]));
    }
    CHECK(tandem_position(&g) == CROSS_NORMAL_END_POS);

    g = tandem_seed(42, 0, 0);
    tandem_next_bool(&g);
    for (i = 0; i < CROSS_NORMAL_COUNT; i++) {
        float z[2];
        tandem_normal2_f32(&g, z);
        CHECK(fabsf(z[0] - CROSS_NORMALF[2 * i]) <= 16.0f * 0x1p-23f * fabsf(CROSS_NORMALF[2 * i]) + 1e-6f);
        CHECK(fabsf(z[1] - CROSS_NORMALF[2 * i + 1]) <= 16.0f * 0x1p-23f * fabsf(CROSS_NORMALF[2 * i + 1]) + 1e-6f);
    }
    CHECK(tandem_position(&g) == CROSS_NORMALF_END_POS);
}

/* The scalar normal is the cos half of the pair, and a fill is the flattened pairs, across
 * block boundaries and from an unaligned start. An odd n drops the last sin half but still
 * consumes both uniforms. */
static void test_normal_fills(void) {
    enum { N = 1000 };
    double want[N], got[N];
    float want32[N], got32[N];
    size_t n, i;

    for (n = 0; n <= N; n += (n < 4 ? 1 : 249)) {
        tandem_rng a = tandem_seed(7, 9, 0), b, c, d;
        size_t pairs = n / 2 + n % 2;
        tandem_next_u8(&a);
        b = c = d = a;
        for (i = 0; i < pairs; i++) {
            double z[2];
            tandem_normal2_f64(&a, z);
            want[2 * i] = z[0];
            if (2 * i + 1 < n) want[2 * i + 1] = z[1];
        }
        tandem_fill_normal_f64(&b, got, n);
        CHECK(memcmp(want, got, n * sizeof *got) == 0);
        CHECK(tandem_position(&b) == tandem_position(&a));

        a = c;
        for (i = 0; i < pairs; i++) {
            float z[2];
            tandem_normal2_f32(&a, z);
            want32[2 * i] = z[0];
            if (2 * i + 1 < n) want32[2 * i + 1] = z[1];
        }
        tandem_fill_normal_f32(&c, got32, n);
        CHECK(memcmp(want32, got32, n * sizeof *got32) == 0);
        CHECK(tandem_position(&c) == tandem_position(&a));

        if (n) {
            double z[2];
            tandem_normal2_f64(&d, z);
            CHECK(want[0] == z[0]);
        }
    }
    {
        tandem_rng a = tandem_seed(3, 4, 0), b = a;
        double z[2];
        float zf[2];
        tandem_normal2_f64(&a, z);
        CHECK(tandem_normal_f64(&b) == z[0] && tandem_position(&b) == tandem_position(&a));
        a = b = tandem_seed(3, 4, 0);
        tandem_normal2_f32(&a, zf);
        CHECK(tandem_normal_f32(&b) == zf[0] && tandem_position(&b) == tandem_position(&a));
    }
}

/* Fixtures that tandem-cuda derives on the device: fills from the key of seed 42 at several
 * start positions. Bounded values are exact, f64 normals match to 1e-12 relative and f32
 * normals to 16 ulps plus an absolute floor. */
static void test_device_fixtures(void) {
    size_t c, i;
    for (c = 0; c < sizeof CROSS_BELOW32 / sizeof CROSS_BELOW32[0]; c++) {
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, 0, 0);
        uint32_t out[64];
        tandem_fill_u32_below(&g, out, 64, CROSS_BELOW32[c].range);
        CHECK(memcmp(out, CROSS_BELOW32[c].out, sizeof out) == 0);
    }
    for (c = 0; c < sizeof CROSS_BELOW64 / sizeof CROSS_BELOW64[0]; c++) {
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, 0, 0);
        uint64_t out[64];
        tandem_fill_u64_below(&g, out, 64, CROSS_BELOW64[c].range);
        CHECK(memcmp(out, CROSS_BELOW64[c].out, sizeof out) == 0);
    }
    for (c = 0; c < sizeof CROSS_NORMAL64 / sizeof CROSS_NORMAL64[0]; c++) {
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, CROSS_NORMAL64[c].pos, 0);
        double out[64];
        tandem_fill_normal_f64(&g, out, CROSS_NORMAL64[c].n);
        for (i = 0; i < CROSS_NORMAL64[c].n; i++)
            CHECK(fabs(out[i] - CROSS_NORMAL64[c].out[i]) <= 1e-12 * fabs(CROSS_NORMAL64[c].out[i]));
    }
    for (c = 0; c < sizeof CROSS_NORMAL32 / sizeof CROSS_NORMAL32[0]; c++) {
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, CROSS_NORMAL32[c].pos, 0);
        float out[64];
        tandem_fill_normal_f32(&g, out, CROSS_NORMAL32[c].n);
        for (i = 0; i < CROSS_NORMAL32[c].n; i++)
            CHECK(fabsf(out[i] - CROSS_NORMAL32[c].out[i]) <= 16.0f * 0x1p-23f * fabsf(CROSS_NORMAL32[c].out[i]) + 1e-6f);
    }
}

/* An empty bounded or normal fill advances nothing, even from an unaligned position. */
static void test_empty_fills(void) {
    uint64_t pos[] = {1, 5, 33, 65, 1001};
    size_t i;
    for (i = 0; i < sizeof pos / sizeof pos[0]; i++) {
        tandem_rng g = tandem_seed(1, 2, 0);
        uint32_t u32;
        uint64_t u64;
        double d;
        float f;
        tandem_set_position(&g, pos[i]);
        tandem_fill_u32_below(&g, &u32, 0, 10);
        tandem_fill_u64_below(&g, &u64, 0, 10);
        tandem_fill_normal_f64(&g, &d, 0);
        tandem_fill_normal_f32(&g, &f, 0);
        CHECK(tandem_position(&g) == pos[i]);
    }
}

int main(void) {
    test_set_position();
    test_below_cross();
    test_fill_below();
    test_fill_below_cut();
    test_below_degenerate();
    test_normal_cross();
    test_normal_fills();
    test_empty_fills();
    test_device_fixtures();
    if (failures) {
        printf("%d failures\n", failures);
        return 1;
    }
    puts("api: ok");
    return 0;
}
