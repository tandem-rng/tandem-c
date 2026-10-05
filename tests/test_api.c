/* Functions beyond the specification's draws: positioning, bounded integers, normals,
 * exponentials. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tandem.h"
#include "../tandem_normal_tables.h"
#include "cross_below.h"
#include "cross_choice.h"
#include "cross_exponential.h"
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

/* The f64 fills of the published fixture, from unaligned starts and with a miss of each kind, and
 * the f32 Box-Muller pairs of tandem-cuda's host normalf2, which shares the explicit-fma loop.
 * Every bit must match. The positions pin one draw per f64 element and two per f32 pair. */
static void test_normal_cross(void) {
    tandem_rng g;
    size_t i;
    for (i = 0; i < sizeof CROSS_NORMAL / sizeof CROSS_NORMAL[0]; i++) {
        double got[CROSS_NORMAL_COUNT];
        g = tandem_seed(42, 0, 0);
        tandem_set_position(&g, CROSS_NORMAL[i].start);
        tandem_fill_normal_f64(&g, got, CROSS_NORMAL_COUNT);
        CHECK(memcmp(got, CROSS_NORMAL[i].want, sizeof got) == 0);
        CHECK(tandem_position(&g) == CROSS_NORMAL[i].end_pos);
    }

    g = tandem_seed(42, 0, 0);
    tandem_next_bool(&g);
    for (i = 0; i < CROSS_NORMAL_COUNT; i++) {
        float z[2];
        tandem_normal2_f32(&g, z);
        CHECK(memcmp(z, &CROSS_NORMALF[2 * i], sizeof z) == 0);
    }
    CHECK(tandem_position(&g) == CROSS_NORMALF_END_POS);
}

/* An f64 fill equals the scalar draws, and a fill cut at odd elements, at a missed element and
 * just after it equals the whole fill, with the same end position. The fallback streams are
 * keyed by the global draw index, so a miss resolves the same in any piece. The fill crosses
 * blocks of 512 draws and several flushes of queued misses, from an unaligned start. */
static void test_normal_fills_f64(void) {
    enum { N = 40000 };
    static double want[N], got[N];
    static uint64_t raw[N];
    size_t cuts[8] = {1, 7, 511, 513, 4097}, ncuts = 5, i;
    tandem_rng a = tandem_seed(7, 9, 0), b, d;
    tandem_next_u8(&a);
    b = d = a;
    for (i = 0; i < N; i++) want[i] = tandem_normal_f64(&a);
    tandem_fill_normal_f64(&b, got, N);
    CHECK(memcmp(want, got, sizeof got) == 0 && tandem_position(&b) == tandem_position(&a));

    /* The first miss, then the first after 20000 elements, past several flushes of the queue. */
    b = d;
    tandem_fill_u64(&b, raw, N);
    for (i = 0; i < N && ncuts < 8; i++)
        if ((raw[i] >> 11) >= ZIG_K[raw[i] & (ZIG_LAYERS - 1u)] && (ncuts == 5 || i > 20000)) {
            cuts[ncuts++] = i;
            if (ncuts == 6) cuts[ncuts++] = i + 1;
        }
    CHECK(ncuts == 8);
    for (i = 0; i < ncuts; i++) {
        b = d;
        tandem_fill_normal_f64(&b, got, cuts[i]);
        tandem_fill_normal_f64(&b, got + cuts[i], N - cuts[i]);
        CHECK(memcmp(want, got, sizeof got) == 0 && tandem_position(&b) == tandem_position(&a));
    }
}

/* The scalar f32 normal is the cos half of the pair, and a fill is the flattened pairs, across
 * block boundaries and from an unaligned start. An odd n drops the last sin half but still
 * consumes both uniforms. */
static void test_normal_fills_f32(void) {
    enum { N = 1000 };
    float want32[N], got32[N];
    size_t n, i;

    for (n = 0; n <= N; n += (n < 4 ? 1 : 249)) {
        tandem_rng a = tandem_seed(7, 9, 0), c;
        size_t pairs = n / 2 + n % 2;
        tandem_next_u8(&a);
        c = a;
        for (i = 0; i < pairs; i++) {
            float z[2];
            tandem_normal2_f32(&a, z);
            want32[2 * i] = z[0];
            if (2 * i + 1 < n) want32[2 * i + 1] = z[1];
        }
        tandem_fill_normal_f32(&c, got32, n);
        CHECK(memcmp(want32, got32, n * sizeof *got32) == 0);
        CHECK(tandem_position(&c) == tandem_position(&a));
    }
    {
        tandem_rng a = tandem_seed(3, 4, 0), b = a;
        float zf[2];
        tandem_normal2_f32(&a, zf);
        CHECK(tandem_normal_f32(&b) == zf[0] && tandem_position(&b) == tandem_position(&a));
    }
}

/* Fixtures that tandem-cuda derives on the device: fills from the key of seed 42 at several
 * start positions. Bounded values and f64 normals are exact, f32 normals match to 16 ulps plus
 * an absolute floor. */
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
    /* Unaligned starts, where the fallback index of element e is not e. */
    for (c = 0; c < sizeof CROSS_BELOW32_AT / sizeof CROSS_BELOW32_AT[0]; c++) {
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, CROSS_BELOW32_AT[c].start, 0);
        uint32_t out[64];
        tandem_fill_u32_below(&g, out, 64, CROSS_BELOW32_AT[c].range);
        CHECK(memcmp(out, CROSS_BELOW32_AT[c].out, sizeof out) == 0);
    }
    for (c = 0; c < sizeof CROSS_BELOW64_AT / sizeof CROSS_BELOW64_AT[0]; c++) {
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, CROSS_BELOW64_AT[c].start, 0);
        uint64_t out[64];
        tandem_fill_u64_below(&g, out, 64, CROSS_BELOW64_AT[c].range);
        CHECK(memcmp(out, CROSS_BELOW64_AT[c].out, sizeof out) == 0);
    }
    for (c = 0; c < sizeof CROSS_NORMAL64 / sizeof CROSS_NORMAL64[0]; c++) {
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, CROSS_NORMAL64[c].pos, 0);
        double out[64];
        tandem_fill_normal_f64(&g, out, CROSS_NORMAL64[c].n);
        CHECK(memcmp(out, CROSS_NORMAL64[c].out, CROSS_NORMAL64[c].n * sizeof *out) == 0);
    }
    for (c = 0; c < sizeof CROSS_NORMAL32 / sizeof CROSS_NORMAL32[0]; c++) {
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, CROSS_NORMAL32[c].pos, 0);
        float out[64];
        tandem_fill_normal_f32(&g, out, CROSS_NORMAL32[c].n);
        for (i = 0; i < CROSS_NORMAL32[c].n; i++)
            CHECK(fabsf(out[i] - CROSS_NORMAL32[c].out[i]) <= 16.0f * 0x1p-23f * fabsf(CROSS_NORMAL32[c].out[i]) + 1e-6f);
    }
}

/* An empty bounded, f32 normal or exponential fill advances nothing, even from an unaligned
 * position. An empty f64 normal fill aligns the position to 64 and writes nothing, as Appendix A
 * says. */
static void test_empty_fills(void) {
    uint64_t pos[] = {1, 5, 33, 65, 1001};
    size_t i;
    for (i = 0; i < sizeof pos / sizeof pos[0]; i++) {
        tandem_rng g = tandem_seed(1, 2, 0);
        uint32_t u32;
        uint64_t u64;
        double d = 2;
        float f;
        tandem_set_position(&g, pos[i]);
        tandem_fill_u32_below(&g, &u32, 0, 10);
        tandem_fill_u64_below(&g, &u64, 0, 10);
        tandem_fill_normal_f32(&g, &f, 0);
        tandem_fill_exponential_f64(&g, &d, 0);
        tandem_fill_exponential_f32(&g, &f, 0);
        CHECK(tandem_position(&g) == pos[i]);
        tandem_fill_normal_f64(&g, &d, 0);
        CHECK(tandem_position(&g) == ((pos[i] + 63u) & ~(uint64_t)63u) && d == 2);
    }
    /* The n = 0 cases of the spec's conformance fixtures, one fill at a time from the key of
     * seed 42 at bit 33. */
    {
        tandem_rng g = tandem_seed(42, 0, 0), h;
        uint32_t u32;
        uint64_t u64;
        double d;
        float f;
        tandem_set_position(&g, 33);
        h = g;
        tandem_fill_u32_below(&h, &u32, 0, 10);
        CHECK(tandem_position(&h) == 33);
        h = g;
        tandem_fill_u64_below(&h, &u64, 0, 10);
        CHECK(tandem_position(&h) == 33);
        h = g;
        tandem_fill_normal_f32(&h, &f, 0);
        CHECK(tandem_position(&h) == 33);
        h = g;
        tandem_fill_exponential_f64(&h, &d, 0);
        CHECK(tandem_position(&h) == 33);
        h = g;
        tandem_fill_exponential_f32(&h, &f, 0);
        CHECK(tandem_position(&h) == 33);
        h = g;
        tandem_fill_normal_f64(&h, &d, 0);
        CHECK(tandem_position(&h) == 64);
        {
            const double w[4] = {1, 2, 3, 4};
            uint64_t cut[4];
            uint32_t alias[4];
            tandem_choice_table t;
            CHECK(tandem_choice_build(&t, w, 4, cut, alias));
            h = g;
            tandem_fill_choice(&h, &u32, 0, &t);
            CHECK(tandem_position(&h) == 64);
        }
    }
}

/* Exponentials of tandem-cuda's core.hpp, whose polynomial logarithm is this library's on the
 * host and the device, so every bit must match. Element i of a fill is scalar draw i. */
static void test_exponential_cross(void) {
    size_t c, i;
    for (c = 0; c < sizeof CROSS_EXPONENTIAL / sizeof CROSS_EXPONENTIAL[0]; c++) {
        tandem_rng a = tandem_seed(42, 0, 0), b;
        double got[CROSS_EXPONENTIAL_COUNT];
        tandem_set_position(&a, CROSS_EXPONENTIAL[c].start);
        b = a;
        tandem_fill_exponential_f64(&a, got, CROSS_EXPONENTIAL_COUNT);
        CHECK(memcmp(got, CROSS_EXPONENTIAL[c].want, sizeof got) == 0);
        CHECK(tandem_position(&a) == CROSS_EXPONENTIAL[c].end_pos);
        for (i = 0; i < CROSS_EXPONENTIAL_COUNT; i++) got[i] = tandem_exponential_f64(&b);
        CHECK(memcmp(got, CROSS_EXPONENTIAL[c].want, sizeof got) == 0);
        CHECK(tandem_position(&b) == CROSS_EXPONENTIAL[c].end_pos);
    }
    for (c = 0; c < sizeof CROSS_EXPONENTIALF / sizeof CROSS_EXPONENTIALF[0]; c++) {
        tandem_rng a = tandem_seed(42, 0, 0), b;
        float got[CROSS_EXPONENTIAL_COUNT];
        tandem_set_position(&a, CROSS_EXPONENTIALF[c].start);
        b = a;
        tandem_fill_exponential_f32(&a, got, CROSS_EXPONENTIAL_COUNT);
        CHECK(memcmp(got, CROSS_EXPONENTIALF[c].want, sizeof got) == 0);
        CHECK(tandem_position(&a) == CROSS_EXPONENTIALF[c].end_pos);
        for (i = 0; i < CROSS_EXPONENTIAL_COUNT; i++) got[i] = tandem_exponential_f32(&b);
        CHECK(memcmp(got, CROSS_EXPONENTIALF[c].want, sizeof got) == 0);
        CHECK(tandem_position(&b) == CROSS_EXPONENTIALF[c].end_pos);
    }
}

/* A fill equals the scalar draws and a fill cut into pieces at any element, across the block
 * boundaries of the fill and from an unaligned start, with the same end position. */
static void test_exponential_fills(void) {
    enum { N = 3000 };
    static double want[N], got[N];
    static float want32[N], got32[N];
    size_t ns[] = {1, 2, 3, 1023, 1024, 1025, N}, cuts[] = {1, 7, 1000, 1024, 2049};
    size_t k, c, i;
    for (k = 0; k < sizeof ns / sizeof ns[0]; k++) {
        size_t n = ns[k];
        tandem_rng a = tandem_seed(7, 9, 0), b, d;
        tandem_next_u8(&a);
        b = d = a;
        for (i = 0; i < n; i++) want[i] = tandem_exponential_f64(&a);
        tandem_fill_exponential_f64(&b, got, n);
        CHECK(memcmp(want, got, n * sizeof *got) == 0);
        CHECK(tandem_position(&b) == tandem_position(&a));
        for (c = 0; c < sizeof cuts / sizeof cuts[0]; c++) {
            size_t cut = cuts[c] < n ? cuts[c] : n;
            tandem_rng e = d;
            tandem_fill_exponential_f64(&e, got, cut);
            tandem_fill_exponential_f64(&e, got + cut, n - cut);
            CHECK(memcmp(want, got, n * sizeof *got) == 0);
            CHECK(tandem_position(&e) == tandem_position(&a));
        }

        a = b = d;
        for (i = 0; i < n; i++) want32[i] = tandem_exponential_f32(&a);
        tandem_fill_exponential_f32(&b, got32, n);
        CHECK(memcmp(want32, got32, n * sizeof *got32) == 0);
        CHECK(tandem_position(&b) == tandem_position(&a));
        for (c = 0; c < sizeof cuts / sizeof cuts[0]; c++) {
            size_t cut = cuts[c] < n ? cuts[c] : n;
            tandem_rng e = d;
            tandem_fill_exponential_f32(&e, got32, cut);
            tandem_fill_exponential_f32(&e, got32 + cut, n - cut);
            CHECK(memcmp(want32, got32, n * sizeof *got32) == 0);
            CHECK(tandem_position(&e) == tandem_position(&a));
        }
    }
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Exp(1) has raw moments E[x^k] = k! and Var(x^k) = (2k)! - (k!)^2. Each sample moment must
 * lie within 5 standard errors, and the Kolmogorov-Smirnov statistic sqrt(n) D below 1.95,
 * the 0.1 % point of the Kolmogorov distribution. */
static void check_exponential_sample(const char *name, double *x, size_t n) {
    const double fact[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};
    double m[5] = {0}, d = 0, p = 0;
    size_t i;
    int k;
    for (i = 0; i < n; i++) {
        double xk = 1;
        for (k = 1; k <= 4; k++) m[k] += xk *= x[i];
    }
    for (k = 1; k <= 4; k++) {
        double se = sqrt((fact[2 * k] - fact[k] * fact[k]) / (double)n);
        m[k] /= (double)n;
        CHECK(fabs(m[k] - fact[k]) < 5 * se);
    }
    qsort(x, n, sizeof *x, cmp_double);
    for (i = 0; i < n; i++) {
        double f = -expm1(-x[i]), lo = f - (double)i / (double)n, hi = (double)(i + 1) / (double)n - f;
        d = fmax(d, fmax(lo, hi));
    }
    d *= sqrt((double)n);
    CHECK(d < 1.95);
    /* The asymptotic p-value, 2 sum (-1)^(k-1) exp(-2 k^2 d^2). */
    for (k = 1; k < 100; k++) p += (k % 2 ? 2.0 : -2.0) * exp(-2.0 * k * k * d * d);
    printf("%s: moments %.5f %.5f %.4f %.3f, KS sqrt(n) D %.3f, p %.3f\n", name, m[1], m[2],
           m[3], m[4], d, p);
}

static void test_exponential_stats(void) {
    enum { N = 10000000 };
    double *x = malloc(N * sizeof *x);
    float *f = malloc(N * sizeof *f);
    size_t i;
    tandem_rng g = tandem_seed(2026, 10, 0);
    tandem_fill_exponential_f64(&g, x, N);
    check_exponential_sample("exponential f64", x, N);
    tandem_fill_exponential_f32(&g, f, N);
    for (i = 0; i < N; i++) x[i] = f[i];
    check_exponential_sample("exponential f32", x, N);
    free(x);
    free(f);
}

/* Weighted choice fills of the published fixture, from aligned and unaligned starts, and the
 * scalar draws, which must equal them. The positions pin one 64-bit draw per element. */
static void test_choice_cross(void) {
    size_t c, i;
    for (c = 0; c < sizeof CROSS_CHOICE / sizeof CROSS_CHOICE[0]; c++) {
        size_t m = CROSS_CHOICE[c].m;
        uint64_t *cut = malloc(m * sizeof *cut);
        uint32_t *alias = malloc(m * sizeof *alias), got[CROSS_CHOICE_COUNT];
        tandem_choice_table t;
        tandem_rng a = tandem_seed(42, 0, 0), b;
        CHECK(tandem_choice_build(&t, CROSS_CHOICE[c].weights, m, cut, alias));
        CHECK(t.capacity == CROSS_CHOICE[c].capacity);
        tandem_set_position(&a, CROSS_CHOICE[c].start);
        b = a;
        tandem_fill_choice(&a, got, CROSS_CHOICE_COUNT, &t);
        CHECK(memcmp(got, CROSS_CHOICE[c].want, sizeof got) == 0);
        CHECK(tandem_position(&a) == CROSS_CHOICE[c].end_pos);
        for (i = 0; i < CROSS_CHOICE_COUNT; i++) CHECK(tandem_choice(&b, &t) == CROSS_CHOICE[c].want[i]);
        CHECK(tandem_position(&b) == CROSS_CHOICE[c].end_pos);
        free(cut);
        free(alias);
    }
}

/* A fill cut at any element equals the whole fill, across the fill's blocks of draws and from an
 * unaligned start. An empty fill aligns to 64. Invalid weights build no table. */
static void test_choice_fills(void) {
    enum { N = 2000 };
    static uint32_t want[N], got[N];
    const double w[] = {3, 0, 1, 7.5, 0.125};
    const double nan = NAN, bad[][2] = {{1, -1}, {1, nan}, {1, INFINITY}, {0, 0}};
    size_t cuts[] = {1, 511, 512, 513, 1999}, c;
    uint64_t cut[5];
    uint32_t alias[5];
    tandem_choice_table t;
    tandem_rng a = tandem_seed(7, 9, 0), b;
    CHECK(tandem_choice_build(&t, w, 5, cut, alias));
    tandem_next_u8(&a);
    b = a;
    tandem_fill_choice(&a, want, N, &t);
    for (c = 0; c < sizeof cuts / sizeof cuts[0]; c++) {
        tandem_rng e = b;
        tandem_fill_choice(&e, got, cuts[c], &t);
        tandem_fill_choice(&e, got + cuts[c], N - cuts[c], &t);
        CHECK(memcmp(want, got, sizeof got) == 0 && tandem_position(&e) == tandem_position(&a));
    }
    tandem_fill_choice(&b, got, 0, &t);
    CHECK(tandem_position(&b) == 64);

    CHECK(!tandem_choice_build(&t, w, 0, cut, alias));
    for (c = 0; c < sizeof bad / sizeof bad[0]; c++) CHECK(!tandem_choice_build(&t, bad[c], 2, cut, alias));
}

/* 10^7 draws against the weights by Pearson's chi-square over the positive weights, 8 degrees of
 * freedom, below 31.83, the 0.01 % point. The zero weight is never drawn. */
static void test_choice_stats(void) {
    enum { N = 10000000, M = 10 };
    const double w[M] = {5, 0, 1, 2, 3, 0.5, 8, 13, 0.25, 21};
    uint32_t *x = malloc(N * sizeof *x), alias[M];
    uint64_t cut[M], count[M] = {0};
    double total = 0, chi2 = 0;
    tandem_choice_table t;
    tandem_rng g = tandem_seed(2026, 11, 0);
    size_t i;
    CHECK(tandem_choice_build(&t, w, M, cut, alias));
    tandem_fill_choice(&g, x, N, &t);
    for (i = 0; i < N; i++) count[x[i]]++;
    for (i = 0; i < M; i++) total += w[i];
    for (i = 0; i < M; i++) {
        double expect = (double)N * w[i] / total, d = (double)count[i] - expect;
        if (w[i] > 0) chi2 += d * d / expect;
    }
    CHECK(count[1] == 0);
    CHECK(chi2 < 31.83);
    printf("choice: chi-square %.2f on 8 degrees of freedom\n", chi2);
    free(x);
}

int main(void) {
    test_set_position();
    test_below_cross();
    test_fill_below();
    test_fill_below_cut();
    test_below_degenerate();
    test_normal_cross();
    test_normal_fills_f64();
    test_normal_fills_f32();
    test_empty_fills();
    test_exponential_cross();
    test_exponential_fills();
    test_exponential_stats();
    test_device_fixtures();
    test_choice_cross();
    test_choice_fills();
    test_choice_stats();
    if (failures) {
        printf("%d failures\n", failures);
        return 1;
    }
    puts("api: ok");
    return 0;
}
