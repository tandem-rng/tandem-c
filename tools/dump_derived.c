/* Maps the derived draws back to uniform bits and writes them to stdout as one packed bit
 * stream, for PractRand and TestU01:
 *   ./dump_derived DRAW SEED [LOG2_BYTES [BITS]] | RNG_test stdin64 ...
 * Without LOG2_BYTES, or with 0, the stream runs until the reader closes it.
 *
 * Each draw x maps to u in [0, 1) by the probability integral transform, which is uniform when
 * x follows its law and independent draws give independent u. The top BITS bits of u enter the
 * stream least significant bit first, so 32 bits give the plain little-endian 32-bit words.
 *   draw             u                                                          default BITS
 *   normal_f64       Phi(x)                                                     52
 *   exponential_f64  1 - exp(-x)                                                32
 *   u64_below        i / n with n = floor(2^64 2/3), exact in integers          52
 *   u32_below        (i + V) / n with n = floor(2^32 2/3)                       32
 *   choice           (C(i) + V p(i)) / (m S), the alias table of w(i) = 1 / (i + 1),
 *                    m = 1000                                                   32
 *   normal_f32       spread over the rounding cell of x by V                    16
 *   exponential_f32  (j + V) 2^-24, j the nearest grid point to 1 - exp(-x)     16
 * A discrete draw needs the uniform V within its atom: otherwise the skewed choice, the bounded
 * index below 2^32 and a Float32 normal, whose rounding cells span about 2^-25 in u, would fail
 * on their resolution alone. V is the Float64 uniform of sub(JITTER) of the generator, an
 * independent stream. Both bounded ranges reject a third of their draws.
 *
 * The default BITS stop above the error of each map. The Float64 exponential is a function of
 * one uniform on a 2^-53 grid, and its error of up to 1.1e-15 moves a tenth of the draws by one
 * 2^-52 step, which biases bit 0 of a 52-bit u. The Float32 draws are functions of uniforms on
 * a 2^-24 grid: at 24 bits PractRand finds that grid within 2^34 bytes even in the controls
 * below. The Float32 exponential is within 0.571 ulp and maps every draw back to its own grid
 * point, as the exact control does.
 *
 * Two controls take the same Float32 uniforms through double-precision libm and round once to
 * float, the law a Float32 draw can at best have on 24-bit uniforms:
 *   normal_f32_exact       Box-Muller from uniforms 2j and 2j + 1, mapped as normal_f32
 *   exponential_f32_exact  -log1p(-u), mapped as exponential_f32
 *
 * Standard error gets bytes_written, first_words (the first 16 64-bit stream words) and
 * final_state (the positions of the generator and of V). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tandem.h"

#define BLOCK 4096u
#define PURPOSE_JITTER UINT64_C(0x4a49545445) /* "JITTE" */
#define CHOICE_M 1000u

typedef struct {
    uint64_t acc, words, limit, seen; /* limit 0: unbounded */
    unsigned nb;
    uint64_t first[16], buf[BLOCK];
    size_t len;
} packer;

static void flush(packer *p) {
    if (p->len && fwrite(p->buf, 8, p->len, stdout) != p->len) exit(3);
    p->len = 0;
}

/* Words past the limit are dropped, so the stream ends exactly at 2^LOG2_BYTES. */
static void emit_word(packer *p, uint64_t w) {
    if (p->limit && p->words * 8 >= p->limit) return;
    if (p->seen < 16) p->first[p->seen++] = w;
    p->buf[p->len++] = w;
    p->words++;
    if (p->len == BLOCK) flush(p);
}

/* Appends the low k < 64 bits of v. */
static void put(packer *p, uint64_t v, unsigned k) {
    if (p->nb + k < 64) {
        p->acc |= v << p->nb;
        p->nb += k;
        return;
    }
    emit_word(p, p->acc | v << p->nb);
    p->acc = v >> (64 - p->nb);
    p->nb = p->nb + k - 64;
}

/* The top k bits of u in [0, 1], with u = 1 kept below 2^k. */
static uint64_t top_bits(double u, unsigned k) {
    double t = floor(ldexp(u, (int)k));
    uint64_t max = (UINT64_C(1) << k) - 1u;
    return t >= (double)max ? max : (uint64_t)t;
}

/* Through the smaller tail, so that u is within half an ulp plus the error of erfc. */
static double phi(double x) {
    const double r = 0.70710678118654752440; /* 1 / sqrt 2; M_SQRT1_2 is not ISO C */
    return x < 0 ? 0.5 * erfc(-x * r) : 1.0 - 0.5 * erfc(x * r);
}

/* u spread by v over the rounding cell of the Float32 normal x. */
static double phi_cell(float x, double v) {
    double lo = phi(0.5 * ((double)x + nextafterf(x, -INFINITY)));
    double hi = phi(0.5 * ((double)x + nextafterf(x, INFINITY)));
    return lo + v * (hi - lo);
}

/* A Float32 exponential is a function of one uniform on the 2^-24 grid, so 1 - exp(-x) lies near
 * a grid point j 2^-24, and a bucket edge of the top bits falls on one: rounding to j and
 * spreading by v over [j, j + 1) 2^-24 keeps the edge from deciding the bucket. */
static double exp_grid(float x, double v) {
    double j = nearbyint(ldexp(-expm1(-(double)x), 24));
    return ldexp((j > 0xffffff ? 0xffffff : j) + v, -24);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: dump_derived DRAW SEED [LOG2_BYTES [BITS]]\n");
        return 2;
    }
    const char *draw = argv[1];
    packer p = {0};
    if (argc >= 4) {
        long e = strtol(argv[3], NULL, 10);
        if (e != 0 && (e < 3 || e > 62)) return 2;
        p.limit = e ? UINT64_C(1) << e : 0;
    }
    unsigned k = 32;
    if (!strcmp(draw, "normal_f64") || !strcmp(draw, "u64_below")) k = 52;
    if (!strncmp(draw, "normal_f32", 10) || !strncmp(draw, "exponential_f32", 15)) k = 16;
    if (argc == 5) {
        long b = strtol(argv[4], NULL, 10);
        if (b < 8 || b > 52) return 2;
        k = (unsigned)b;
    }
    tandem_rng g = tandem_seed(strtoull(argv[2], NULL, 10), 0, 0);
    tandem_rng jit = tandem_sub(&g, PURPOSE_JITTER);
    static double d[BLOCK], u[BLOCK], v[BLOCK];
    static float f[BLOCK];
    static uint32_t i32[BLOCK];
    static uint64_t i64[BLOCK];

    uint64_t cut[CHOICE_M], cum[CHOICE_M + 1], mass[CHOICE_M] = {0};
    uint32_t alias[CHOICE_M];
    tandem_choice_table table;
    {
        double w[CHOICE_M];
        for (unsigned i = 0; i < CHOICE_M; i++) w[i] = 1.0 / (i + 1.0);
        if (!tandem_choice_build(&table, w, CHOICE_M, cut, alias)) return 2;
        /* p(i) from the table itself: column j gives cut[j] to j and the rest to alias[j]. */
        for (unsigned j = 0; j < CHOICE_M; j++) {
            mass[j] += cut[j];
            mass[alias[j]] += table.capacity - cut[j];
        }
        cum[0] = 0;
        for (unsigned i = 0; i < CHOICE_M; i++) cum[i + 1] = cum[i] + mass[i];
    }
    const double total = (double)cum[CHOICE_M];
    const uint32_t n32 = UINT32_C(0xaaaaaaaa);
    const uint64_t n64 = UINT64_C(0xaaaaaaaaaaaaaaaa);

    while (!(p.limit && p.words * 8 >= p.limit)) {
        if (!strcmp(draw, "u64_below")) {
            tandem_fill_u64_below(&g, i64, BLOCK, n64);
            for (unsigned i = 0; i < BLOCK; i++)
                put(&p, (uint64_t)(((unsigned __int128)i64[i] << k) / n64), k);
            continue;
        }
        if (!strcmp(draw, "normal_f64")) {
            tandem_fill_normal_f64(&g, d, BLOCK);
            for (unsigned i = 0; i < BLOCK; i++) u[i] = phi(d[i]);
        } else if (!strcmp(draw, "exponential_f64")) {
            tandem_fill_exponential_f64(&g, d, BLOCK);
            for (unsigned i = 0; i < BLOCK; i++) u[i] = -expm1(-d[i]);
        } else if (!strcmp(draw, "normal_f32")) {
            tandem_fill_normal_f32(&g, f, BLOCK);
            tandem_fill_f64(&jit, v, BLOCK);
            for (unsigned i = 0; i < BLOCK; i++) u[i] = phi_cell(f[i], v[i]);
        } else if (!strcmp(draw, "normal_f32_exact")) {
            tandem_fill_f32(&g, f, BLOCK);
            tandem_fill_f64(&jit, v, BLOCK);
            for (unsigned i = 0; i < BLOCK; i += 2) {
                double r = sqrt(-2.0 * log1p(-(double)f[i])), t = 6.283185307179586 * f[i + 1];
                u[i] = phi_cell((float)(r * cos(t)), v[i]);
                u[i + 1] = phi_cell((float)(r * sin(t)), v[i + 1]);
            }
        } else if (!strcmp(draw, "exponential_f32")) {
            tandem_fill_exponential_f32(&g, f, BLOCK);
            tandem_fill_f64(&jit, v, BLOCK);
            for (unsigned i = 0; i < BLOCK; i++) u[i] = exp_grid(f[i], v[i]);
        } else if (!strcmp(draw, "exponential_f32_exact")) {
            tandem_fill_f32(&g, f, BLOCK);
            tandem_fill_f64(&jit, v, BLOCK);
            for (unsigned i = 0; i < BLOCK; i++)
                u[i] = exp_grid((float)-log1p(-(double)f[i]), v[i]);
        } else if (!strcmp(draw, "u32_below")) {
            tandem_fill_u32_below(&g, i32, BLOCK, n32);
            tandem_fill_f64(&jit, v, BLOCK);
            for (unsigned i = 0; i < BLOCK; i++) u[i] = (i32[i] + v[i]) / n32;
        } else if (!strcmp(draw, "choice")) {
            tandem_fill_choice(&g, i32, BLOCK, &table);
            tandem_fill_f64(&jit, v, BLOCK);
            for (unsigned i = 0; i < BLOCK; i++)
                u[i] = ((double)cum[i32[i]] + v[i] * (double)mass[i32[i]]) / total;
        } else {
            fprintf(stderr, "unknown draw: %s\n", draw);
            return 2;
        }
        for (unsigned i = 0; i < BLOCK; i++) put(&p, top_bits(u[i], k), k);
    }
    flush(&p);
    fprintf(stderr, "bytes_written\t%llu\n", (unsigned long long)(p.words * 8));
    fprintf(stderr, "bits\t%u\n", k);
    fprintf(stderr, "first_words\t");
    for (unsigned i = 0; i < 16; i++)
        fprintf(stderr, "%s%016llx", i ? " " : "", (unsigned long long)p.first[i]);
    fprintf(stderr, "\nfinal_state\tposition=%llu jitter_position=%llu\n",
            (unsigned long long)tandem_position(&g), (unsigned long long)tandem_position(&jit));
    return 0;
}
