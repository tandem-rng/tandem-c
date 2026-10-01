/* Long-stream agreement with TandemRNG.jl. The dumps in tests/data are raw little-endian
 * fills produced by the Julia reference (tools/dump_streams.jl). Each is compared against a
 * fill, against scalar draws, and at sampled indices against random access. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tandem.h"

static int failures;

static void *slurp(const char *dir, const char *name, size_t *len) {
    char path[512];
    FILE *f;
    void *buf;
    snprintf(path, sizeof path, "%s/%s", dir, name);
    f = fopen(path, "rb");
    if (!f) {
        printf("FAIL cannot open %s\n", path);
        failures++;
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc(*len);
    if (fread(buf, 1, *len, f) != *len) {
        printf("FAIL short read %s\n", path);
        failures++;
    }
    fclose(f);
    return buf;
}

static const uint32_t KEY[4] = {1, 2, 3, 4};

static tandem_rng make(const char *name) {
    if (strncmp(name, "k1234_K8_", 9) == 0) return tandem_from_key(KEY, 0, 8);
    if (strncmp(name, "k1234_", 6) == 0) return tandem_from_key(KEY, 0, 32);
    return tandem_seed(42, 0, 32);
}

#define CHECK_STREAM(name, type, fill, next, at)                                                   \
    do {                                                                                           \
        size_t len, n, i;                                                                          \
        type *want = slurp(dir, name, &len), *got;                                                 \
        tandem_rng a, b, c;                                                                        \
        if (!want) break;                                                                          \
        n = len / sizeof(type);                                                                    \
        got = malloc(len);                                                                         \
        a = b = c = make(name);                                                                    \
        fill(&a, got, n);                                                                          \
        if (memcmp(got, want, len) != 0) {                                                         \
            for (i = 0; i < n && got[i] == want[i]; i++) {}                                        \
            printf("FAIL %s: fill differs first at element %zu\n", name, i);                      \
            failures++;                                                                            \
        }                                                                                          \
        for (i = 0; i < n; i++)                                                                    \
            if (next(&b) != want[i]) {                                                             \
                printf("FAIL %s: scalar draw differs at element %zu\n", name, i);                 \
                failures++;                                                                        \
                break;                                                                             \
            }                                                                                      \
        if (tandem_position(&a) != tandem_position(&b)) {                                          \
            printf("FAIL %s: fill and draws end at different positions\n", name);                 \
            failures++;                                                                            \
        }                                                                                          \
        for (i = 0; i < n; i += 97)                                                                \
            if (at(&c, i) != want[i]) {                                                            \
                printf("FAIL %s: random access differs at element %zu\n", name, i);               \
                failures++;                                                                        \
                break;                                                                             \
            }                                                                                      \
        free(got);                                                                                 \
        free(want);                                                                                \
    } while (0)

static uint32_t at_u32(const tandem_rng *r, size_t i) { return tandem_at_u32(r, i); }
static uint64_t at_u64(const tandem_rng *r, size_t i) { return tandem_at_u64(r, i); }
static float at_f32(const tandem_rng *r, size_t i) { return tandem_at_f32(r, i); }
static double at_f64(const tandem_rng *r, size_t i) { return tandem_at_f64(r, i); }
static uint16_t at_f16(const tandem_rng *r, size_t i) {
    tandem_rng t = *r;
    uint16_t v = 0;
    for (size_t k = 0; k <= i; k++) v = tandem_next_f16_bits(&t);
    return v;
}
static uint32_t at_char(const tandem_rng *r, size_t i) {
    tandem_rng t = *r;
    uint32_t v = 0;
    for (size_t k = 0; k <= i; k++) v = tandem_next_char(&t);
    return v;
}
static uint8_t at_u8(const tandem_rng *r, size_t i) {
    tandem_rng t = *r;
    uint8_t v = 0;
    for (size_t k = 0; k <= i; k++) v = tandem_next_u8(&t);
    return v;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/data";
    CHECK_STREAM("k1234_K32_u32.bin", uint32_t, tandem_fill_u32, tandem_next_u32, at_u32);
    CHECK_STREAM("k1234_K32_u64.bin", uint64_t, tandem_fill_u64, tandem_next_u64, at_u64);
    CHECK_STREAM("k1234_K8_u32.bin", uint32_t, tandem_fill_u32, tandem_next_u32, at_u32);
    CHECK_STREAM("seed42_K32_f64.bin", double, tandem_fill_f64, tandem_next_f64, at_f64);
    CHECK_STREAM("seed42_K32_f32.bin", float, tandem_fill_f32, tandem_next_f32, at_f32);
    CHECK_STREAM("seed42_K32_u8.bin", uint8_t, tandem_fill_u8, tandem_next_u8, at_u8);
    CHECK_STREAM("seed42_K32_f16bits.bin", uint16_t, tandem_fill_f16_bits, tandem_next_f16_bits,
                 at_f16);
    CHECK_STREAM("seed42_K32_char.bin", uint32_t, tandem_fill_char, tandem_next_char, at_char);
    {
        size_t len, n, i;
        tandem_u128 *want = slurp(dir, "seed42_K32_u128.bin", &len), *got;
        if (want) {
            tandem_rng a = tandem_seed(42, 0, 32), b = a;
            n = len / sizeof *want;
            got = malloc(len);
            tandem_fill_u128(&a, got, n);
            for (i = 0; i < n; i++) {
                tandem_u128 s = tandem_next_u128(&b);
                if (got[i].lo != want[i].lo || got[i].hi != want[i].hi || s.lo != want[i].lo ||
                    s.hi != want[i].hi) {
                    printf("FAIL u128 differs at element %zu\n", i);
                    failures++;
                    break;
                }
            }
            free(got);
            free(want);
        }
    }
    {
        size_t len, n, i;
        double *want = slurp(dir, "seed42_K32_c64.bin", &len), *got;
        if (want) {
            tandem_rng a = tandem_seed(42, 0, 32), b = a;
            n = len / (2 * sizeof *want);
            got = malloc(len);
            tandem_fill_c64(&a, got, n);
            for (i = 0; i < n; i++) {
                double z[2];
                tandem_next_c64(&b, z);
                if (got[2 * i] != want[2 * i] || got[2 * i + 1] != want[2 * i + 1] ||
                    z[0] != want[2 * i] || z[1] != want[2 * i + 1]) {
                    printf("FAIL c64 differs at element %zu\n", i);
                    failures++;
                    break;
                }
            }
            free(got);
            free(want);
        }
    }
    {
        size_t len, n, i;
        float *want = slurp(dir, "seed42_K32_c32.bin", &len), *got;
        if (want) {
            tandem_rng a = tandem_seed(42, 0, 32), b = a;
            n = len / (2 * sizeof *want);
            got = malloc(len);
            tandem_fill_c32(&a, got, n);
            for (i = 0; i < n; i++) {
                float z[2];
                tandem_next_c32(&b, z);
                if (got[2 * i] != want[2 * i] || got[2 * i + 1] != want[2 * i + 1] ||
                    z[0] != want[2 * i] || z[1] != want[2 * i + 1]) {
                    printf("FAIL c32 differs at element %zu\n", i);
                    failures++;
                    break;
                }
            }
            free(got);
            free(want);
        }
    }
    {
        size_t len, i;
        uint8_t *want = slurp(dir, "seed42_K32_bool.bin", &len);
        if (want) {
            bool *got = malloc(len);
            tandem_rng r = tandem_seed(42, 0, 32);
            tandem_fill_bool(&r, got, len);
            for (i = 0; i < len; i++)
                if ((uint8_t)got[i] != want[i]) {
                    printf("FAIL bool fill differs at element %zu\n", i);
                    failures++;
                    break;
                }
            if (tandem_position(&r) != len) {
                printf("FAIL bool fill position\n");
                failures++;
            }
            free(got);
            free(want);
        }
    }
    if (failures) {
        printf("%d failures\n", failures);
        return 1;
    }
    puts("streams: ok");
    return 0;
}
