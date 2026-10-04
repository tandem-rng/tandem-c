/* The normal fills are bit identical on every compiler and target, and the f64 normals across
 * ports. Each check hashes the output bytes with FNV-1a.
 * - f64: 10^6 normals from each of five positions, the bytes tools/dump_normals.c writes, whose
 *   SHA-256 is 700ec4d2f4d6b82aaa56c6eff18a4e5919585fdbd093988773383d580ea610d1.
 * - f64 against a Python implementation written from the text of Appendix A: 2 x 10^5 normals
 *   from key {1, 2, 3, 4}, K = 32, at bits 0 and 2373, with their end positions.
 * - f32: 2 x 10^6 - 1 normals from each of the five positions, unchanged since the explicit fused
 *   multiply-adds. */
#include <stdio.h>
#include <stdlib.h>

#include "../tandem.h"

#define F64_HASH 0xa61cfa844c85f7c1ull
#define F32_HASH 0xaa1ea656ce73a4fbull

static int failures;

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

static void check(const char *name, uint64_t got, uint64_t want) {
    printf("%s: FNV-1a %016llx, expected %016llx\n", name, (unsigned long long)got,
           (unsigned long long)want);
    if (got != want) failures++;
}

int main(void) {
    enum { N = 1000000, PAIRS = 1000000, REF = 200000 };
    uint64_t starts[] = {0, 1, 77, 12345, 1u << 30}, h = 0xcbf29ce484222325ull;
    double *d = malloc(N * sizeof *d);
    float *f = malloc(2 * PAIRS * sizeof *f);
    size_t i;

    for (i = 0; i < sizeof starts / sizeof starts[0]; i++) {
        tandem_rng g = tandem_seed(2026, 7, 0);
        tandem_set_position(&g, starts[i]);
        tandem_fill_normal_f64(&g, d, N);
        h = fnv(h, d, N * sizeof *d);
    }
    check("normal f64", h, F64_HASH);

    {
        const uint32_t key[4] = {1, 2, 3, 4};
        const struct {
            uint64_t start, hash, end;
        } ref[] = {{0, 0x0c4059ed409d578dull, 12800000}, {2373, 0x30ce40c86b295193ull, 12802432}};
        for (i = 0; i < 2; i++) {
            tandem_rng g = tandem_from_key(key, ref[i].start, 32);
            tandem_fill_normal_f64(&g, d, REF);
            check("normal f64, Python reference", fnv(0xcbf29ce484222325ull, d, REF * sizeof *d), ref[i].hash);
            if (tandem_position(&g) != ref[i].end) failures++;
        }
    }

    h = 0xcbf29ce484222325ull;
    for (i = 0; i < sizeof starts / sizeof starts[0]; i++) {
        tandem_rng g = tandem_seed(2026, 7, 0);
        tandem_set_position(&g, starts[i]);
        tandem_fill_normal_f32(&g, f, 2 * PAIRS - 1);
        h = fnv(h, f, (2 * PAIRS - 1) * sizeof *f);
    }
    check("normal f32", h, F32_HASH);

    free(d);
    free(f);
    if (failures) {
        puts("FAIL normal bits");
        return 1;
    }
    return 0;
}
