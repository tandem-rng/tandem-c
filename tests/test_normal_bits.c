/* The normal fills are bit identical on every compiler and target: a hash of 1e6 pairs of f64 and
 * f32 normals from several positions must equal the value recorded from the M4 clang build,
 * which tools/dump_normals.c writes as raw bytes. Explicit fused multiply-adds make this hold. */
#include <stdio.h>
#include <stdlib.h>

#include "../tandem.h"

#define EXPECTED_HASH 0x9414e1315e2653beull

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

int main(void) {
    enum { PAIRS = 1000000 };
    uint64_t starts[] = {0, 1, 77, 12345, 1u << 30}, h = 0xcbf29ce484222325ull;
    double *d = malloc(2 * PAIRS * sizeof *d);
    float *f = malloc(2 * PAIRS * sizeof *f);
    for (size_t i = 0; i < sizeof starts / sizeof starts[0]; i++) {
        tandem_rng g = tandem_seed(2026, 7, 0);
        tandem_set_position(&g, starts[i]);
        tandem_fill_normal_f64(&g, d, 2 * PAIRS - 1);
        h = fnv(h, d, (2 * PAIRS - 1) * sizeof *d);
        tandem_fill_normal_f32(&g, f, 2 * PAIRS - 1);
        h = fnv(h, f, (2 * PAIRS - 1) * sizeof *f);
    }
    free(d);
    free(f);
    if (h != EXPECTED_HASH) {
        printf("FAIL normal bits: hash %016llx, expected %016llx\n", (unsigned long long)h,
               (unsigned long long)EXPECTED_HASH);
        return 1;
    }
    puts("normal bits: ok");
    return 0;
}
