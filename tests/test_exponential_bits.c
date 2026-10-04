/* The exponential fills are bit identical on every compiler and target: a hash of 1e6 f64 and
 * 1e6 f32 exponentials from each of several positions must equal the value recorded from the M4
 * clang build. tools/dump_exponentials.c writes the same bytes, whose SHA-256 is
 * 5c035a4ef1368231d25a9c2f9201be2df3224e28a14549a50625d0db3770ef4e. */
#include <stdio.h>
#include <stdlib.h>

#include "../tandem.h"

#define EXPECTED_HASH 0x47f8f98297d94ee2ull

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

int main(void) {
    enum { N = 1000000 };
    uint64_t starts[] = {0, 1, 77, 12345, 1u << 30}, h = 0xcbf29ce484222325ull;
    double *d = malloc(N * sizeof *d);
    float *f = malloc(N * sizeof *f);
    for (size_t i = 0; i < sizeof starts / sizeof starts[0]; i++) {
        tandem_rng g = tandem_seed(2026, 7, 0);
        tandem_set_position(&g, starts[i]);
        tandem_fill_exponential_f64(&g, d, N);
        h = fnv(h, d, N * sizeof *d);
        tandem_fill_exponential_f32(&g, f, N);
        h = fnv(h, f, N * sizeof *f);
    }
    free(d);
    free(f);
    printf("exponential bits: FNV-1a %016llx, expected %016llx\n", (unsigned long long)h,
           (unsigned long long)EXPECTED_HASH);
    if (h != EXPECTED_HASH) {
        puts("FAIL exponential bits");
        return 1;
    }
    return 0;
}
