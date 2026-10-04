/* Writes the exponential fills of tests/test_exponential_bits.c to stdout as raw bytes, for
 * comparing builds and ports byte for byte:
 *   ./dump_exponentials | shasum -a 256
 * The same bytes must come out of every compiler and target. */
#include <stdio.h>
#include <stdlib.h>

#include "../tandem.h"

int main(void) {
    enum { N = 1000000 };
    uint64_t starts[] = {0, 1, 77, 12345, 1u << 30};
    double *d = malloc(N * sizeof *d);
    float *f = malloc(N * sizeof *f);
    for (size_t i = 0; i < sizeof starts / sizeof starts[0]; i++) {
        tandem_rng g = tandem_seed(2026, 7, 0);
        tandem_set_position(&g, starts[i]);
        tandem_fill_exponential_f64(&g, d, N);
        fwrite(d, sizeof *d, N, stdout);
        tandem_fill_exponential_f32(&g, f, N);
        fwrite(f, sizeof *f, N, stdout);
    }
    free(d);
    free(f);
    return 0;
}
