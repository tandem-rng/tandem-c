/* Writes the f64 normal fills of tests/test_normal_bits.c to stdout as raw bytes, for comparing
 * builds and ports byte for byte:
 *   ./dump_normals | shasum -a 256
 * The same bytes must come out of every compiler, target and port. */
#include <stdio.h>
#include <stdlib.h>

#include "../tandem.h"

int main(void) {
    enum { N = 1000000 };
    uint64_t starts[] = {0, 1, 77, 12345, 1u << 30};
    double *d = malloc(N * sizeof *d);
    for (size_t i = 0; i < sizeof starts / sizeof starts[0]; i++) {
        tandem_rng g = tandem_seed(2026, 7, 0);
        tandem_set_position(&g, starts[i]);
        tandem_fill_normal_f64(&g, d, N);
        fwrite(d, sizeof *d, N, stdout);
    }
    free(d);
    return 0;
}
