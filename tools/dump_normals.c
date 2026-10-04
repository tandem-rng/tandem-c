/* Writes normal fills to stdout as raw bytes, for comparing builds byte for byte:
 *   ./dump_normals | shasum
 * The same bytes must come out of every compiler and target. */
#include <stdio.h>
#include <stdlib.h>

#include "../tandem.h"

int main(void) {
    enum { PAIRS = 1000000 };
    uint64_t starts[] = {0, 1, 77, 12345, 1u << 30};
    double *d = malloc(2 * PAIRS * sizeof *d);
    float *f = malloc(2 * PAIRS * sizeof *f);
    for (size_t i = 0; i < sizeof starts / sizeof starts[0]; i++) {
        tandem_rng g = tandem_seed(2026, 7, 0);
        tandem_set_position(&g, starts[i]);
        tandem_fill_normal_f64(&g, d, 2 * PAIRS - 1); /* odd count, drops the last sin half */
        fwrite(d, sizeof *d, 2 * PAIRS - 1, stdout);
        tandem_fill_normal_f32(&g, f, 2 * PAIRS - 1);
        fwrite(f, sizeof *f, 2 * PAIRS - 1, stdout);
    }
    return 0;
}
