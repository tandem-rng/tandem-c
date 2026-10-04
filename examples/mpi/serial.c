/* The reference: one fill of the whole field on one thread. */
#include "common.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    tandem_rng rng = root();
    tandem_rng field = tandem_sub(&rng, FIELD), noise = tandem_sub(&rng, NOISE);
    double *x = malloc(N * sizeof *x);
    if (!x) return 1;
    tandem_fill_f64(&field, x, N);
    uint32_t h[BLOCKS];
    for (uint64_t b = 0; b < BLOCKS; b++) h[b] = block_hash(&noise, b, x + b * BLOCK);
    printf("%08x\n", combine(h));
    free(x);
    return 0;
}
