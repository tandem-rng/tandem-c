/* Each thread fills an element range whose ends fall anywhere, not on block edges, to show that
 * any cut of a fill gives the same values. Threads then share the blocks for the normals. */
#include "common.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    tandem_rng rng = root();
    tandem_rng field = tandem_sub(&rng, FIELD), noise = tandem_sub(&rng, NOISE);
    uint32_t key[4], K = tandem_chunk_length(&field);
    tandem_key(&field, key);
    double *x = malloc(N * sizeof *x);
    if (!x) return 1;
    uint32_t h[BLOCKS];

#pragma omp parallel
    {
        uint64_t t = (uint64_t)omp_get_thread_num(), T = (uint64_t)omp_get_num_threads();
        uint64_t a = N * t / T, b = N * (t + 1) / T;
        tandem_rng mine = tandem_from_key(key, 64 * a, K);
        tandem_fill_f64(&mine, x + a, (size_t)(b - a));
#pragma omp barrier
#pragma omp for schedule(dynamic)
        for (int64_t g = 0; g < BLOCKS; g++)
            h[g] = block_hash(&noise, (uint64_t)g, x + (uint64_t)g * BLOCK);
    }
    printf("%08x\n", combine(h));
    free(x);
    return 0;
}
