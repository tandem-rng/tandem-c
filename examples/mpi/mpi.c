/* Each rank owns a contiguous range of blocks. It fills its part of the field from the position
 * where that part starts in the global fill, and draws each block's normals from split(block). */
#include "common.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    tandem_rng rng = root();
    tandem_rng field = tandem_sub(&rng, FIELD), noise = tandem_sub(&rng, NOISE);
    uint64_t lo = (uint64_t)BLOCKS * (uint64_t)rank / (uint64_t)size;
    uint64_t hi = (uint64_t)BLOCKS * (uint64_t)(rank + 1) / (uint64_t)size;
    size_t n = (size_t)((hi - lo) * BLOCK);
    double *x = malloc(n * sizeof *x);
    if (!x && n) MPI_Abort(MPI_COMM_WORLD, 1);

    /* Element i of the field is draw i, at bit 64 i. */
    uint32_t key[4];
    tandem_key(&field, key);
    tandem_rng mine = tandem_from_key(key, 64 * lo * BLOCK, tandem_chunk_length(&field));
    tandem_fill_f64(&mine, x, n);

    uint32_t h[BLOCKS] = {0}, all[BLOCKS];
    for (uint64_t b = lo; b < hi; b++) h[b] = block_hash(&noise, b, x + (b - lo) * BLOCK);
    MPI_Reduce(h, all, BLOCKS, MPI_UINT32_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%08x\n", combine(all));

    free(x);
    MPI_Finalize();
    return 0;
}
