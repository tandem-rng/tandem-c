/* Accuracy of the f32 normal fills against libm Box-Muller on the same uniforms.
 * Build: make tools/normal_accuracy. Run: ./tools/normal_accuracy [pairs, default 5e7]
 * The reference is the host rule of tandem-cuda, a float radius and a double angle, with the
 * relative error taken where |reference| > 1e-3 and the absolute error reported for the rest.
 * The f64 normals are a ziggurat, which samples its law exactly, so they have no counterpart. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../tandem.h"

#define BLOCK 8192u

int main(int argc, char **argv) {
    size_t pairs = argc > 1 ? (size_t)atof(argv[1]) : 50000000u, done;
    static float uf[2 * BLOCK], zf[2 * BLOCK];
    tandem_rng uni32 = tandem_seed(2025, 0, 0), g32 = uni32;
    double max_rel32 = 0, max_abs32 = 0;

    for (done = 0; done < pairs; done += BLOCK) {
        tandem_fill_f32(&uni32, uf, 2 * BLOCK);
        tandem_fill_normal_f32(&g32, zf, 2 * BLOCK);
        for (size_t j = 0; j < BLOCK; j++) {
            float rf = sqrtf(-2.0f * logf(1.0f - uf[2 * j]));
            double tf = 6.283185307179586 * (double)uf[2 * j + 1];
            float reff[2] = {rf * (float)cos(tf), rf * (float)sin(tf)};
            for (size_t k = 0; k < 2; k++) {
                double ef = fabs((double)zf[2 * j + k] - (double)reff[k]);
                if (ef > max_abs32) max_abs32 = ef;
                if (fabs((double)reff[k]) > 1e-3 && ef / fabs((double)reff[k]) > max_rel32)
                    max_rel32 = ef / fabs((double)reff[k]);
            }
        }
    }
    printf("%zu pairs\nf32 max abs %.3g, max rel %.3g (%.2f ulps)\n", done, max_abs32, max_rel32,
           max_rel32 / 0x1p-23);
    return 0;
}
