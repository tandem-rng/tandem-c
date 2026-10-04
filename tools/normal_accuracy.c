/* Accuracy of the normal fills against libm Box-Muller on the same uniforms.
 * Build: make tools/normal_accuracy. Run: ./tools/normal_accuracy [pairs, default 5e7]
 * f64: the reference does the quarter-turn reduction exactly, so that libm sees an angle in
 * [-pi/4, pi/4] and the relative error is meaningful near the zeros of cos and sin too.
 * f32: the host rule of tandem-cuda, a float radius and a double angle, with the relative error
 * taken where |reference| > 1e-3 and the absolute error reported for the rest. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../tandem.h"

#define BLOCK 8192u

int main(int argc, char **argv) {
    size_t pairs = argc > 1 ? (size_t)atof(argv[1]) : 50000000u, done;
    static double u[2 * BLOCK], z[2 * BLOCK];
    static float uf[2 * BLOCK], zf[2 * BLOCK];
    tandem_rng uni = tandem_seed(2024, 0, 0), g = uni, uni32 = tandem_seed(2025, 0, 0), g32 = uni32;
    double max_rel = 0, max_abs = 0, max_rel32 = 0, max_abs32 = 0;

    for (done = 0; done < pairs; done += BLOCK) {
        tandem_fill_f64(&uni, u, 2 * BLOCK);
        tandem_fill_normal_f64(&g, z, 2 * BLOCK);
        tandem_fill_f32(&uni32, uf, 2 * BLOCK);
        tandem_fill_normal_f32(&g32, zf, 2 * BLOCK);
        for (size_t j = 0; j < BLOCK; j++) {
            double r = sqrt(-2.0 * log(1.0 - u[2 * j])), b = u[2 * j + 1];
            double q = floor(4.0 * b + 0.5), t = 6.283185307179586 * (b - 0.25 * q);
            double c0 = cos(t), s0 = sin(t), ref[2];
            int qm = (int)q & 3;
            ref[0] = r * (qm == 0 ? c0 : qm == 1 ? -s0 : qm == 2 ? -c0 : s0);
            ref[1] = r * (qm == 0 ? s0 : qm == 1 ? c0 : qm == 2 ? -s0 : -c0);
            float rf = sqrtf(-2.0f * logf(1.0f - uf[2 * j]));
            double tf = 6.283185307179586 * (double)uf[2 * j + 1];
            float reff[2] = {rf * (float)cos(tf), rf * (float)sin(tf)};
            for (size_t k = 0; k < 2; k++) {
                double ea = fabs(z[2 * j + k] - ref[k]), ef = fabs((double)zf[2 * j + k] - (double)reff[k]);
                if (ea > max_abs) max_abs = ea;
                if (ef > max_abs32) max_abs32 = ef;
                if (ref[k] != 0 && ea / fabs(ref[k]) > max_rel) max_rel = ea / fabs(ref[k]);
                if (fabs((double)reff[k]) > 1e-3 && ef / fabs((double)reff[k]) > max_rel32)
                    max_rel32 = ef / fabs((double)reff[k]);
            }
        }
    }
    printf("%zu pairs\nf64 max abs %.3g, max rel %.3g\nf32 max abs %.3g, max rel %.3g (%.2f ulps)\n",
           done, max_abs, max_rel, max_abs32, max_rel32, max_rel32 / 0x1p-23);
    return 0;
}
