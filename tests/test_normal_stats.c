/* The law of 10^8 f64 normals against the standard normal: raw moments 1 to 6, the tail counts
 * beyond 3, 3.5, 4, 4.5 and 5, each within 4 standard errors, and Kolmogorov-Smirnov and
 * Anderson-Darling p-values above 0.001. Run by make stats. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../tandem.h"

#define N 100000000u

static int failures;

static void check(const char *name, double stat, int ok) {
    printf("%-22s %10.4f %s\n", name, stat, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Neumaier's compensated sum: the Anderson-Darling sum has 10^8 terms of size up to 10^9 and
 * must stay exact to well below 10^8. */
typedef struct {
    double s, c;
} sum;

static void add(sum *a, double x) {
    double t = a->s + x;
    a->c += fabs(a->s) >= fabs(x) ? (a->s - t) + x : (x - t) + a->s;
    a->s = t;
}

/* Pr(A^2 < z) for n to infinity, Marsaglia and Marsaglia, J. Stat. Softw. 9(2), 2004. */
static double ad_cdf(double z) {
    if (z < 2)
        return exp(-1.2337141 / z) / sqrt(z) *
               (2.00012 + (.247105 - (.0649821 - (.0347962 - (.011672 - .00168691 * z) * z) * z) * z) * z);
    return exp(-exp(1.0776 - (2.30695 - (.43424 - (.082433 - (.008056 - .0003146 * z) * z) * z) * z) * z));
}

int main(void) {
    static const double mean[13] = {1, 0, 1, 0, 3, 0, 15, 0, 105, 0, 945, 0, 10395};
    static const double tails[5] = {3, 3.5, 4, 4.5, 5};
    double *x = malloc(N * sizeof *x), m[7] = {0}, d = 0, p = 0, a2;
    size_t count[5] = {0}, i;
    sum s = {0, 0};
    int k;
    tandem_rng g = tandem_seed(2026, 11, 0);

    tandem_fill_normal_f64(&g, x, N);
    for (i = 0; i < N; i++) {
        double xk = 1, ax = fabs(x[i]);
        for (k = 1; k <= 6; k++) m[k] += xk *= x[i];
        for (k = 0; k < 5; k++) count[k] += (size_t)(ax > tails[k]);
    }
    for (k = 1; k <= 6; k++) {
        char name[32];
        double se = sqrt((mean[2 * k] - mean[k] * mean[k]) / N), z = (m[k] / N - mean[k]) / se;
        snprintf(name, sizeof name, "moment %d, z", k);
        check(name, z, fabs(z) < 4);
    }
    for (k = 0; k < 5; k++) {
        char name[32];
        double q = erfc(tails[k] / sqrt(2.0)), z = ((double)count[k] - N * q) / sqrt(N * q * (1 - q));
        snprintf(name, sizeof name, "|x| > %.1f, z", tails[k]);
        check(name, z, fabs(z) < 4);
    }

    qsort(x, N, sizeof *x, cmp_double);
    for (i = 0; i < N; i++) {
        double lo = log(0.5 * erfc(-x[i] / sqrt(2.0))), hi = log(0.5 * erfc(x[i] / sqrt(2.0)));
        double f = exp(lo);
        d = fmax(d, fmax(f - (double)i / N, (double)(i + 1) / N - f));
        add(&s, (2.0 * (double)i + 1) * lo + (2.0 * (double)(N - i) - 1) * hi);
    }
    d *= sqrt((double)N);
    for (k = 1; k < 100; k++) p += (k % 2 ? 2.0 : -2.0) * exp(-2.0 * k * k * d * d);
    check("KS sqrt(n) D", d, 1);
    check("KS p", p, p > 0.001);
    a2 = -(double)N - (s.s + s.c) / N;
    check("AD A^2", a2, 1);
    check("AD p", 1 - ad_cdf(a2), 1 - ad_cdf(a2) > 0.001);
    free(x);
    if (failures) {
        puts("FAIL normal stats");
        return 1;
    }
    return 0;
}
