/* Conformance against the specification's test vectors (tests/vectors.h). */
#include <stdio.h>
#include <string.h>

#include "../tandem.h"
#include "vectors.h"

static int failures;

#define LEN(a) (sizeof(a) / sizeof((a)[0]))

static void check_words(const char *what, const uint32_t got[4], const uint32_t want[4]) {
    if (memcmp(got, want, 4 * sizeof *got) != 0) {
        failures++;
        printf("FAIL %s: got %08x %08x %08x %08x, want %08x %08x %08x %08x\n", what, got[0], got[1],
               got[2], got[3], want[0], want[1], want[2], want[3]);
    }
}

static void check_u64(const char *what, uint64_t i, uint64_t got, uint64_t want) {
    if (got != want) {
        failures++;
        printf("FAIL %s[%llu]: got %llx, want %llx\n", what, (unsigned long long)i,
               (unsigned long long)got, (unsigned long long)want);
    }
}

static void check_f64(const char *what, uint64_t i, double got, double want) {
    if (got != want) {
        failures++;
        printf("FAIL %s[%llu]: got %.17g, want %.17g\n", what, (unsigned long long)i, got, want);
    }
}

int main(void) {
    for (size_t i = 0; i < LEN(VEC_T); i++) {
        uint32_t o[4], h[4];
        memcpy(o, VEC_T[i].o, sizeof o);
        memcpy(h, VEC_T[i].h, sizeof h);
        tandem_T(o, h);
        check_words("T o", o, VEC_T[i].o_out);
        check_words("T h", h, VEC_T[i].h_out);
    }

    for (size_t i = 0; i < LEN(VEC_F); i++) {
        uint32_t o[4], h[4];
        tandem_F_keyed(VEC_KEY, VEC_F[i].counter, 0x9e3779b9u, 0x94d049bbu, o, h);
        check_words("F o", o, VEC_F[i].o);
        check_words("F h", h, VEC_F[i].h);
    }

    /* Stream words by three routes: a long fill, scalar draws, and random access. */
    {
        tandem_rng rng = tandem_from_key(VEC_KEY, 0, VEC_K);
        tandem_rng scalar = rng;
        uint32_t fill[64];
        tandem_fill_u32(&rng, fill, LEN(fill));
        for (size_t i = 0; i < LEN(VEC_STREAM); i++) {
            uint64_t w0 = VEC_STREAM[i].first_word;
            for (unsigned k = 0; k < 4; k++) {
                uint32_t want = VEC_STREAM[i].words[k];
                check_u64("fill u32", w0 + k, fill[w0 + k], want);
                check_u64("at u32", w0 + k, tandem_at_u32(&scalar, w0 + k), want);
            }
        }
        for (uint64_t w = 0; w < LEN(fill); w++)
            check_u64("next u32", w, tandem_next_u32(&scalar), fill[w]);
        check_u64("position", 0, tandem_position(&scalar), 32u * LEN(fill));
    }

    {
        tandem_rng rng = tandem_from_key(VEC_KEY, 0, VEC_K);
        for (size_t i = 0; i < LEN(VEC_F64); i++)
            check_f64("f64", VEC_F64[i].index, tandem_at_f64(&rng, VEC_F64[i].index),
                      VEC_F64[i].value);
        for (size_t i = 0; i < LEN(VEC_F32); i++)
            check_f64("f32", VEC_F32[i].index, tandem_at_f32(&rng, VEC_F32[i].index),
                      VEC_F32[i].value);
        for (size_t i = 0; i < LEN(VEC_BOOL); i++) {
            tandem_rng b = rng;
            bool got = false;
            for (uint64_t k = 0; k <= VEC_BOOL[i].index; k++) got = tandem_next_bool(&b);
            check_u64("bool", VEC_BOOL[i].index, got, (uint64_t)VEC_BOOL[i].value);
        }
    }

    {
        tandem_rng rng = tandem_from_key(VEC_KEY, 0, VEC_K), kids[2], c;
        uint32_t key[4];
        c = tandem_split(&rng, 0);
        tandem_key(&c, key);
        check_words("split 0", key, VEC_SPLIT0);
        c = tandem_split(&rng, 1);
        tandem_key(&c, key);
        check_words("split 1", key, VEC_SPLIT1);
        c = tandem_sub(&rng, 7);
        tandem_key(&c, key);
        check_words("purpose 7", key, VEC_PURPOSE7);
        tandem_fork(&rng, kids, 2);
        tandem_key(&kids[0], key);
        check_words("fork 0", key, VEC_FORK0);
        check_u64("fork parent position", 0, tandem_position(&rng), 128u);
    }

    {
        tandem_rng rng = tandem_seed(VEC_SEED, 0, VEC_K);
        uint32_t key[4];
        tandem_key(&rng, key);
        check_words("seed key", key, VEC_SEED_KEY);
        for (size_t i = 0; i < LEN(VEC_SEED_F64); i++)
            check_f64("seed f64", VEC_SEED_F64[i].index,
                      tandem_at_f64(&rng, VEC_SEED_F64[i].index), VEC_SEED_F64[i].value);
        for (size_t i = 0; i < LEN(VEC_SEED_U32); i++)
            check_u64("seed u32", VEC_SEED_U32[i].index,
                      tandem_at_u32(&rng, VEC_SEED_U32[i].index), VEC_SEED_U32[i].value);
    }

    /* Weighted choice, Appendix C: the table and the indices of a fill and of scalar draws. */
    for (size_t c = 0; c < LEN(VEC_CHOICE); c++) {
        enum { N = LEN(VEC_CHOICE[0].indices) };
        size_t m = VEC_CHOICE[c].m;
        uint64_t cut[16];
        uint32_t alias[16], fill[N];
        tandem_choice_table t;
        tandem_rng rng = tandem_from_key(VEC_KEY, 0, VEC_K), scalar = rng;
        if (m > LEN(cut) || !tandem_choice_build(&t, VEC_CHOICE[c].w, m, cut, alias)) {
            failures++;
            printf("FAIL choice %zu: no table\n", c);
            continue;
        }
        check_u64("choice capacity", c, t.capacity, VEC_CHOICE[c].capacity);
        for (size_t j = 0; j < m; j++) {
            check_u64("choice cut", j, cut[j], VEC_CHOICE[c].cut[j]);
            check_u64("choice alias", j, alias[j], VEC_CHOICE[c].alias[j]);
        }
        tandem_fill_choice(&rng, fill, N, &t);
        for (size_t i = 0; i < N; i++) {
            check_u64("choice fill", i, fill[i], VEC_CHOICE[c].indices[i]);
            check_u64("choice scalar", i, tandem_choice(&scalar, &t), VEC_CHOICE[c].indices[i]);
        }
        check_u64("choice position", c, tandem_position(&rng), 64u * N);
    }

    if (failures) {
        printf("%d failures\n", failures);
        return 1;
    }
    puts("vectors: ok");
    return 0;
}
