/* Functions beyond the specification's draws: positioning, bounded integers, normals. */
#include <stdio.h>
#include <string.h>

#include "../tandem.h"
#include "cross_below.h"

static int failures;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                \
        }                                                                                          \
    } while (0)

/* A rewound or jumped generator draws what a fresh one at that position draws, whatever the
 * cache held, and a rejected position changes nothing. */
static void test_set_position(void) {
    uint32_t key[4] = {1, 2, 3, 4};
    tandem_rng g = tandem_from_key(key, 0, 0), fresh;
    uint64_t pos[] = {0, 64, 1000, 1024 * 33 + 64, 4096, 17};
    size_t i;

    for (i = 0; i < 5000; i++) tandem_next_u64(&g);
    for (i = 0; i < sizeof pos / sizeof pos[0]; i++) {
        fresh = tandem_from_key(key, pos[i], 0);
        CHECK(tandem_set_position(&g, pos[i]));
        CHECK(tandem_position(&g) == pos[i]);
        CHECK(tandem_next_u64(&g) == tandem_next_u64(&fresh));
        CHECK(tandem_next_u32(&g) == tandem_next_u32(&fresh));
    }

    {
        tandem_rng before = g;
        CHECK(tandem_set_position(&g, ((uint64_t)1 << 63) - 1u));
        CHECK(tandem_set_position(&g, 0));
        g = before;
        CHECK(!tandem_set_position(&g, (uint64_t)1 << 63));
        CHECK(!tandem_set_position(&g, ~(uint64_t)0));
        CHECK(memcmp(&g, &before, sizeof g) == 0);
    }
}

/* Values and stream position of tandem-cuda's urand(range) and urand64(range). The position
 * pins the number of rejected draws as well as the values. */
static void test_below_cross(void) {
    size_t c, i;
    for (c = 0; c < sizeof CROSS_U32 / sizeof CROSS_U32[0]; c++) {
        tandem_rng g = tandem_seed(42, 0, 0), f;
        tandem_next_bool(&g);
        f = g;
        for (i = 0; i < CROSS_COUNT; i++) CHECK(tandem_u32_below(&g, CROSS_U32[c].n) == CROSS_U32[c].want[i]);
        CHECK(tandem_position(&g) == CROSS_U32[c].end_pos);
        {
            uint32_t out[CROSS_COUNT];
            tandem_fill_u32_below(&f, out, CROSS_COUNT, CROSS_U32[c].n);
            CHECK(memcmp(out, CROSS_U32[c].want, sizeof out) == 0);
            CHECK(tandem_position(&f) == CROSS_U32[c].end_pos);
        }
    }
    for (c = 0; c < sizeof CROSS_U64 / sizeof CROSS_U64[0]; c++) {
        tandem_rng g = tandem_seed(42, 0, 0), f;
        tandem_next_bool(&g);
        f = g;
        for (i = 0; i < CROSS_COUNT; i++) CHECK(tandem_u64_below(&g, CROSS_U64[c].n) == CROSS_U64[c].want[i]);
        CHECK(tandem_position(&g) == CROSS_U64[c].end_pos);
        {
            uint64_t out[CROSS_COUNT];
            tandem_fill_u64_below(&f, out, CROSS_COUNT, CROSS_U64[c].n);
            CHECK(memcmp(out, CROSS_U64[c].want, sizeof out) == 0);
            CHECK(tandem_position(&f) == CROSS_U64[c].end_pos);
        }
    }
}

/* n = 1 and n = 0 return 0 and consume one draw, as core.hpp does. */
static void test_below_degenerate(void) {
    tandem_rng g = tandem_seed(1, 2, 0);
    CHECK(tandem_u32_below(&g, 0) == 0 && tandem_position(&g) == 32);
    CHECK(tandem_u64_below(&g, 0) == 0 && tandem_position(&g) == 128);
}

int main(void) {
    test_set_position();
    test_below_cross();
    test_below_degenerate();
    if (failures) {
        printf("%d failures\n", failures);
        return 1;
    }
    puts("api: ok");
    return 0;
}
