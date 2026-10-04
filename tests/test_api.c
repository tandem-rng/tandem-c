/* Functions beyond the specification's draws: positioning, bounded integers, normals. */
#include <stdio.h>
#include <string.h>

#include "../tandem.h"

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

int main(void) {
    test_set_position();
    if (failures) {
        printf("%d failures\n", failures);
        return 1;
    }
    puts("api: ok");
    return 0;
}
