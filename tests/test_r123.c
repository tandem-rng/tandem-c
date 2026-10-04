/* tandem123.h agrees with the fills and with the reference stream dumps. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tandem.h"
#include "../tandem123.h"

static int failures;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                \
        }                                                                                          \
    } while (0)

static tandem4x32_ctr_t counter(uint64_t block, uint32_t K) {
    tandem4x32_ctr_t c = {{(uint32_t)block, (uint32_t)(block >> 32), K, 0}};
    return c;
}

/* Block b is words 4b .. 4b + 3 of the u32 fill, for chunk lengths from 1 to 65536 and blocks
 * across lane, row, and group boundaries. */
static void test_against_fill(void) {
    uint32_t key[4] = {0x421d21ebu, 0x32d31777u, 0x62e7564bu, 0xdf2bdf82u};
    uint32_t Ks[] = {0, 1, 2, 8, 32, 256, 65536};
    tandem4x32_key_t k = {{key[0], key[1], key[2], key[3]}};
    uint64_t blocks[] = {0, 1, 7, 8, 9, 255, 256, 257, 1000, 4096, 100003, 1u << 20};
    size_t i, j;

    for (i = 0; i < sizeof Ks / sizeof Ks[0]; i++)
        for (j = 0; j < sizeof blocks / sizeof blocks[0]; j++) {
            tandem_rng g = tandem_from_key(key, 128u * blocks[j], Ks[i]);
            uint32_t want[4];
            tandem4x32_ctr_t r = tandem4x32(counter(blocks[j], Ks[i]), k);
            tandem_fill_u32(&g, want, 4);
            CHECK(memcmp(r.v, want, sizeof want) == 0);
        }
}

/* The block function agrees with the specification's tandem_block, also for counters above 2^32. */
static void test_against_block(void) {
    uint32_t key[4] = {1, 2, 3, 4};
    tandem4x32_key_t k = {{1, 2, 3, 4}};
    uint64_t block = ((uint64_t)5 << 32) + 77u, row = block >> 3, c, step;
    uint32_t want[4];
    tandem4x32_ctr_t r = tandem4x32(counter(block, 8), k);

    c = 8u * (row >> 3) + (block & 7u);
    step = row & 7u;
    tandem_block(key, c, (uint32_t)step, want);
    CHECK(memcmp(r.v, want, sizeof want) == 0);
    CHECK(tandem4x32(counter(block, 8), k).v[0] == r.v[0]); /* pure function */
}

static void test_seed_key(void) {
    tandem_rng g = tandem_seed(42, 7, 0);
    uint32_t key[4];
    tandem4x32_key_t k = tandem4x32_key_from_seed(42, 7);
    tandem_key(&g, key);
    CHECK(memcmp(k.v, key, sizeof key) == 0);
}

/* Reference dumps written by TandemRNG.jl: the u32 stream of key {1, 2, 3, 4}. */
static void test_against_dump(const char *dir, const char *name, uint32_t K) {
    char path[512];
    FILE *f;
    long len;
    uint32_t *want;
    tandem4x32_key_t k = {{1, 2, 3, 4}};
    uint64_t b;

    snprintf(path, sizeof path, "%s/%s", dir, name);
    f = fopen(path, "rb");
    if (!f) {
        printf("FAIL cannot open %s\n", path);
        failures++;
        return;
    }
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    want = malloc((size_t)len);
    if (fread(want, 1, (size_t)len, f) != (size_t)len) failures++;
    fclose(f);
    for (b = 0; b < (uint64_t)len / 16u; b++) {
        tandem4x32_ctr_t r = tandem4x32(counter(b, K), k);
        if (memcmp(r.v, want + 4u * b, 16) != 0) {
            printf("FAIL %s: block %llu differs\n", name, (unsigned long long)b);
            failures++;
            break;
        }
    }
    free(want);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/data";
    test_against_fill();
    test_against_block();
    test_seed_key();
    test_against_dump(dir, "k1234_K32_u32.bin", 32);
    test_against_dump(dir, "k1234_K8_u32.bin", 8);
    if (failures) {
        printf("%d failures\n", failures);
        return 1;
    }
    puts("random123: ok");
    return 0;
}
