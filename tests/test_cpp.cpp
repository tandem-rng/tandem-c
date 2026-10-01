// The C++ wrapper agrees with the C API and drives <random>.
#include <cstdio>
#include <random>
#include <vector>

#include "../tandem.hpp"

static int failures;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                            \
        }                                                                                          \
    } while (0)

int main() {
    tandem::rng g(42);
    tandem_rng c = tandem_seed(42, 0, 0);

    for (int i = 0; i < 1000; i++) CHECK(g() == tandem_next_u64(&c));
    CHECK(g.position() == tandem_position(&c));

    CHECK(g.next<bool>() == tandem_next_bool(&c));
    CHECK(g.next<std::uint8_t>() == tandem_next_u8(&c));
    CHECK(g.next<std::uint16_t>() == tandem_next_u16(&c));
    CHECK(g.next<std::uint32_t>() == tandem_next_u32(&c));
    CHECK(g.next<float>() == tandem_next_f32(&c));
    CHECK(g.next<double>() == tandem_next_f64(&c));

    {
        std::vector<float> want(5000);
        tandem_fill_f32(&c, want.data(), want.size());
        CHECK(g.fill<float>(want.size()) == want);
        std::vector<std::uint64_t> w64(777), g64(777);
        tandem_fill_u64(&c, w64.data(), w64.size());
        g.fill(g64.data(), g64.size());
        CHECK(g64 == w64);
        CHECK(g.position() == tandem_position(&c));
    }

    {
        tandem::rng h = g;
        tandem_rng d = c;
        h.discard(10);
        for (int i = 0; i < 10; i++) tandem_next_u64(&d);
        CHECK(h.position() == tandem_position(&d));
        CHECK(h() == tandem_next_u64(&d));
    }

    {
        tandem_rng s = tandem_split(&c, 9), u = tandem_sub(&c, 3), kids[3];
        using key_type = tandem::rng::key_type;
        CHECK(g.split(9).key() == (key_type{s.key[0], s.key[1], s.key[2], s.key[3]}));
        CHECK(g.sub(3).key() == (key_type{u.key[0], u.key[1], u.key[2], u.key[3]}));
        auto forks = g.fork(3);
        tandem_fork(&c, kids, 3);
        CHECK(forks.size() == 3);
        for (int i = 0; i < 3; i++) CHECK(forks[i].key()[0] == kids[i].key[0] && forks[i].position() == 0);
        CHECK(g.position() == tandem_position(&c));
    }

    {
        // <random> distributions compile and run; the mean of a uniform is near one half.
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::normal_distribution<double> gauss(0.0, 1.0);
        double sum = 0;
        for (int i = 0; i < 100000; i++) sum += uni(g);
        CHECK(sum > 49000 && sum < 51000);
        volatile double sink = gauss(g);
        (void)sink;
        std::uniform_int_distribution<int> die(1, 6);
        int v = die(g);
        CHECK(v >= 1 && v <= 6);
    }

    CHECK(tandem::rng(7) == tandem::rng(7));
    CHECK(tandem::rng(7) != tandem::rng(8));

    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::puts("c++: ok");
    return 0;
}
