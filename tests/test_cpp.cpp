// The C++ wrapper agrees with the C API and drives <random>.
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <random>
#include <vector>

#include "../tandem.hpp"
#include "../tandem123.h" // compiles as C++ too

#if __cplusplus >= 202002L
static_assert(std::uniform_random_bit_generator<tandem::rng>);
#endif

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

    {
        // The remaining draw types, positioning, random access, bounded integers, normals.
        tandem::rng h(5);
        tandem_rng d = tandem_seed(5, 0, 0);
        CHECK(h.next<char32_t>() == tandem_next_char(&d));
        CHECK(h.next<tandem::float16_bits>().bits == tandem_next_f16_bits(&d));
        std::complex<float> c32 = h.next<std::complex<float>>();
        float f2[2];
        tandem_next_c32(&d, f2);
        CHECK(c32.real() == f2[0] && c32.imag() == f2[1]);
        std::complex<double> c64 = h.next<std::complex<double>>();
        double d2[2];
        tandem_next_c64(&d, d2);
        CHECK(c64.real() == d2[0] && c64.imag() == d2[1]);
#ifdef __SIZEOF_INT128__
        tandem_u128 w = tandem_next_u128(&d);
        CHECK(h.next<tandem::uint128>() == ((tandem::uint128{w.hi} << 64) | w.lo));
#endif
        CHECK(h.position() == tandem_position(&d));

        CHECK(h.at<std::uint32_t>(3) == tandem_at_u32(&d, 3));
        CHECK(h.at<std::uint64_t>(3) == tandem_at_u64(&d, 3));
        CHECK(h.at<float>(3) == tandem_at_f32(&d, 3));
        CHECK(h.at<double>(3) == tandem_at_f64(&d, 3));
        CHECK(h.position() == tandem_position(&d));

        CHECK(h.below<std::uint32_t>(1000) == tandem_u32_below(&d, 1000));
        CHECK(h.below<std::uint64_t>(1ull << 40) == tandem_u64_below(&d, 1ull << 40));
        CHECK(h.normal() == tandem_normal_f64(&d));
        CHECK(h.normal<float>() == tandem_normal_f32(&d));
        double z2[2];
        tandem_normal2_f64(&d, z2);
        CHECK((h.normal2() == std::array<double, 2>{z2[0], z2[1]}));
        CHECK(h.position() == tandem_position(&d));

        CHECK(h.set_position(100) && tandem_set_position(&d, 100));
        CHECK(h.next<std::uint64_t>() == tandem_next_u64(&d));
        CHECK(!h.set_position(std::uint64_t{1} << 63));
        CHECK(h.position() == tandem_position(&d));
    }

    {
        // The standard engine requirements: seed, discard in constant time, stream round trip,
        // and the standard algorithms and distributions, which must repeat across two runs.
        tandem::rng a(11), b(11);
        a.discard(1000000000000ull);
        CHECK(a.position() == (std::uint64_t{1000000000000ull} * 64u));
        a.seed(11);
        CHECK(a == b);
        a.seed();
        CHECK(a == tandem::rng(0));

        tandem::rng x(5);
        x();
        x.discard(3);
        tandem::rng y(5);
        for (int i = 0; i < 4; i++) y();
        CHECK(x == y && x() == y());

        std::stringstream ss;
        ss << x;
        tandem::rng restored;
        ss >> restored;
        CHECK(restored == x && restored() == x());

        auto shuffled = [] {
            tandem::rng r(3);
            std::vector<int> v(100);
            std::iota(v.begin(), v.end(), 0);
            std::shuffle(v.begin(), v.end(), r);
            return v;
        };
        std::vector<int> s1 = shuffled(), s2 = shuffled(), id(100);
        std::iota(id.begin(), id.end(), 0);
        CHECK(s1 == s2 && s1 != id);
        CHECK(std::is_permutation(s1.begin(), s1.end(), id.begin()));

        auto gauss = [] {
            tandem::rng r(3);
            std::normal_distribution<double> n;
            return std::vector<double>{n(r), n(r), n(r)};
        };
        CHECK(gauss() == gauss());
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
