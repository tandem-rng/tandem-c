/* The C++ wrapper against the C and C++ standard library generators. Build: make bench. */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../tandem.hpp"

#if defined(__GLIBC__)
#include <features.h>
#define HAVE_ARC4RANDOM __GLIBC_PREREQ(2, 36)
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#define HAVE_ARC4RANDOM 1
#else
#define HAVE_ARC4RANDOM 0
#endif

static double now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

template <class F> static void bench(const char *label, std::size_t bytes, F body) {
    double best = 1e9;
    for (int rep = 0; rep < 7; rep++) {
        double t0 = now();
        body();
        double t = now() - t0;
        if (t < best) best = t;
    }
    std::printf("%-46s %8.2f GiB/s\n", label, double(bytes) / best / (1024.0 * 1024.0 * 1024.0));
}

int main() {
    const std::size_t n = std::size_t{1} << 24;
    std::vector<std::uint32_t> u32(n);
    std::vector<std::uint64_t> u64(n);
    std::vector<float> f32(n);
    std::vector<double> f64(n);
    volatile double sink = 0;

    tandem::rng rng(42);
    std::mt19937 mt(42);
    std::mt19937_64 mt64(42);
    std::uniform_real_distribution<float> uf(0.0f, 1.0f);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    // The chain has its own generator: sharing mt64 with the other loops halves its speed.
    std::mt19937_64 mtc(42);
    std::uniform_real_distribution<double> udc(0.0, 1.0);

    for (double t0 = now(); now() - t0 < 0.5;) rng.fill(u32.data(), n);

    std::puts("tandem::rng");
    bench("fill<uint32_t>", n * 4, [&] { rng.fill(u32.data(), n); });
    bench("fill<uint64_t>", n * 8, [&] { rng.fill(u64.data(), n); });
    bench("fill<float>", n * 4, [&] { rng.fill(f32.data(), n); });
    bench("fill<double>", n * 8, [&] { rng.fill(f64.data(), n); });
    bench("next<double> chain", n * 8, [&] {
        double acc = 0;
        for (std::size_t i = 0; i < n; i++) acc += rng.next<double>();
        sink = acc;
    });
    bench("uniform_real_distribution<double>", n * 8, [&] {
        for (std::size_t i = 0; i < n; i++) f64[i] = ud(rng);
    });

    std::puts("C standard library");
    bench("rand(), 31 bits per call", n * 4, [&] {
        for (std::size_t i = 0; i < n; i++) u32[i] = std::uint32_t(std::rand());
    });
#if !defined(_WIN32)
    bench("random(), 31 bits per call", n * 4, [&] {
        for (std::size_t i = 0; i < n; i++) u32[i] = std::uint32_t(random());
    });
#endif
#if HAVE_ARC4RANDOM
    bench("arc4random_buf", n * 4, [&] { arc4random_buf(u32.data(), n * 4); });
#endif

    std::puts("C++ standard library");
    bench("mt19937, uint32_t", n * 4, [&] {
        for (std::size_t i = 0; i < n; i++) u32[i] = mt();
    });
    bench("mt19937_64, uint64_t", n * 8, [&] {
        for (std::size_t i = 0; i < n; i++) u64[i] = mt64();
    });
    bench("mt19937, uniform_real_distribution<float>", n * 4, [&] {
        for (std::size_t i = 0; i < n; i++) f32[i] = uf(mt);
    });
    bench("mt19937_64, uniform_real_distribution<double>", n * 8, [&] {
        for (std::size_t i = 0; i < n; i++) f64[i] = ud(mt64);
    });
    bench("mt19937_64, generate_canonical<double, 53>", n * 8, [&] {
        for (std::size_t i = 0; i < n; i++) f64[i] = std::generate_canonical<double, 53>(mt64);
    });
    bench("mt19937_64, uniform_real_distribution<double> chain", n * 8, [&] {
        double acc = 0;
        for (std::size_t i = 0; i < n; i++) acc += udc(mtc);
        sink = acc;
    });
    (void)sink;
    return 0;
}
