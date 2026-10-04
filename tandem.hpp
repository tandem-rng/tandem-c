/* Tandem8x32 for C++17: a thin value type over the C implementation that satisfies
 * std::uniform_random_bit_generator, so it drives every <random> distribution.
 *
 * Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#ifndef TANDEM_HPP
#define TANDEM_HPP

#include <array>
#include <cstddef>
#include <complex>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

#include "tandem.h"

namespace tandem {

#ifdef __SIZEOF_INT128__
/* __extension__ keeps -Wpedantic quiet about the compiler's 128-bit type. */
__extension__ typedef unsigned __int128 uint128;
#endif

/* The bit pattern of an IEEE binary16 value, a distinct type because uint16_t is already the
 * 16-bit integer draw. */
struct float16_bits {
    std::uint16_t bits;
};

class rng {
public:
    using result_type = std::uint64_t;
    using key_type = std::array<std::uint32_t, 4>;

    static constexpr result_type min() noexcept { return 0; }
    static constexpr result_type max() noexcept { return std::numeric_limits<result_type>::max(); }

    /* 128-bit seed as two halves. K is the chunk length, 0 for the default of 32. */
    explicit rng(std::uint64_t seed_lo = 0, std::uint64_t seed_hi = 0, std::uint32_t K = 0) noexcept
        : s_(tandem_seed(seed_lo, seed_hi, K)) {}

    static rng from_key(const key_type &key, std::uint64_t pos = 0, std::uint32_t K = 0) noexcept {
        return rng(tandem_from_key(key.data(), pos, K));
    }

    /* Transport form. */
    key_type key() const noexcept {
        key_type k;
        tandem_key(&s_, k.data());
        return k;
    }
    std::uint64_t position() const noexcept { return tandem_position(&s_); }
    /* False, with the generator unchanged, when pos >= 2^63. */
    bool set_position(std::uint64_t pos) noexcept { return tandem_set_position(&s_, pos); }
    std::uint32_t chunk_length() const noexcept { return tandem_chunk_length(&s_); }

    /* std::uniform_random_bit_generator: 64 bits per call. */
    result_type operator()() noexcept { return tandem_next_u64(&s_); }

    static constexpr result_type default_seed = 0;

    /* Reseed as the constructor does. */
    void seed(std::uint64_t seed_lo = default_seed, std::uint64_t seed_hi = 0,
              std::uint32_t K = 0) noexcept {
        s_ = tandem_seed(seed_lo, seed_hi, K);
    }

    /* Skip z 64-bit draws in constant time: the generator is counter based, so this is a move
     * of the position. */
    void discard(unsigned long long z) noexcept {
        set_position(((position() + 63u) & ~std::uint64_t{63}) + 64u * z);
    }

    /* The transport form as six decimal words: the key, the position, the chunk length. */
    friend std::ostream &operator<<(std::ostream &os, const rng &g) {
        key_type k = g.key();
        return os << k[0] << ' ' << k[1] << ' ' << k[2] << ' ' << k[3] << ' ' << g.position() << ' '
                  << g.chunk_length();
    }
    friend std::istream &operator>>(std::istream &is, rng &g) {
        key_type k;
        std::uint64_t pos;
        std::uint32_t K;
        if (is >> k[0] >> k[1] >> k[2] >> k[3] >> pos >> K) g = from_key(k, pos, K);
        return is;
    }

    /* Draw one value of T: bool, the fixed-width unsigned integers (uint128 where the compiler
     * has it), float, double, float16_bits, char32_t, or std::complex of float or double. */
    template <class T> T next() noexcept {
        if constexpr (std::is_same_v<T, bool>) return tandem_next_bool(&s_);
        else if constexpr (std::is_same_v<T, std::uint8_t>) return tandem_next_u8(&s_);
        else if constexpr (std::is_same_v<T, std::uint16_t>) return tandem_next_u16(&s_);
        else if constexpr (std::is_same_v<T, std::uint32_t>) return tandem_next_u32(&s_);
        else if constexpr (std::is_same_v<T, std::uint64_t>) return tandem_next_u64(&s_);
        else if constexpr (std::is_same_v<T, float>) return tandem_next_f32(&s_);
        else if constexpr (std::is_same_v<T, double>) return tandem_next_f64(&s_);
#ifdef __SIZEOF_INT128__
        else if constexpr (std::is_same_v<T, uint128>) {
            tandem_u128 v = tandem_next_u128(&s_);
            return (uint128{v.hi} << 64) | v.lo;
        }
#endif
        else if constexpr (std::is_same_v<T, float16_bits>) return {tandem_next_f16_bits(&s_)};
        else if constexpr (std::is_same_v<T, char32_t>) return static_cast<char32_t>(tandem_next_char(&s_));
        else if constexpr (std::is_same_v<T, std::complex<float>>) {
            float c[2];
            tandem_next_c32(&s_, c);
            return {c[0], c[1]};
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            double c[2];
            tandem_next_c64(&s_, c);
            return {c[0], c[1]};
        }
        else static_assert(sizeof(T) == 0, "tandem::rng::next: unsupported type");
    }

    /* Element i of the fill that would start here, without advancing: uint32_t, uint64_t,
     * float, or double. */
    template <class T> T at(std::uint64_t i) const noexcept {
        if constexpr (std::is_same_v<T, std::uint32_t>) return tandem_at_u32(&s_, i);
        else if constexpr (std::is_same_v<T, std::uint64_t>) return tandem_at_u64(&s_, i);
        else if constexpr (std::is_same_v<T, float>) return tandem_at_f32(&s_, i);
        else if constexpr (std::is_same_v<T, double>) return tandem_at_f64(&s_, i);
        else static_assert(sizeof(T) == 0, "tandem::rng::at: unsupported type");
    }

    /* Uniform on [0, n) for uint32_t or uint64_t, by Lemire's method. */
    template <class T> T below(T n) noexcept {
        if constexpr (std::is_same_v<T, std::uint32_t>) return tandem_u32_below(&s_, n);
        else if constexpr (std::is_same_v<T, std::uint64_t>) return tandem_u64_below(&s_, n);
        else static_assert(sizeof(T) == 0, "tandem::rng::below: unsupported type");
    }

    /* Both Box-Muller normals of one pair of uniforms, cos half first: tandem_normal2_f64 or f32. */
    template <class T = double> std::array<T, 2> normal2() noexcept {
        std::array<T, 2> z;
        if constexpr (std::is_same_v<T, double>) tandem_normal2_f64(&s_, z.data());
        else if constexpr (std::is_same_v<T, float>) tandem_normal2_f32(&s_, z.data());
        else static_assert(sizeof(T) == 0, "tandem::rng::normal2: unsupported type");
        return z;
    }

    /* Standard normal by Box-Muller, float or double: the values of tandem_normal_f64 and f32. */
    template <class T = double> T normal() noexcept {
        if constexpr (std::is_same_v<T, double>) return tandem_normal_f64(&s_);
        else if constexpr (std::is_same_v<T, float>) return tandem_normal_f32(&s_);
        else static_assert(sizeof(T) == 0, "tandem::rng::normal: unsupported type");
    }

    /* Standard exponential -ln(1 - u), float or double: tandem_exponential_f64 and f32. */
    template <class T = double> T exponential() noexcept {
        if constexpr (std::is_same_v<T, double>) return tandem_exponential_f64(&s_);
        else if constexpr (std::is_same_v<T, float>) return tandem_exponential_f32(&s_);
        else static_assert(sizeof(T) == 0, "tandem::rng::exponential: unsupported type");
    }

    /* Fill n values of T, the same values as n calls of next<T>(). */
    template <class T> void fill(T *out, std::size_t n) noexcept {
        if constexpr (std::is_same_v<T, bool>) tandem_fill_bool(&s_, out, n);
        else if constexpr (std::is_same_v<T, std::uint8_t>) tandem_fill_u8(&s_, out, n);
        else if constexpr (std::is_same_v<T, std::uint16_t>) tandem_fill_u16(&s_, out, n);
        else if constexpr (std::is_same_v<T, std::uint32_t>) tandem_fill_u32(&s_, out, n);
        else if constexpr (std::is_same_v<T, std::uint64_t>) tandem_fill_u64(&s_, out, n);
        else if constexpr (std::is_same_v<T, float>) tandem_fill_f32(&s_, out, n);
        else if constexpr (std::is_same_v<T, double>) tandem_fill_f64(&s_, out, n);
        else static_assert(sizeof(T) == 0, "tandem::rng::fill: unsupported type");
    }

    template <class T> std::vector<T> fill(std::size_t n) {
        std::vector<T> v(n);
        fill(v.data(), n);
        return v;
    }

    /* Derived generators: by index, by purpose, and a batch forked at the current block. */
    rng split(std::uint64_t index) const noexcept { return rng(tandem_split(&s_, index)); }
    rng sub(std::uint64_t purpose) const noexcept { return rng(tandem_sub(&s_, purpose)); }
    std::vector<rng> fork(std::uint64_t n) {
        std::vector<tandem_rng> raw(n);
        std::vector<rng> out;
        tandem_fork(&s_, raw.data(), n);
        out.reserve(n);
        for (const tandem_rng &r : raw) out.push_back(rng(r));
        return out;
    }

    /* The C state, for code that mixes the two APIs. */
    const tandem_rng &c_state() const noexcept { return s_; }
    tandem_rng &c_state() noexcept { return s_; }

    friend bool operator==(const rng &a, const rng &b) noexcept {
        return a.key() == b.key() && a.position() == b.position() &&
               a.chunk_length() == b.chunk_length();
    }
    friend bool operator!=(const rng &a, const rng &b) noexcept { return !(a == b); }

private:
    explicit rng(const tandem_rng &s) noexcept : s_(s) {}
    tandem_rng s_;
};

} // namespace tandem

#endif /* TANDEM_HPP */
