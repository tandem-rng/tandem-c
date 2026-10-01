/* Tandem8x32 for C++17: a thin value type over the C implementation that satisfies
 * std::uniform_random_bit_generator, so it drives every <random> distribution.
 *
 * Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#ifndef TANDEM_HPP
#define TANDEM_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "tandem.h"

namespace tandem {

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
    std::uint32_t chunk_length() const noexcept { return tandem_chunk_length(&s_); }

    /* std::uniform_random_bit_generator: 64 bits per call. */
    result_type operator()() noexcept { return tandem_next_u64(&s_); }

    /* Skip z 64-bit draws. */
    void discard(unsigned long long z) noexcept {
        s_.pos = ((s_.pos + 63u) & ~std::uint64_t{63}) + 64u * z;
    }

    /* Draw one value of T: bool, the fixed-width unsigned integers, float, or double. */
    template <class T> T next() noexcept {
        if constexpr (std::is_same_v<T, bool>) return tandem_next_bool(&s_);
        else if constexpr (std::is_same_v<T, std::uint8_t>) return tandem_next_u8(&s_);
        else if constexpr (std::is_same_v<T, std::uint16_t>) return tandem_next_u16(&s_);
        else if constexpr (std::is_same_v<T, std::uint32_t>) return tandem_next_u32(&s_);
        else if constexpr (std::is_same_v<T, std::uint64_t>) return tandem_next_u64(&s_);
        else if constexpr (std::is_same_v<T, float>) return tandem_next_f32(&s_);
        else if constexpr (std::is_same_v<T, double>) return tandem_next_f64(&s_);
        else static_assert(sizeof(T) == 0, "tandem::rng::next: unsupported type");
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
