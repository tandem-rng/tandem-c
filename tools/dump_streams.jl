# Write the reference fills that tests/test_stream.c compares against.
# Run from an environment with TandemRNG: julia tools/dump_streams.jl tests/data
using TandemRNG

const dir = get(ARGS, 1, "tests/data")
mkpath(dir)
const key = (UInt32(1), UInt32(2), UInt32(3), UInt32(4))

function dump(name, rng, ::Type{T}, n) where {T}
    A = Vector{T}(undef, n)
    rand_fill!(rng, A; nthreads = 1)
    open(io -> write(io, T === Bool ? UInt8.(A) : A), joinpath(dir, name), "w")
end

dump("k1234_K32_u32.bin", Tandem8x32(key), UInt32, 65536)
dump("k1234_K32_u64.bin", Tandem8x32(key), UInt64, 2048)
dump("k1234_K8_u32.bin", Tandem8x32{8}(key), UInt32, 16384)
dump("seed42_K32_f64.bin", Tandem8x32(42), Float64, 4096)
dump("seed42_K32_f32.bin", Tandem8x32(42), Float32, 4096)
dump("seed42_K32_u8.bin", Tandem8x32(42), UInt8, 8192)
dump("seed42_K32_bool.bin", Tandem8x32(42), Bool, 4096)
dump("seed42_K32_u128.bin", Tandem8x32(42), UInt128, 1024)
dump("seed42_K32_c64.bin", Tandem8x32(42), ComplexF64, 1024)
dump("seed42_K32_c32.bin", Tandem8x32(42), ComplexF32, 1024)
let A = Vector{Float16}(undef, 4096)
    rand_fill!(Tandem8x32(42), A; nthreads = 1)
    open(io -> write(io, reinterpret(UInt16, A)), joinpath(dir, "seed42_K32_f16bits.bin"), "w")
end
let A = Vector{Char}(undef, 4096)
    rand_fill!(Tandem8x32(42), A; nthreads = 1)
    open(io -> write(io, UInt32.(A)), joinpath(dir, "seed42_K32_char.bin"), "w")
end
