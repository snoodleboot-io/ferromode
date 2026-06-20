module Ferromode

using Libdl

# Holds the resolved library path (ccall's (:sym, lib) form takes a path string).
const libferromode = Ref{String}()

function __init__()
    lib_name = Sys.iswindows() ? "ferromode.dll" :
               (Sys.isapple() ? "libferromode.dylib" : "libferromode.so")
    candidates = String[]
    envpath = get(ENV, "FERROMODE_LIB", "")
    isempty(envpath) || push!(candidates, envpath)
    # Package lives at crates/ferromode-julia/julia/Ferromode.jl/src; the cargo
    # target dir is at the workspace root (five levels up).
    root = normpath(joinpath(@__DIR__, "..", "..", "..", "..", ".."))
    for profile in ("debug", "release")
        push!(candidates, joinpath(root, "target", profile, lib_name))
    end
    for c in candidates
        if isfile(c)
            libferromode[] = c
            return
        end
    end
    # Fall back to letting the system loader resolve the bare name.
    libferromode[] = lib_name
end

# ---------------------------------------------------------------------------
# Config structs (match C layout)
# ---------------------------------------------------------------------------

# Field order/types MUST match crates/ferromode/src/ffi.rs::CEmdConfig exactly.
struct EmdConfig
    max_imfs::Csize_t
    sd_threshold::Cdouble
    s_number::Csize_t
    max_sifting_iterations::Csize_t
    boundary_condition::Cint
    spline_type::Cint
    fixed_iterations::Clonglong
    energy_threshold::Cdouble
    reconstruction_tolerance::Cdouble
    validate_reconstruction::Cint
    intermittency_cv::Cdouble
    intermittency_min_intervals::Csize_t
end

EmdConfig(;
    max_imfs::Integer=0,
    sd_threshold::Real=0.2,
    s_number::Integer=5,
    max_sifting_iterations::Integer=100,
    boundary_condition::Integer=0,
    spline_type::Integer=0,
    fixed_iterations::Integer=-1,
    energy_threshold::Real=1e-6,
    reconstruction_tolerance::Real=1e-12,
    validate_reconstruction::Integer=0,
    intermittency_cv::Real=-1.0,
    intermittency_min_intervals::Integer=0,
) = EmdConfig(
    Csize_t(max_imfs),
    Cdouble(sd_threshold),
    Csize_t(s_number),
    Csize_t(max_sifting_iterations),
    Cint(boundary_condition),
    Cint(spline_type),
    Clonglong(fixed_iterations),
    Cdouble(energy_threshold),
    Cdouble(reconstruction_tolerance),
    Cint(validate_reconstruction),
    Cdouble(intermittency_cv),
    Csize_t(intermittency_min_intervals),
)

struct EnsembleConfig
    num_ensembles::Csize_t
    noise_std::Cdouble
    seed::Culonglong
    use_seed::Cint
end

EnsembleConfig(;
    num_ensembles::Integer=100,
    noise_std::Real=0.2,
    seed::Union{Integer,Nothing}=nothing,
) = EnsembleConfig(
    Csize_t(num_ensembles),
    Cdouble(noise_std),
    Culonglong(something(seed, 0)),
    Cint(seed !== nothing ? 1 : 0),
)

struct VmdConfig
    n_modes::Csize_t
    alpha::Cdouble
    tau::Cdouble
    tol::Cdouble
    max_iterations::Csize_t
end

VmdConfig(;
    n_modes::Integer=3,
    alpha::Real=2000.0,
    tau::Real=0.0,
    tol::Real=1e-7,
    max_iterations::Integer=500,
) = VmdConfig(
    Csize_t(n_modes),
    Cdouble(alpha),
    Cdouble(tau),
    Cdouble(tol),
    Csize_t(max_iterations),
)

struct MemdConfig
    num_directions::Csize_t
    direction_seed::Culonglong
    max_imfs::Csize_t
    sd_threshold::Cdouble
    s_number::Csize_t
    max_sifting_iterations::Csize_t
end

MemdConfig(;
    num_directions::Integer=16,
    direction_seed::Integer=42,
    max_imfs::Integer=0,
    sd_threshold::Real=0.2,
    s_number::Integer=3,
    max_sifting_iterations::Integer=50,
) = MemdConfig(
    Csize_t(num_directions),
    Culonglong(direction_seed),
    Csize_t(max_imfs),
    Cdouble(sd_threshold),
    Csize_t(s_number),
    Csize_t(max_sifting_iterations),
)

struct NaMemdConfig
    base::MemdConfig
    n_noise_channels::Csize_t
    noise_std::Cdouble
    seed::Culonglong
    use_seed::Cint
end

NaMemdConfig(;
    num_directions::Integer=16,
    direction_seed::Integer=42,
    max_imfs::Integer=0,
    sd_threshold::Real=0.2,
    s_number::Integer=3,
    max_sifting_iterations::Integer=50,
    n_noise_channels::Integer=2,
    noise_std::Real=0.1,
    seed::Union{Integer,Nothing}=nothing,
) = NaMemdConfig(
    MemdConfig(;
        num_directions, direction_seed, max_imfs, sd_threshold, s_number,
        max_sifting_iterations,
    ),
    Csize_t(n_noise_channels),
    Cdouble(noise_std),
    Culonglong(something(seed, 0)),
    Cint(seed !== nothing ? 1 : 0),
)

# ---------------------------------------------------------------------------
# Result wrapper with finalizer
# ---------------------------------------------------------------------------

struct ImfCollection
    ptr::Ptr{Cvoid}
    n_imfs::Csize_t
    n_samples::Csize_t
end

# mutable so a finalizer can be attached (frees the Rust-side result).
mutable struct DecompositionResult
    ptr::Ptr{Cvoid}
    algorithm::Cint
    elapsed_ms::Cdouble
    n_siftings::Csize_t
    n_imfs::Csize_t
    n_samples::Csize_t
end

function DecompositionResult(ptr::Ptr{Cvoid})
    if ptr == C_NULL
        error("null result pointer")
    end
    algo = ccall((:ferromode_result_algorithm, libferromode[]), Cint, (Ptr{Cvoid},), ptr)
    elapsed = ccall((:ferromode_result_elapsed_ms, libferromode[]), Cdouble, (Ptr{Cvoid},), ptr)
    n_siftings = ccall((:ferromode_result_n_siftings, libferromode[]), Csize_t, (Ptr{Cvoid},), ptr)
    n_imfs = ccall((:ferromode_result_n_imfs, libferromode[]), Csize_t, (Ptr{Cvoid},), ptr)
    n_samples = ccall((:ferromode_result_n_samples, libferromode[]), Csize_t, (Ptr{Cvoid},), ptr)

    result = DecompositionResult(ptr, algo, elapsed, n_siftings, n_imfs, n_samples)
    finalizer(ferromode_free_result, result)
    return result
end

function ferromode_free_result(result::DecompositionResult)
    if result.ptr != C_NULL
        ccall((:ferromode_free_result, libferromode[]), Cvoid, (Ptr{Cvoid},), result.ptr)
    end
end

function get_imf(result::DecompositionResult, imf_index::Integer)
    if imf_index < 0 || imf_index >= result.n_imfs
        throw(BoundsError(result, imf_index))
    end
    buf = Vector{Cdouble}(undef, result.n_samples)
    ret = ccall(
        (:ferromode_copy_imf, libferromode[]),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ptr{Cdouble}, Csize_t),
        result.ptr, Csize_t(imf_index), buf, Csize_t(result.n_samples),
    )
    if ret != 0
        error("failed to copy IMF $imf_index")
    end
    return Vector{Float64}(buf)
end

function get_residue(result::DecompositionResult)
    buf = Vector{Cdouble}(undef, result.n_samples)
    ret = ccall(
        (:ferromode_copy_residue, libferromode[]),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t),
        result.ptr, buf, Csize_t(result.n_samples),
    )
    if ret != 0
        error("failed to copy residue")
    end
    return Vector{Float64}(buf)
end

function reconstruct(result::DecompositionResult)
    buf = Vector{Cdouble}(undef, result.n_samples)
    ret = ccall(
        (:ferromode_reconstruct, libferromode[]),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t),
        result.ptr, buf, Csize_t(result.n_samples),
    )
    if ret != 0
        error("failed to reconstruct signal")
    end
    return Vector{Float64}(buf)
end

const ALGORITHM_NAMES = Dict(
    0 => "emd",
    1 => "eemd",
    2 => "ceemd",
    3 => "ceemdan",
    4 => "iceemdan",
    5 => "memd",
    6 => "namemd",
    7 => "vmd",
)

algorithm_name(result::DecompositionResult) = get(ALGORITHM_NAMES, result.algorithm, "unknown")

# ---------------------------------------------------------------------------
# Algorithm wrappers
# ---------------------------------------------------------------------------

function emd(signal::AbstractVector{<:Real}; kwargs...)
    n = length(signal)
    config = EmdConfig(; kwargs...)
    ptr = ccall(
        (:ferromode_emd, libferromode[]),
        Ptr{Cvoid},
        (Ptr{Cdouble}, Csize_t, Ref{EmdConfig}),
        signal, Csize_t(n), config,
    )
    return DecompositionResult(ptr)
end

# Ensemble methods take a flat kwarg set; split into ensemble vs EMD configs.
const _ENSEMBLE_KEYS = (:num_ensembles, :noise_std, :seed)

function _split_ensemble_kwargs(kwargs)
    ens = Dict{Symbol,Any}()
    emdk = Dict{Symbol,Any}()
    for (k, v) in kwargs
        if k in _ENSEMBLE_KEYS
            ens[k] = v
        else
            emdk[k] = v
        end
    end
    return EnsembleConfig(; ens...), EmdConfig(; emdk...)
end

for (jl_name, c_sym) in (
    (:eemd, :ferromode_eemd),
    (:ceemd, :ferromode_ceemd),
    (:ceemdan, :ferromode_ceemdan),
    (:iceemdan, :ferromode_iceemdan),
)
    @eval function $jl_name(signal::AbstractVector{<:Real}; kwargs...)
        n = length(signal)
        ens_config, emd_config = _split_ensemble_kwargs(kwargs)
        ptr = ccall(
            ($(QuoteNode(c_sym)), libferromode[]),
            Ptr{Cvoid},
            (Ptr{Cdouble}, Csize_t, Ref{EnsembleConfig}, Ref{EmdConfig}),
            signal, Csize_t(n), ens_config, emd_config,
        )
        return DecompositionResult(ptr)
    end
end

function vmd(signal::AbstractVector{<:Real}; kwargs...)
    n = length(signal)
    config = VmdConfig(; kwargs...)
    ptr = ccall(
        (:ferromode_vmd, libferromode[]),
        Ptr{Cvoid},
        (Ptr{Cdouble}, Csize_t, Ref{VmdConfig}),
        signal, Csize_t(n), config,
    )
    return DecompositionResult(ptr)
end

function memd(signal::AbstractMatrix{<:Real}; kwargs...)
    n_samples, n_channels = size(signal)
    flat_signal = vec(signal')  # Row-major for C
    config = MemdConfig(; kwargs...)
    ptr = ccall(
        (:ferromode_memd, libferromode[]),
        Ptr{Cvoid},
        (Ptr{Cdouble}, Csize_t, Csize_t, Ref{MemdConfig}),
        flat_signal, Csize_t(n_channels), Csize_t(n_samples), config,
    )
    return DecompositionResult(ptr)
end

function namemd(signal::AbstractMatrix{<:Real}; kwargs...)
    n_samples, n_channels = size(signal)
    flat_signal = vec(signal')  # Row-major for C
    config = NaMemdConfig(; kwargs...)
    ptr = ccall(
        (:ferromode_namemd, libferromode[]),
        Ptr{Cvoid},
        (Ptr{Cdouble}, Csize_t, Csize_t, Ref{NaMemdConfig}),
        flat_signal, Csize_t(n_channels), Csize_t(n_samples), config,
    )
    return DecompositionResult(ptr)
end

# ---------------------------------------------------------------------------
# Hilbert spectral analysis
# ---------------------------------------------------------------------------

# Matches crates/ferromode/src/ffi.rs::CHilbertResult.
struct CHilbertResult
    instantaneous_amplitude::Ptr{Cdouble}
    instantaneous_frequency::Ptr{Cdouble}
    marginal_spectrum::Ptr{Cdouble}
    n_imfs::Csize_t
    n_samples::Csize_t
    n_freq_bins::Csize_t
    error::Ptr{Cint}
    error_msg::Ptr{Cchar}
end

struct HilbertResult
    instantaneous_amplitude::Matrix{Float64}  # n_imfs × n_samples
    instantaneous_frequency::Matrix{Float64}
    marginal_spectrum::Vector{Float64}
end

function hilbert(imfs::AbstractVector{<:AbstractVector{<:Real}}, sample_rate::Real=1.0)
    n_imfs = length(imfs)
    n_imfs == 0 && error("hilbert: need at least one IMF")
    n_samples = length(imfs[1])
    flat = Vector{Cdouble}(undef, n_imfs * n_samples)
    for i in 1:n_imfs, j in 1:n_samples
        flat[(i - 1) * n_samples + j] = imfs[i][j]
    end
    ptr = ccall(
        (:ferromode_hilbert, libferromode[]),
        Ptr{CHilbertResult},
        (Ptr{Cdouble}, Csize_t, Csize_t, Cdouble),
        flat, Csize_t(n_imfs), Csize_t(n_samples), Cdouble(sample_rate),
    )
    ptr == C_NULL && error("hilbert returned null")
    r = unsafe_load(ptr)
    if r.error != C_NULL
        msg = r.error_msg != C_NULL ? unsafe_string(r.error_msg) : "hilbert failed"
        ccall((:ferromode_free_hilbert, libferromode[]), Cvoid, (Ptr{CHilbertResult},), ptr)
        error(msg)
    end
    nm = Int(r.n_imfs) * Int(r.n_samples)
    amp = copy(unsafe_wrap(Array, r.instantaneous_amplitude, nm))
    freq = copy(unsafe_wrap(Array, r.instantaneous_frequency, nm))
    marg = copy(unsafe_wrap(Array, r.marginal_spectrum, Int(r.n_freq_bins)))
    ccall((:ferromode_free_hilbert, libferromode[]), Cvoid, (Ptr{CHilbertResult},), ptr)
    A = permutedims(reshape(amp, Int(r.n_samples), Int(r.n_imfs)))
    F = permutedims(reshape(freq, Int(r.n_samples), Int(r.n_imfs)))
    return HilbertResult(A, F, marg)
end

# ---------------------------------------------------------------------------
# Streaming decomposition
# ---------------------------------------------------------------------------

# Matches crates/ferromode/src/ffi.rs::CChunkResult.
struct CChunkResult
    imfs::Ptr{Ptr{Cdouble}}
    n_imfs::Csize_t
    n_samples::Csize_t
    residue::Ptr{Cdouble}
    spectral_entropy::Cdouble
    stationarity_score::Cdouble
    extrema_spacing_cv::Cdouble
    error::Ptr{Cint}
    error_msg::Ptr{Cchar}
end

mutable struct StreamingDecomposer
    handle::Ptr{Cvoid}
end

function StreamingDecomposer(; buffer_size::Integer=4096, ar_order::Integer=3, kwargs...)
    config = EmdConfig(; kwargs...)
    handle = ccall(
        (:ferromode_streaming_new, libferromode[]),
        Ptr{Cvoid},
        (Ref{EmdConfig}, Csize_t, Csize_t),
        config, Csize_t(buffer_size), Csize_t(ar_order),
    )
    handle == C_NULL && error("failed to create streaming decomposer")
    dec = StreamingDecomposer(handle)
    finalizer(dec) do d
        if d.handle != C_NULL
            ccall((:ferromode_streaming_free, libferromode[]), Cvoid, (Ptr{Cvoid},), d.handle)
            d.handle = C_NULL
        end
    end
    return dec
end

function decompose_chunk(dec::StreamingDecomposer, chunk::AbstractVector{<:Real})
    cdata = Vector{Cdouble}(chunk)
    ptr = ccall(
        (:ferromode_streaming_decompose_chunk, libferromode[]),
        Ptr{CChunkResult},
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t),
        dec.handle, cdata, Csize_t(length(cdata)),
    )
    ptr == C_NULL && error("decompose_chunk returned null")
    r = unsafe_load(ptr)
    if r.error != C_NULL
        msg = r.error_msg != C_NULL ? unsafe_string(r.error_msg) : "decompose_chunk failed"
        ccall((:ferromode_free_chunk_result, libferromode[]), Cvoid, (Ptr{CChunkResult},), ptr)
        error(msg)
    end
    n_imfs = Int(r.n_imfs)
    n_samples = Int(r.n_samples)
    imf_ptrs = unsafe_wrap(Array, r.imfs, n_imfs)
    imfs = [copy(unsafe_wrap(Array, imf_ptrs[i], n_samples)) for i in 1:n_imfs]
    residue = r.residue != C_NULL ? copy(unsafe_wrap(Array, r.residue, n_samples)) : Float64[]
    metrics = (
        spectral_entropy=Float64(r.spectral_entropy),
        stationarity_score=Float64(r.stationarity_score),
        extrema_spacing_cv=Float64(r.extrema_spacing_cv),
    )
    ccall((:ferromode_free_chunk_result, libferromode[]), Cvoid, (Ptr{CChunkResult},), ptr)
    return (imfs=imfs, residue=residue, metrics=metrics)
end

function reset!(dec::StreamingDecomposer)
    dec.handle != C_NULL &&
        ccall((:ferromode_streaming_reset, libferromode[]), Cvoid, (Ptr{Cvoid},), dec.handle)
    return nothing
end

# ---------------------------------------------------------------------------
# Differentiable EMD
# ---------------------------------------------------------------------------

mutable struct EmdForwardContext
    handle::Ptr{Cvoid}
    n_imfs::Int
    n_samples::Int
end

function emd_forward(signal::AbstractVector{<:Real}; kwargs...)
    cdata = Vector{Cdouble}(signal)
    config = EmdConfig(; kwargs...)
    handle = ccall(
        (:ferromode_diff_forward, libferromode[]),
        Ptr{Cvoid},
        (Ptr{Cdouble}, Csize_t, Ref{EmdConfig}),
        cdata, Csize_t(length(cdata)), config,
    )
    handle == C_NULL && error("emd_forward failed")
    n_imfs = Int(ccall((:ferromode_diff_n_imfs, libferromode[]), Csize_t, (Ptr{Cvoid},), handle))
    n_samples = Int(ccall((:ferromode_diff_n_samples, libferromode[]), Csize_t, (Ptr{Cvoid},), handle))
    ctx = EmdForwardContext(handle, n_imfs, n_samples)
    finalizer(ctx) do c
        if c.handle != C_NULL
            ccall((:ferromode_diff_free, libferromode[]), Cvoid, (Ptr{Cvoid},), c.handle)
            c.handle = C_NULL
        end
    end
    return ctx
end

function get_imf(ctx::EmdForwardContext, index::Integer)
    (index < 0 || index >= ctx.n_imfs) && throw(BoundsError(ctx, index))
    p = ccall(
        (:ferromode_diff_imf_ptr, libferromode[]),
        Ptr{Cdouble},
        (Ptr{Cvoid}, Csize_t),
        ctx.handle, Csize_t(index),
    )
    return copy(unsafe_wrap(Array, p, ctx.n_samples))
end

function reconstruction_error(ctx::EmdForwardContext)
    return Float64(ccall(
        (:ferromode_diff_reconstruction_error, libferromode[]),
        Cdouble,
        (Ptr{Cvoid},),
        ctx.handle,
    ))
end

function emd_backward(
    grad_imfs::AbstractVector{<:AbstractVector{<:Real}},
    signal::AbstractVector{<:Real},
)
    n = length(signal)
    n_imfs = length(grad_imfs)
    (n == 0 || n_imfs == 0) && error("emd_backward: empty input")
    flat = Vector{Cdouble}(undef, n_imfs * n)
    for i in 1:n_imfs, j in 1:n
        flat[(i - 1) * n + j] = grad_imfs[i][j]
    end
    out = Vector{Cdouble}(undef, n)
    rc = ccall(
        (:ferromode_diff_backward, libferromode[]),
        Cint,
        (Ptr{Cdouble}, Csize_t, Csize_t, Ptr{Cdouble}),
        flat, Csize_t(n_imfs), Csize_t(n), out,
    )
    rc != 0 && error("emd_backward failed")
    return Vector{Float64}(out)
end

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

function version()
    return "0.1.0"
end

export EmdConfig, EnsembleConfig, VmdConfig, MemdConfig, NaMemdConfig
export DecompositionResult, HilbertResult, StreamingDecomposer, EmdForwardContext
export emd, eemd, ceemd, ceemdan, iceemdan, memd, namemd, vmd
export hilbert, reconstruct, get_imf, get_residue, algorithm_name
export decompose_chunk, reset!, emd_forward, emd_backward, reconstruction_error, version

end # module Ferromode
