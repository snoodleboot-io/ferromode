module Ferromode

using Libdl

const libferromode = Ref{Ptr{Cvoid}}()

function __init__()
    lib_path = joinpath(@__DIR__, "..", "..", "..", "..", "target", "debug")
    lib_name = Sys.iswindows() ? "ferromode.dll" : (Sys.isapple() ? "libferromode.dylib" : "libferromode.so")
    full_path = joinpath(lib_path, lib_name)
    if isfile(full_path)
        libferromode[] = Libdl.dlopen(full_path)
    else
        libferromode[] = Libdl.find_library("ferromode", [lib_path])
    end
end

# ---------------------------------------------------------------------------
# Config structs (match C layout)
# ---------------------------------------------------------------------------

struct EmdConfig
    max_imfs::Csize_t
    sd_threshold::Cdouble
    s_number::Csize_t
    max_sifting_iterations::Csize_t
    boundary_condition::Cint
end

EmdConfig(;
    max_imfs::Integer=0,
    sd_threshold::Real=0.2,
    s_number::Integer=5,
    max_sifting_iterations::Integer=100,
    boundary_condition::Integer=0,
) = EmdConfig(
    Csize_t(max_imfs),
    Cdouble(sd_threshold),
    Csize_t(s_number),
    Csize_t(max_sifting_iterations),
    Cint(boundary_condition),
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
    base::MemdConfig=MemdConfig(),
    n_noise_channels::Integer=2,
    noise_std::Real=0.1,
    seed::Union{Integer,Nothing}=nothing,
) = NaMemdConfig(
    base,
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

struct DecompositionResult
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

function eemd(signal::AbstractVector{<:Real}; ensemble_kwargs..., emd_kwargs...)
    n = length(signal)
    ens_config = EnsembleConfig(; ensemble_kwargs...)
    emd_config = EmdConfig(; emd_kwargs...)
    ptr = ccall(
        (:ferromode_eemd, libferromode[]),
        Ptr{Cvoid},
        (Ptr{Cdouble}, Csize_t, Ref{EnsembleConfig}, Ref{EmdConfig}),
        signal, Csize_t(n), ens_config, emd_config,
    )
    return DecompositionResult(ptr)
end

function ceemd(signal::AbstractVector{<:Real}; ensemble_kwargs..., emd_kwargs...)
    n = length(signal)
    ens_config = EnsembleConfig(; ensemble_kwargs...)
    emd_config = EmdConfig(; emd_kwargs...)
    ptr = ccall(
        (:ferromode_ceemd, libferromode[]),
        Ptr{Cvoid},
        (Ptr{Cdouble}, Csize_t, Ref{EnsembleConfig}, Ref{EmdConfig}),
        signal, Csize_t(n), ens_config, emd_config,
    )
    return DecompositionResult(ptr)
end

function ceemdan(signal::AbstractVector{<:Real}; ensemble_kwargs..., emd_kwargs...)
    n = length(signal)
    ens_config = EnsembleConfig(; ensemble_kwargs...)
    emd_config = EmdConfig(; emd_kwargs...)
    ptr = ccall(
        (:ferromode_ceemdan, libferromode[]),
        Ptr{Cvoid},
        (Ptr{Cdouble}, Csize_t, Ref{EnsembleConfig}, Ref{EmdConfig}),
        signal, Csize_t(n), ens_config, emd_config,
    )
    return DecompositionResult(ptr)
end

function iceemdan(signal::AbstractVector{<:Real}; ensemble_kwargs..., emd_kwargs...)
    n = length(signal)
    ens_config = EnsembleConfig(; ensemble_kwargs...)
    emd_config = EmdConfig(; emd_kwargs...)
    ptr = ccall(
        (:ferromode_iceemdan, libferromode[]),
        Ptr{Cvoid},
        (Ptr{Cdouble}, Csize_t, Ref{EnsembleConfig}, Ref{EmdConfig}),
        signal, Csize_t(n), ens_config, emd_config,
    )
    return DecompositionResult(ptr)
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
# Version
# ---------------------------------------------------------------------------

function version()
    return "0.1.0"
end

end # module Ferromode
