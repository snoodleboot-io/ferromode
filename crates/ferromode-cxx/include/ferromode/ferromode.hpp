#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declarations for cxx-generated bridge types
// ---------------------------------------------------------------------------

namespace rust {
inline namespace cxxbridge1 {
struct CImfResult;
struct CHilbertResult;
struct CEmdConfig;
struct CEnsembleConfig;
struct CMemdConfig;
struct CNaMemdConfig;
struct CVmdConfig;
} // namespace cxxbridge1
} // namespace rust

// ---------------------------------------------------------------------------
// ferromode:: namespace — C++17 header-only wrapper
// ---------------------------------------------------------------------------

namespace ferromode {

// ---------------------------------------------------------------------------
// Config structs (plain aggregates mirroring C structs)
// ---------------------------------------------------------------------------

/// Configuration for EMD decomposition.
struct EmdConfig {
    std::size_t max_imfs = 0;
    double sd_threshold = 0.2;
    std::size_t s_number = 5;
    std::size_t max_sifting_iterations = 100;
    int boundary_condition = 0; // 0=MirrorEven, 1=MirrorOdd, 2=Periodic, 3=Slope, 4=ARModel, 5=CharacteristicWave, 6=WaveformMatching
};

/// Configuration for ensemble algorithms (EEMD, CEEMD, CEEMDAN, ICEEMDAN).
struct EnsembleConfig {
    std::size_t num_ensembles = 100;
    double noise_std = 0.2;
    std::uint64_t seed = 0;
    bool use_seed = false;
};

/// Configuration for MEMD decomposition.
struct MemdConfig {
    std::size_t num_directions = 64;
    std::uint64_t direction_seed = 0;
    std::size_t max_imfs = 0;
    double sd_threshold = 0.2;
    std::size_t s_number = 5;
    std::size_t max_sifting_iterations = 100;
};

/// Configuration for NA-MEMD decomposition.
struct NaMemdConfig {
    MemdConfig base;
    std::size_t n_noise_channels = 2;
    double noise_std = 0.1;
    std::uint64_t seed = 0;
    bool use_seed = false;
};

/// Configuration for VMD decomposition.
struct VmdConfig {
    std::size_t n_modes = 3;
    double alpha = 2000.0;
    double tau = 0.0;
    double tol = 1e-7;
    std::size_t max_iterations = 500;
};

// ---------------------------------------------------------------------------
// RAII result wrappers
// ---------------------------------------------------------------------------

/// RAII wrapper for an IMF collection returned by decomposition algorithms.
///
/// Owns the underlying CImfResult and frees it on destruction.
class ImfCollection {
public:
    ImfCollection() = default;
    ~ImfCollection();

    // Move-only
    ImfCollection(ImfCollection&&) noexcept;
    ImfCollection& operator=(ImfCollection&&) noexcept;
    ImfCollection(const ImfCollection&) = delete;
    ImfCollection& operator=(const ImfCollection&) = delete;

    /// Number of IMFs extracted.
    [[nodiscard]] std::size_t n_imfs() const noexcept;

    /// Number of samples per IMF.
    [[nodiscard]] std::size_t n_samples() const noexcept;

    /// Return all IMFs as a vector of spans.
    [[nodiscard]] std::vector<std::vector<double>> imfs() const;

    /// Return the residue as a vector.
    [[nodiscard]] std::vector<double> residue() const;

    /// Reconstruct the original signal from IMFs + residue.
    [[nodiscard]] std::vector<double> reconstruct() const;

private:
    friend ImfCollection emd(std::span<const double>, const EmdConfig&);
    friend ImfCollection eemd(std::span<const double>, const EnsembleConfig&, const EmdConfig&);
    friend ImfCollection ceemd(std::span<const double>, const EnsembleConfig&, const EmdConfig&);
    friend ImfCollection ceemdan(std::span<const double>, const EnsembleConfig&, const EmdConfig&);
    friend ImfCollection iceemdan(std::span<const double>, const EnsembleConfig&, const EmdConfig&);
    friend ImfCollection memd(std::span<const double*>, std::size_t, const MemdConfig&);
    friend ImfCollection namemd(std::span<const double*>, std::size_t, const NaMemdConfig&);
    friend ImfCollection vmd(std::span<const double>, const VmdConfig&);

    explicit ImfCollection(rust::cxxbridge1::CImfResult* raw);

    rust::cxxbridge1::CImfResult* raw_ = nullptr;
};

/// RAII wrapper for Hilbert spectral analysis results.
class HilbertResult {
public:
    HilbertResult() = default;
    ~HilbertResult();

    // Move-only
    HilbertResult(HilbertResult&&) noexcept;
    HilbertResult& operator=(HilbertResult&&) noexcept;
    HilbertResult(const HilbertResult&) = delete;
    HilbertResult& operator=(const HilbertResult&) = delete;

    [[nodiscard]] std::size_t n_imfs() const noexcept;
    [[nodiscard]] std::size_t n_samples() const noexcept;
    [[nodiscard]] std::size_t n_freq_bins() const noexcept;

    [[nodiscard]] std::vector<std::vector<double>> instantaneous_amplitude() const;
    [[nodiscard]] std::vector<std::vector<double>> instantaneous_frequency() const;
    [[nodiscard]] std::vector<double> marginal_spectrum() const;

private:
    friend HilbertResult hilbert(const ImfCollection&, double sample_rate);

    explicit HilbertResult(rust::cxxbridge1::CHilbertResult* raw);

    rust::cxxbridge1::CHilbertResult* raw_ = nullptr;
};

// ---------------------------------------------------------------------------
// Exception type
// ---------------------------------------------------------------------------

/// Exception thrown when a Ferromode algorithm fails.
class FerromodeError : public std::runtime_error {
public:
    explicit FerromodeError(const std::string& msg) : std::runtime_error(msg) {}
};

// ---------------------------------------------------------------------------
// Free functions — 8 algorithms + Hilbert
// ---------------------------------------------------------------------------

/// Empirical Mode Decomposition.
ImfCollection emd(std::span<const double> signal, const EmdConfig& config);

/// Ensemble Empirical Mode Decomposition.
ImfCollection eemd(std::span<const double> signal, const EnsembleConfig& config,
                   const EmdConfig& emd_config = {});

/// Complementary Ensemble EMD.
ImfCollection ceemd(std::span<const double> signal, const EnsembleConfig& config,
                    const EmdConfig& emd_config = {});

/// Complete Ensemble EMD with Adaptive Noise.
ImfCollection ceemdan(std::span<const double> signal, const EnsembleConfig& config,
                      const EmdConfig& emd_config = {});

/// Improved Complete Ensemble EMD with Adaptive Noise.
ImfCollection iceemdan(std::span<const double> signal, const EnsembleConfig& config,
                       const EmdConfig& emd_config = {});

/// Multivariate EMD.
///
/// \param channels Array of pointers, each pointing to a channel's data.
/// \param n_channels Number of channels.
/// \param config MEMD configuration.
ImfCollection memd(std::span<const double*> channels, std::size_t n_channels,
                   const MemdConfig& config);

/// Noise-Assisted Multivariate EMD.
ImfCollection namemd(std::span<const double*> channels, std::size_t n_channels,
                     const NaMemdConfig& config);

/// Variational Mode Decomposition.
ImfCollection vmd(std::span<const double> signal, const VmdConfig& config);

/// Hilbert spectral analysis on an IMF collection.
///
/// \param imfs IMF collection from any decomposition algorithm.
/// \param sample_rate Sampling rate in Hz (default 1.0).
HilbertResult hilbert(const ImfCollection& imfs, double sample_rate = 1.0);

} // namespace ferromode
