#pragma once

#include <span>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <cstring>

namespace ferromode {

class FerromodeError : public std::runtime_error {
public:
    explicit FerromodeError(const std::string& msg)
        : std::runtime_error(msg) {}
};

struct EmdConfig {
    size_t max_imfs = 0;
    double sd_threshold = 0.2;
    size_t s_number = 5;
    size_t max_sifting_iterations = 100;
    int32_t boundary_condition = 0;
};

struct EnsembleConfig {
    size_t num_ensembles = 10;
    double noise_std = 0.2;
    uint64_t seed = 0;
    bool use_seed = false;
};

struct MemdConfig {
    size_t num_directions = 64;
    uint64_t direction_seed = 0;
    size_t max_imfs = 0;
    double sd_threshold = 0.2;
    size_t s_number = 5;
    size_t max_sifting_iterations = 100;
};

struct NaMemdConfig {
    MemdConfig base{};
    size_t n_noise_channels = 1;
    double noise_std = 0.1;
    uint64_t seed = 0;
    bool use_seed = false;
};

struct VmdConfig {
    size_t n_modes = 3;
    double alpha = 2000.0;
    double tau = 0.0;
    double tol = 1e-7;
    size_t max_iterations = 500;
};

struct CImfResult {
    const double* imfs_data;
    size_t n_imfs;
    size_t n_samples;
    const double* residue;
};

class ImfCollection {
public:
    ImfCollection() = default;

    ~ImfCollection() {
        free_data();
    }

    ImfCollection(ImfCollection&& other) noexcept
        : imfs_data_(other.imfs_data_),
          n_imfs_(other.n_imfs_),
          n_samples_(other.n_samples_),
          residue_(std::move(other.residue_)),
          owns_data_(other.owns_data_) {
        other.imfs_data_ = nullptr;
        other.n_imfs_ = 0;
        other.n_samples_ = 0;
        other.owns_data_ = false;
    }

    ImfCollection& operator=(ImfCollection&& other) noexcept {
        if (this != &other) {
            free_data();
            imfs_data_ = other.imfs_data_;
            n_imfs_ = other.n_imfs_;
            n_samples_ = other.n_samples_;
            residue_ = std::move(other.residue_);
            owns_data_ = other.owns_data_;
            other.imfs_data_ = nullptr;
            other.n_imfs_ = 0;
            other.n_samples_ = 0;
            other.owns_data_ = false;
        }
        return *this;
    }

    ImfCollection(const ImfCollection&) = delete;
    ImfCollection& operator=(const ImfCollection&) = delete;

    void free_data() {
        if (owns_data_ && imfs_data_ != nullptr) {
            delete[] imfs_data_;
            imfs_data_ = nullptr;
        }
        residue_.clear();
        owns_data_ = false;
    }

    void take_ownership(CImfResult result) {
        free_data();
        n_imfs_ = result.n_imfs;
        n_samples_ = result.n_samples;

        if (result.n_imfs > 0 && result.n_samples > 0) {
            size_t total = result.n_imfs * result.n_samples;
            imfs_data_ = new double[total];
            std::memcpy(imfs_data_, result.imfs_data, total * sizeof(double));
        }

        if (result.n_samples > 0 && result.residue != nullptr) {
            residue_.assign(result.residue, result.residue + result.n_samples);
        }

        owns_data_ = true;
    }

    size_t n_imfs() const { return n_imfs_; }
    size_t n_samples() const { return n_samples_; }

    std::vector<std::span<const double>> imfs() const {
        std::vector<std::span<const double>> result;
        result.reserve(n_imfs_);
        for (size_t i = 0; i < n_imfs_; ++i) {
            result.emplace_back(imfs_data_ + i * n_samples_, n_samples_);
        }
        return result;
    }

    std::span<const double> residue() const {
        return std::span<const double>(residue_.data(), residue_.size());
    }

    std::vector<double> reconstruct() const {
        std::vector<double> result(n_samples_, 0.0);

        for (size_t i = 0; i < n_imfs_; ++i) {
            const double* imf = imfs_data_ + i * n_samples_;
            for (size_t j = 0; j < n_samples_; ++j) {
                result[j] += imf[j];
            }
        }

        for (size_t j = 0; j < residue_.size() && j < n_samples_; ++j) {
            result[j] += residue_[j];
        }

        return result;
    }

private:
    const double* imfs_data_ = nullptr;
    size_t n_imfs_ = 0;
    size_t n_samples_ = 0;
    std::vector<double> residue_;
    bool owns_data_ = false;
};

struct CHilbertResult {
    const double* instantaneous_amplitude;
    const double* instantaneous_frequency;
    const double* marginal_spectrum;
    size_t n_imfs;
    size_t n_samples;
    size_t n_freq_bins;
};

class HilbertResult {
public:
    HilbertResult() = default;

    ~HilbertResult() {
        free_data();
    }

    HilbertResult(HilbertResult&& other) noexcept
        : amplitude_(std::move(other.amplitude_)),
          frequency_(std::move(other.frequency_)),
          marginal_spectrum_(std::move(other.marginal_spectrum_)),
          n_imfs_(other.n_imfs_),
          n_samples_(other.n_samples_),
          n_freq_bins_(other.n_freq_bins_) {
        other.n_imfs_ = 0;
        other.n_samples_ = 0;
        other.n_freq_bins_ = 0;
    }

    HilbertResult& operator=(HilbertResult&& other) noexcept {
        if (this != &other) {
            free_data();
            amplitude_ = std::move(other.amplitude_);
            frequency_ = std::move(other.frequency_);
            marginal_spectrum_ = std::move(other.marginal_spectrum_);
            n_imfs_ = other.n_imfs_;
            n_samples_ = other.n_samples_;
            n_freq_bins_ = other.n_freq_bins_;
            other.n_imfs_ = 0;
            other.n_samples_ = 0;
            other.n_freq_bins_ = 0;
        }
        return *this;
    }

    HilbertResult(const HilbertResult&) = delete;
    HilbertResult& operator=(const HilbertResult&) = delete;

    void free_data() {
        amplitude_.clear();
        frequency_.clear();
        marginal_spectrum_.clear();
    }

    void take_ownership(CHilbertResult result) {
        free_data();
        n_imfs_ = result.n_imfs;
        n_samples_ = result.n_samples;
        n_freq_bins_ = result.n_freq_bins;

        if (n_imfs_ > 0 && n_samples_ > 0) {
            size_t total = n_imfs_ * n_samples_;
            amplitude_.assign(result.instantaneous_amplitude,
                              result.instantaneous_amplitude + total);
            frequency_.assign(result.instantaneous_frequency,
                              result.instantaneous_frequency + total);
        }

        if (n_freq_bins_ > 0) {
            marginal_spectrum_.assign(result.marginal_spectrum,
                                      result.marginal_spectrum + n_freq_bins_);
        }
    }

    size_t n_imfs() const { return n_imfs_; }
    size_t n_samples() const { return n_samples_; }
    size_t n_freq_bins() const { return n_freq_bins_; }

    std::vector<std::span<const double>> instantaneous_amplitude() const {
        std::vector<std::span<const double>> result;
        result.reserve(n_imfs_);
        for (size_t i = 0; i < n_imfs_; ++i) {
            result.emplace_back(amplitude_.data() + i * n_samples_, n_samples_);
        }
        return result;
    }

    std::vector<std::span<const double>> instantaneous_frequency() const {
        std::vector<std::span<const double>> result;
        result.reserve(n_imfs_);
        for (size_t i = 0; i < n_imfs_; ++i) {
            result.emplace_back(frequency_.data() + i * n_samples_, n_samples_);
        }
        return result;
    }

    std::span<const double> marginal_spectrum() const {
        return std::span<const double>(marginal_spectrum_.data(),
                                       marginal_spectrum_.size());
    }

private:
    std::vector<double> amplitude_;
    std::vector<double> frequency_;
    std::vector<double> marginal_spectrum_;
    size_t n_imfs_ = 0;
    size_t n_samples_ = 0;
    size_t n_freq_bins_ = 0;
};

extern "C" {
    CImfResult ferromode_emd_cxx(const double* signal, size_t n, const EmdConfig* config);
    CImfResult ferromode_eemd_cxx(const double* signal, size_t n, const EnsembleConfig* config, const EmdConfig* emd_config);
    CImfResult ferromode_ceemd_cxx(const double* signal, size_t n, const EnsembleConfig* config, const EmdConfig* emd_config);
    CImfResult ferromode_ceemdan_cxx(const double* signal, size_t n, const EnsembleConfig* config, const EmdConfig* emd_config);
    CImfResult ferromode_iceemdan_cxx(const double* signal, size_t n, const EnsembleConfig* config, const EmdConfig* emd_config);
    CImfResult ferromode_memd_cxx(const double* channels, size_t n_channels, size_t n_samples, const MemdConfig* config);
    CImfResult ferromode_namemd_cxx(const double* channels, size_t n_channels, size_t n_samples, const NaMemdConfig* config);
    CImfResult ferromode_vmd_cxx(const double* signal, size_t n, const VmdConfig* config);
    CHilbertResult ferromode_hilbert_cxx(const double* imfs, size_t n_imfs, size_t n_samples, double sample_rate);
}

inline ImfCollection emd(std::span<const double> signal, const EmdConfig& config) {
    CImfResult result = ferromode_emd_cxx(signal.data(), signal.size(), &config);
    ImfCollection collection;
    collection.take_ownership(result);
    return collection;
}

inline ImfCollection eemd(std::span<const double> signal, const EnsembleConfig& config, const EmdConfig& emd_config = EmdConfig{}) {
    CImfResult result = ferromode_eemd_cxx(signal.data(), signal.size(), &config, &emd_config);
    ImfCollection collection;
    collection.take_ownership(result);
    return collection;
}

inline ImfCollection ceemd(std::span<const double> signal, const EnsembleConfig& config, const EmdConfig& emd_config = EmdConfig{}) {
    CImfResult result = ferromode_ceemd_cxx(signal.data(), signal.size(), &config, &emd_config);
    ImfCollection collection;
    collection.take_ownership(result);
    return collection;
}

inline ImfCollection ceemdan(std::span<const double> signal, const EnsembleConfig& config, const EmdConfig& emd_config = EmdConfig{}) {
    CImfResult result = ferromode_ceemdan_cxx(signal.data(), signal.size(), &config, &emd_config);
    ImfCollection collection;
    collection.take_ownership(result);
    return collection;
}

inline ImfCollection iceemdan(std::span<const double> signal, const EnsembleConfig& config, const EmdConfig& emd_config = EmdConfig{}) {
    CImfResult result = ferromode_iceemdan_cxx(signal.data(), signal.size(), &config, &emd_config);
    ImfCollection collection;
    collection.take_ownership(result);
    return collection;
}

inline ImfCollection memd(std::span<const double*> channels, size_t n_channels, size_t n_samples, const MemdConfig& config) {
    std::vector<double> flat_data;
    flat_data.reserve(n_channels * n_samples);
    for (size_t i = 0; i < n_channels; ++i) {
        flat_data.insert(flat_data.end(), channels[i], channels[i] + n_samples);
    }
    CImfResult result = ferromode_memd_cxx(flat_data.data(), n_channels, n_samples, &config);
    ImfCollection collection;
    collection.take_ownership(result);
    return collection;
}

inline ImfCollection namemd(std::span<const double*> channels, size_t n_channels, size_t n_samples, const NaMemdConfig& config) {
    std::vector<double> flat_data;
    flat_data.reserve(n_channels * n_samples);
    for (size_t i = 0; i < n_channels; ++i) {
        flat_data.insert(flat_data.end(), channels[i], channels[i] + n_samples);
    }
    CImfResult result = ferromode_namemd_cxx(flat_data.data(), n_channels, n_samples, &config);
    ImfCollection collection;
    collection.take_ownership(result);
    return collection;
}

inline ImfCollection vmd(std::span<const double> signal, const VmdConfig& config) {
    CImfResult result = ferromode_vmd_cxx(signal.data(), signal.size(), &config);
    ImfCollection collection;
    collection.take_ownership(result);
    return collection;
}

inline HilbertResult hilbert(const ImfCollection& imfs, double sample_rate = 1.0) {
    std::vector<double> flat_imfs;
    flat_imfs.reserve(imfs.n_imfs() * imfs.n_samples());
    auto imf_spans = imfs.imfs();
    for (const auto& imf : imf_spans) {
        flat_imfs.insert(flat_imfs.end(), imf.begin(), imf.end());
    }
    CHilbertResult result = ferromode_hilbert_cxx(flat_imfs.data(), imfs.n_imfs(), imfs.n_samples(), sample_rate);
    HilbertResult hilbert_result;
    hilbert_result.take_ownership(result);
    return hilbert_result;
}

} // namespace ferromode
