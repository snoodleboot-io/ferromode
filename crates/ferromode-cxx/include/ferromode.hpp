#pragma once

// C++ convenience layer over the Ferromode C ABI (crates/ferromode/src/ffi.rs,
// re-exported by the ferromode-cxx cdylib). The structs below must match the
// `#[repr(C)]` layouts in ffi.rs exactly. RAII wrappers own the opaque Rust
// handles and free them via the C ABI.

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ferromode {

class FerromodeError : public std::runtime_error {
public:
    explicit FerromodeError(const std::string& msg) : std::runtime_error(msg) {}
};

// --- C ABI config structs (must match ffi.rs repr(C) layouts) ---------------

enum class Boundary : int32_t {
    MirrorEven = 0, MirrorOdd = 1, Periodic = 2, Slope = 3, ARModel = 4,
    CharacteristicWave = 5, WaveformMatching = 6, PalindromeCyclic = 7
};
enum class Spline : int32_t { Natural = 0, Periodic = 1, NotAKnot = 2 };

struct EmdConfig {
    size_t max_imfs = 0;
    double sd_threshold = 0.2;
    size_t s_number = 5;
    size_t max_sifting_iterations = 100;
    int32_t boundary_condition = 0;
    int32_t spline_type = 0;
    int64_t fixed_iterations = -1;       // < 0 = unset
    double energy_threshold = 1e-6;
    double reconstruction_tolerance = 1e-12;
    int32_t validate_reconstruction = 0;
    double intermittency_cv = -1.0;      // < 0 = disabled
    size_t intermittency_min_intervals = 0;
};

struct EnsembleConfig {
    size_t num_ensembles = 100;
    double noise_std = 0.2;
    uint64_t seed = 0;
    int32_t use_seed = 0;
};

struct MemdConfig {
    size_t num_directions = 8;
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
    int32_t use_seed = 0;
};

struct VmdConfig {
    size_t n_modes = 3;
    double alpha = 2000.0;
    double tau = 0.0;
    double tol = 1e-7;
    size_t max_iterations = 500;
};

struct CHilbertResult {
    const double* instantaneous_amplitude;
    const double* instantaneous_frequency;
    const double* marginal_spectrum;
    size_t n_imfs;
    size_t n_samples;
    size_t n_freq_bins;
    int32_t* error;
    char* error_msg;
};

struct CChunkResult {
    const double* const* imfs;
    size_t n_imfs;
    size_t n_samples;
    const double* residue;
    double spectral_entropy;
    double stationarity_score;
    double extrema_spacing_cv;
    int32_t* error;
    char* error_msg;
};

extern "C" {
// Univariate / multivariate decompositions return an opaque result handle.
void* ferromode_emd(const double* signal, size_t n, const EmdConfig* cfg);
void* ferromode_eemd(const double* signal, size_t n, const EnsembleConfig* ens, const EmdConfig* emd);
void* ferromode_ceemd(const double* signal, size_t n, const EnsembleConfig* ens, const EmdConfig* emd);
void* ferromode_ceemdan(const double* signal, size_t n, const EnsembleConfig* ens, const EmdConfig* emd);
void* ferromode_iceemdan(const double* signal, size_t n, const EnsembleConfig* ens, const EmdConfig* emd);
void* ferromode_memd(const double* channels, size_t n_channels, size_t n_samples, const MemdConfig* cfg);
void* ferromode_namemd(const double* channels, size_t n_channels, size_t n_samples, const NaMemdConfig* cfg);
void* ferromode_vmd(const double* signal, size_t n, const VmdConfig* cfg);

// Result accessors.
size_t ferromode_result_n_imfs(const void* r);
size_t ferromode_result_n_samples(const void* r);
size_t ferromode_result_n_siftings(const void* r);
int32_t ferromode_result_algorithm(const void* r);
double ferromode_result_elapsed_ms(const void* r);
int32_t ferromode_result_has_error(const void* r);
const char* ferromode_result_error_msg(const void* r);
int32_t ferromode_copy_imf(const void* r, size_t imf_index, double* out, size_t n_samples);
int32_t ferromode_copy_residue(const void* r, double* out, size_t n_samples);
int32_t ferromode_reconstruct(const void* r, double* out, size_t n_samples);
void ferromode_free_result(void* r);

// Hilbert.
CHilbertResult* ferromode_hilbert(const double* imfs, size_t n_imfs, size_t n_samples, double sample_rate);
void ferromode_free_hilbert(CHilbertResult* h);

// Streaming.
void* ferromode_streaming_new(const EmdConfig* cfg, size_t buffer_size, size_t ar_order);
CChunkResult* ferromode_streaming_decompose_chunk(void* handle, const double* chunk, size_t len);
void ferromode_streaming_reset(void* handle);
void ferromode_streaming_free(void* handle);
void ferromode_free_chunk_result(CChunkResult* c);

// Differentiable.
void* ferromode_diff_forward(const double* signal, size_t len, const EmdConfig* cfg);
size_t ferromode_diff_n_imfs(const void* ctx);
size_t ferromode_diff_n_samples(const void* ctx);
const double* ferromode_diff_imf_ptr(const void* ctx, size_t index);
const double* ferromode_diff_residue_ptr(const void* ctx);
double ferromode_diff_reconstruction_error(const void* ctx);
int32_t ferromode_diff_backward(const double* grad_imfs, size_t n_imfs, size_t len, double* out);
void ferromode_diff_free(void* ctx);
}  // extern "C"

// --- RAII wrappers ----------------------------------------------------------

/// Owns an opaque decomposition-result handle and exposes copied IMF data.
class DecompositionResult {
public:
    explicit DecompositionResult(void* handle) : handle_(handle) {
        if (handle_ == nullptr) {
            throw FerromodeError("decomposition returned null");
        }
        if (ferromode_result_has_error(handle_) != 0) {
            const char* m = ferromode_result_error_msg(handle_);
            std::string msg = m ? m : "decomposition failed";
            ferromode_free_result(handle_);
            handle_ = nullptr;
            throw FerromodeError(msg);
        }
    }
    ~DecompositionResult() { if (handle_) ferromode_free_result(handle_); }
    DecompositionResult(const DecompositionResult&) = delete;
    DecompositionResult& operator=(const DecompositionResult&) = delete;
    DecompositionResult(DecompositionResult&& o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }

    size_t n_imfs() const { return ferromode_result_n_imfs(handle_); }
    size_t n_samples() const { return ferromode_result_n_samples(handle_); }
    size_t n_siftings() const { return ferromode_result_n_siftings(handle_); }
    int algorithm() const { return ferromode_result_algorithm(handle_); }
    double elapsed_ms() const { return ferromode_result_elapsed_ms(handle_); }

    std::vector<double> imf(size_t i) const {
        std::vector<double> out(n_samples());
        if (ferromode_copy_imf(handle_, i, out.data(), out.size()) != 0)
            throw FerromodeError("copy_imf failed");
        return out;
    }
    std::vector<double> residue() const {
        std::vector<double> out(n_samples());
        if (ferromode_copy_residue(handle_, out.data(), out.size()) != 0)
            throw FerromodeError("copy_residue failed");
        return out;
    }
    std::vector<double> reconstruct() const {
        std::vector<double> out(n_samples());
        if (ferromode_reconstruct(handle_, out.data(), out.size()) != 0)
            throw FerromodeError("reconstruct failed");
        return out;
    }
    void* raw() const { return handle_; }

private:
    void* handle_;
};

class HilbertResult {
public:
    explicit HilbertResult(CHilbertResult* h) : h_(h) {
        if (h_ == nullptr || h_->error != nullptr) {
            std::string msg = (h_ && h_->error_msg) ? h_->error_msg : "hilbert failed";
            if (h_) ferromode_free_hilbert(h_);
            h_ = nullptr;
            throw FerromodeError(msg);
        }
    }
    ~HilbertResult() { if (h_) ferromode_free_hilbert(h_); }
    HilbertResult(const HilbertResult&) = delete;
    HilbertResult& operator=(const HilbertResult&) = delete;
    HilbertResult(HilbertResult&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }

    size_t n_imfs() const { return h_->n_imfs; }
    size_t n_samples() const { return h_->n_samples; }
    std::span<const double> marginal_spectrum() const {
        return {h_->marginal_spectrum, h_->n_freq_bins};
    }
    // Flattened row-major (n_imfs x n_samples).
    std::span<const double> amplitude() const {
        return {h_->instantaneous_amplitude, h_->n_imfs * h_->n_samples};
    }
    std::span<const double> frequency() const {
        return {h_->instantaneous_frequency, h_->n_imfs * h_->n_samples};
    }

private:
    CHilbertResult* h_;
};

struct ChunkResult {
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    double spectral_entropy = 0.0;
    double stationarity_score = 0.0;
    double extrema_spacing_cv = 0.0;
};

class StreamingDecomposer {
public:
    StreamingDecomposer(const EmdConfig& cfg, size_t buffer_size = 4096, size_t ar_order = 3)
        : handle_(ferromode_streaming_new(&cfg, buffer_size, ar_order)) {
        if (handle_ == nullptr) throw FerromodeError("failed to create streaming decomposer");
    }
    ~StreamingDecomposer() { if (handle_) ferromode_streaming_free(handle_); }
    StreamingDecomposer(const StreamingDecomposer&) = delete;
    StreamingDecomposer& operator=(const StreamingDecomposer&) = delete;

    ChunkResult decompose_chunk(std::span<const double> chunk) {
        CChunkResult* c = ferromode_streaming_decompose_chunk(handle_, chunk.data(), chunk.size());
        if (c == nullptr || c->error != nullptr) {
            std::string msg = (c && c->error_msg) ? c->error_msg : "decompose_chunk failed";
            if (c) ferromode_free_chunk_result(c);
            throw FerromodeError(msg);
        }
        ChunkResult out;
        out.imfs.reserve(c->n_imfs);
        for (size_t i = 0; i < c->n_imfs; ++i) {
            out.imfs.emplace_back(c->imfs[i], c->imfs[i] + c->n_samples);
        }
        if (c->residue) out.residue.assign(c->residue, c->residue + c->n_samples);
        out.spectral_entropy = c->spectral_entropy;
        out.stationarity_score = c->stationarity_score;
        out.extrema_spacing_cv = c->extrema_spacing_cv;
        ferromode_free_chunk_result(c);
        return out;
    }
    void reset() { ferromode_streaming_reset(handle_); }

private:
    void* handle_;
};

class EmdForwardResult {
public:
    explicit EmdForwardResult(void* ctx) : ctx_(ctx) {
        if (ctx_ == nullptr) throw FerromodeError("emd_forward failed");
    }
    ~EmdForwardResult() { if (ctx_) ferromode_diff_free(ctx_); }
    EmdForwardResult(const EmdForwardResult&) = delete;
    EmdForwardResult& operator=(const EmdForwardResult&) = delete;
    EmdForwardResult(EmdForwardResult&& o) noexcept : ctx_(o.ctx_) { o.ctx_ = nullptr; }

    size_t n_imfs() const { return ferromode_diff_n_imfs(ctx_); }
    size_t n_samples() const { return ferromode_diff_n_samples(ctx_); }
    double reconstruction_error() const { return ferromode_diff_reconstruction_error(ctx_); }
    std::vector<double> imf(size_t i) const {
        const double* p = ferromode_diff_imf_ptr(ctx_, i);
        if (!p) throw FerromodeError("imf index out of range");
        return {p, p + n_samples()};
    }

private:
    void* ctx_;
};

// --- Free-function API ------------------------------------------------------

inline DecompositionResult emd(std::span<const double> s, const EmdConfig& c = EmdConfig{}) {
    return DecompositionResult(ferromode_emd(s.data(), s.size(), &c));
}
inline DecompositionResult eemd(std::span<const double> s, const EnsembleConfig& e, const EmdConfig& c = EmdConfig{}) {
    return DecompositionResult(ferromode_eemd(s.data(), s.size(), &e, &c));
}
inline DecompositionResult ceemd(std::span<const double> s, const EnsembleConfig& e, const EmdConfig& c = EmdConfig{}) {
    return DecompositionResult(ferromode_ceemd(s.data(), s.size(), &e, &c));
}
inline DecompositionResult ceemdan(std::span<const double> s, const EnsembleConfig& e, const EmdConfig& c = EmdConfig{}) {
    return DecompositionResult(ferromode_ceemdan(s.data(), s.size(), &e, &c));
}
inline DecompositionResult iceemdan(std::span<const double> s, const EnsembleConfig& e, const EmdConfig& c = EmdConfig{}) {
    return DecompositionResult(ferromode_iceemdan(s.data(), s.size(), &e, &c));
}
inline DecompositionResult vmd(std::span<const double> s, const VmdConfig& c) {
    return DecompositionResult(ferromode_vmd(s.data(), s.size(), &c));
}

// `channels` is row-major: channel 0's samples, then channel 1's, ...
inline DecompositionResult memd(const std::vector<std::vector<double>>& channels, const MemdConfig& c) {
    std::vector<double> flat;
    for (const auto& ch : channels) flat.insert(flat.end(), ch.begin(), ch.end());
    size_t n_samples = channels.empty() ? 0 : channels[0].size();
    return DecompositionResult(ferromode_memd(flat.data(), channels.size(), n_samples, &c));
}
inline DecompositionResult namemd(const std::vector<std::vector<double>>& channels, const NaMemdConfig& c) {
    std::vector<double> flat;
    for (const auto& ch : channels) flat.insert(flat.end(), ch.begin(), ch.end());
    size_t n_samples = channels.empty() ? 0 : channels[0].size();
    return DecompositionResult(ferromode_namemd(flat.data(), channels.size(), n_samples, &c));
}

// `imfs` is row-major (n_imfs x n_samples).
inline HilbertResult hilbert(std::span<const double> imfs, size_t n_imfs, size_t n_samples, double sample_rate = 1.0) {
    return HilbertResult(ferromode_hilbert(imfs.data(), n_imfs, n_samples, sample_rate));
}

inline EmdForwardResult emd_forward(std::span<const double> s, const EmdConfig& c = EmdConfig{}) {
    return EmdForwardResult(ferromode_diff_forward(s.data(), s.size(), &c));
}
// `grad_imfs` is row-major (n_imfs x len); writes `len` gradients.
inline std::vector<double> emd_backward(std::span<const double> grad_imfs, size_t n_imfs, size_t len) {
    std::vector<double> out(len);
    if (ferromode_diff_backward(grad_imfs.data(), n_imfs, len, out.data()) != 0)
        throw FerromodeError("emd_backward failed");
    return out;
}

}  // namespace ferromode
