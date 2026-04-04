// C++ implementation calling cxx-generated bindings.
// This file is compiled by cxx-build and linked into the static library.

#include "rust/cxx.h"
#include "ferromode.hpp"
#include "crates/ferromode-cxx/src/lib.rs.h"

namespace ferromode {

extern "C" {

CImfResult ferromode_emd_cxx(const double* signal, size_t n, const EmdConfig* config) {
    ::CEmdConfig c_config;
    c_config.max_imfs = config->max_imfs;
    c_config.sd_threshold = config->sd_threshold;
    c_config.s_number = config->s_number;
    c_config.max_sifting_iterations = config->max_sifting_iterations;
    c_config.boundary_condition = config->boundary_condition;

    rust::Slice<const double> signal_slice(signal, n);
    auto result = ::ferromode_emd_cxx(signal_slice, c_config);

    CImfResult out{};
    out.n_imfs = result.n_imfs;
    out.n_samples = result.n_samples;
    out.imfs_data = result.imfs_data;
    out.residue = result.residue;
    return out;
}

CImfResult ferromode_eemd_cxx(const double* signal, size_t n, const EnsembleConfig* config, const EmdConfig* emd_config) {
    ::CEnsembleConfig c_ens;
    c_ens.num_ensembles = config->num_ensembles;
    c_ens.noise_std = config->noise_std;
    c_ens.seed = config->seed;
    c_ens.use_seed = config->use_seed ? 1 : 0;

    ::CEmdConfig c_emd;
    c_emd.max_imfs = emd_config->max_imfs;
    c_emd.sd_threshold = emd_config->sd_threshold;
    c_emd.s_number = emd_config->s_number;
    c_emd.max_sifting_iterations = emd_config->max_sifting_iterations;
    c_emd.boundary_condition = emd_config->boundary_condition;

    rust::Slice<const double> signal_slice(signal, n);
    auto result = ::ferromode_eemd_cxx(signal_slice, c_ens, c_emd);

    CImfResult out{};
    out.n_imfs = result.n_imfs;
    out.n_samples = result.n_samples;
    out.imfs_data = result.imfs_data;
    out.residue = result.residue;
    return out;
}

CImfResult ferromode_ceemd_cxx(const double* signal, size_t n, const EnsembleConfig* config, const EmdConfig* emd_config) {
    ::CEnsembleConfig c_ens;
    c_ens.num_ensembles = config->num_ensembles;
    c_ens.noise_std = config->noise_std;
    c_ens.seed = config->seed;
    c_ens.use_seed = config->use_seed ? 1 : 0;

    ::CEmdConfig c_emd;
    c_emd.max_imfs = emd_config->max_imfs;
    c_emd.sd_threshold = emd_config->sd_threshold;
    c_emd.s_number = emd_config->s_number;
    c_emd.max_sifting_iterations = emd_config->max_sifting_iterations;
    c_emd.boundary_condition = emd_config->boundary_condition;

    rust::Slice<const double> signal_slice(signal, n);
    auto result = ::ferromode_ceemd_cxx(signal_slice, c_ens, c_emd);

    CImfResult out{};
    out.n_imfs = result.n_imfs;
    out.n_samples = result.n_samples;
    out.imfs_data = result.imfs_data;
    out.residue = result.residue;
    return out;
}

CImfResult ferromode_ceemdan_cxx(const double* signal, size_t n, const EnsembleConfig* config, const EmdConfig* emd_config) {
    ::CEnsembleConfig c_ens;
    c_ens.num_ensembles = config->num_ensembles;
    c_ens.noise_std = config->noise_std;
    c_ens.seed = config->seed;
    c_ens.use_seed = config->use_seed ? 1 : 0;

    ::CEmdConfig c_emd;
    c_emd.max_imfs = emd_config->max_imfs;
    c_emd.sd_threshold = emd_config->sd_threshold;
    c_emd.s_number = emd_config->s_number;
    c_emd.max_sifting_iterations = emd_config->max_sifting_iterations;
    c_emd.boundary_condition = emd_config->boundary_condition;

    rust::Slice<const double> signal_slice(signal, n);
    auto result = ::ferromode_ceemdan_cxx(signal_slice, c_ens, c_emd);

    CImfResult out{};
    out.n_imfs = result.n_imfs;
    out.n_samples = result.n_samples;
    out.imfs_data = result.imfs_data;
    out.residue = result.residue;
    return out;
}

CImfResult ferromode_iceemdan_cxx(const double* signal, size_t n, const EnsembleConfig* config, const EmdConfig* emd_config) {
    ::CEnsembleConfig c_ens;
    c_ens.num_ensembles = config->num_ensembles;
    c_ens.noise_std = config->noise_std;
    c_ens.seed = config->seed;
    c_ens.use_seed = config->use_seed ? 1 : 0;

    ::CEmdConfig c_emd;
    c_emd.max_imfs = emd_config->max_imfs;
    c_emd.sd_threshold = emd_config->sd_threshold;
    c_emd.s_number = emd_config->s_number;
    c_emd.max_sifting_iterations = emd_config->max_sifting_iterations;
    c_emd.boundary_condition = emd_config->boundary_condition;

    rust::Slice<const double> signal_slice(signal, n);
    auto result = ::ferromode_iceemdan_cxx(signal_slice, c_ens, c_emd);

    CImfResult out{};
    out.n_imfs = result.n_imfs;
    out.n_samples = result.n_samples;
    out.imfs_data = result.imfs_data;
    out.residue = result.residue;
    return out;
}

CImfResult ferromode_memd_cxx(const double* channels, size_t n_channels, size_t n_samples, const MemdConfig* config) {
    ::CMemdConfig c_config;
    c_config.num_directions = config->num_directions;
    c_config.direction_seed = config->direction_seed;
    c_config.max_imfs = config->max_imfs;
    c_config.sd_threshold = config->sd_threshold;
    c_config.s_number = config->s_number;
    c_config.max_sifting_iterations = config->max_sifting_iterations;

    rust::Slice<const double> channel_slice(channels, n_channels * n_samples);
    auto result = ::ferromode_memd_cxx(channel_slice, n_channels, n_samples, c_config);

    CImfResult out{};
    out.n_imfs = result.n_imfs;
    out.n_samples = result.n_samples;
    out.imfs_data = result.imfs_data;
    out.residue = result.residue;
    return out;
}

CImfResult ferromode_namemd_cxx(const double* channels, size_t n_channels, size_t n_samples, const NaMemdConfig* config) {
    ::CNaMemdConfig c_config;
    c_config.base.num_directions = config->base.num_directions;
    c_config.base.direction_seed = config->base.direction_seed;
    c_config.base.max_imfs = config->base.max_imfs;
    c_config.base.sd_threshold = config->base.sd_threshold;
    c_config.base.s_number = config->base.s_number;
    c_config.base.max_sifting_iterations = config->base.max_sifting_iterations;
    c_config.n_noise_channels = config->n_noise_channels;
    c_config.noise_std = config->noise_std;
    c_config.seed = config->seed;
    c_config.use_seed = config->use_seed ? 1 : 0;

    rust::Slice<const double> channel_slice(channels, n_channels * n_samples);
    auto result = ::ferromode_namemd_cxx(channel_slice, n_channels, n_samples, c_config);

    CImfResult out{};
    out.n_imfs = result.n_imfs;
    out.n_samples = result.n_samples;
    out.imfs_data = result.imfs_data;
    out.residue = result.residue;
    return out;
}

CImfResult ferromode_vmd_cxx(const double* signal, size_t n, const VmdConfig* config) {
    ::CVmdConfig c_config;
    c_config.n_modes = config->n_modes;
    c_config.alpha = config->alpha;
    c_config.tau = config->tau;
    c_config.tol = config->tol;
    c_config.max_iterations = config->max_iterations;

    rust::Slice<const double> signal_slice(signal, n);
    auto result = ::ferromode_vmd_cxx(signal_slice, c_config);

    CImfResult out{};
    out.n_imfs = result.n_imfs;
    out.n_samples = result.n_samples;
    out.imfs_data = result.imfs_data;
    out.residue = result.residue;
    return out;
}

CHilbertResult ferromode_hilbert_cxx(const double* imfs, size_t n_imfs, size_t n_samples, double sample_rate) {
    rust::Slice<const double> imf_slice(imfs, n_imfs * n_samples);
    auto result = ::ferromode_hilbert_cxx(imf_slice, n_imfs, n_samples, sample_rate);

    CHilbertResult out{};
    out.n_imfs = result.n_imfs;
    out.n_samples = result.n_samples;
    out.n_freq_bins = result.n_freq_bins;
    out.instantaneous_amplitude = result.instantaneous_amplitude;
    out.instantaneous_frequency = result.instantaneous_frequency;
    out.marginal_spectrum = result.marginal_spectrum;
    return out;
}

} // extern "C"

} // namespace ferromode
