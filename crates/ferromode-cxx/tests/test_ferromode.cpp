// Catch2 test suite for Ferromode C++ bindings.
// Tests verify output vector sizes, reconstruction, and error handling.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "ferromode.hpp"

#include <cmath>

namespace {

constexpr double PI = 3.14159265358979323846;

std::vector<double> make_sine_signal(size_t n) {
    std::vector<double> signal(n);
    for (size_t i = 0; i < n; ++i) {
        signal[i] = std::sin(2.0 * PI * static_cast<double>(i) / static_cast<double>(n));
    }
    return signal;
}

std::vector<double> make_multi_component_signal(size_t n) {
    std::vector<double> signal(n);
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n);
        signal[i] = std::sin(2.0 * PI * 5.0 * t) + 0.5 * std::sin(2.0 * PI * 20.0 * t);
    }
    return signal;
}

} // namespace

// =========================================================================
// EMD tests
// =========================================================================

TEST_CASE("EMD produces IMFs for sine wave", "[emd]") {
    auto signal = make_sine_signal(100);
    ferromode::EmdConfig config;
    config.sd_threshold = 0.2;
    config.s_number = 5;
    config.max_sifting_iterations = 100;

    auto result = ferromode::emd(signal, config);

    REQUIRE(result.n_imfs() >= 1);
    REQUIRE(result.n_samples() == 100);
}

TEST_CASE("EMD reconstruction returns correct length", "[emd]") {
    auto signal = make_sine_signal(100);
    ferromode::EmdConfig config;

    auto result = ferromode::emd(signal, config);
    auto reconstructed = result.reconstruct();

    REQUIRE(reconstructed.size() == signal.size());
}

TEST_CASE("EMD IMF spans have correct sizes", "[emd]") {
    auto signal = make_sine_signal(100);
    ferromode::EmdConfig config;

    auto result = ferromode::emd(signal, config);
    auto imfs = result.imfs();

    REQUIRE(imfs.size() == result.n_imfs());
    for (const auto& imf : imfs) {
        REQUIRE(imf.size() == result.n_samples());
    }
}

TEST_CASE("EMD residue has correct size", "[emd]") {
    auto signal = make_sine_signal(100);
    ferromode::EmdConfig config;

    auto result = ferromode::emd(signal, config);
    auto residue = result.residue();

    REQUIRE(residue.size() == result.n_samples());
}

// =========================================================================
// EEMD tests
// =========================================================================

TEST_CASE("EEMD produces IMFs for sine wave", "[eemd]") {
    auto signal = make_sine_signal(100);
    ferromode::EnsembleConfig ens_config;
    ens_config.num_ensembles = 5;
    ens_config.noise_std = 0.2;
    ens_config.seed = 42;
    ens_config.use_seed = true;

    ferromode::EmdConfig emd_config;

    auto result = ferromode::eemd(signal, ens_config, emd_config);

    REQUIRE(result.n_imfs() >= 1);
    REQUIRE(result.n_samples() == 100);
}

TEST_CASE("EEMD reconstruction returns correct length", "[eemd]") {
    auto signal = make_sine_signal(100);
    ferromode::EnsembleConfig ens_config;
    ens_config.num_ensembles = 5;
    ens_config.noise_std = 0.2;
    ens_config.seed = 42;
    ens_config.use_seed = true;

    ferromode::EmdConfig emd_config;
    auto result = ferromode::eemd(signal, ens_config, emd_config);
    auto reconstructed = result.reconstruct();

    REQUIRE(reconstructed.size() == signal.size());
}

// =========================================================================
// CEEMD tests
// =========================================================================

TEST_CASE("CEEMD produces IMFs for sine wave", "[ceemd]") {
    auto signal = make_sine_signal(100);
    ferromode::EnsembleConfig ens_config;
    ens_config.num_ensembles = 5;
    ens_config.noise_std = 0.2;
    ens_config.seed = 42;
    ens_config.use_seed = true;

    ferromode::EmdConfig emd_config;
    auto result = ferromode::ceemd(signal, ens_config, emd_config);

    REQUIRE(result.n_imfs() >= 1);
    REQUIRE(result.n_samples() == 100);
}

// =========================================================================
// CEEMDAN tests
// =========================================================================

TEST_CASE("CEEMDAN produces IMFs for sine wave", "[ceemdan]") {
    auto signal = make_sine_signal(100);
    ferromode::EnsembleConfig ens_config;
    ens_config.num_ensembles = 5;
    ens_config.noise_std = 0.2;
    ens_config.seed = 42;
    ens_config.use_seed = true;

    ferromode::EmdConfig emd_config;
    auto result = ferromode::ceemdan(signal, ens_config, emd_config);

    REQUIRE(result.n_imfs() >= 1);
    REQUIRE(result.n_samples() == 100);
}

// =========================================================================
// ICEEMDAN tests
// =========================================================================

TEST_CASE("ICEEMDAN produces IMFs for sine wave", "[iceemdan]") {
    auto signal = make_sine_signal(100);
    ferromode::EnsembleConfig ens_config;
    ens_config.num_ensembles = 5;
    ens_config.noise_std = 0.2;
    ens_config.seed = 42;
    ens_config.use_seed = true;

    ferromode::EmdConfig emd_config;
    auto result = ferromode::iceemdan(signal, ens_config, emd_config);

    REQUIRE(result.n_imfs() >= 1);
    REQUIRE(result.n_samples() == 100);
}

// =========================================================================
// VMD tests
// =========================================================================

TEST_CASE("VMD produces requested number of modes", "[vmd]") {
    auto signal = make_multi_component_signal(200);
    ferromode::VmdConfig config;
    config.n_modes = 2;
    config.alpha = 2000.0;
    config.tau = 0.0;
    config.tol = 1e-7;
    config.max_iterations = 500;

    auto result = ferromode::vmd(signal, config);

    REQUIRE(result.n_imfs() == 2);
    REQUIRE(result.n_samples() == 200);
}

TEST_CASE("VMD reconstruction returns correct length", "[vmd]") {
    auto signal = make_multi_component_signal(200);
    ferromode::VmdConfig config;
    config.n_modes = 2;
    config.alpha = 2000.0;

    auto result = ferromode::vmd(signal, config);
    auto reconstructed = result.reconstruct();

    REQUIRE(reconstructed.size() == signal.size());
}

// =========================================================================
// Hilbert spectral analysis tests
// =========================================================================

TEST_CASE("Hilbert analysis produces correct output sizes", "[hilbert]") {
    auto signal = make_sine_signal(100);
    ferromode::EmdConfig emd_config;
    auto imfs = ferromode::emd(signal, emd_config);

    auto hilbert_result = ferromode::hilbert(imfs);

    REQUIRE(hilbert_result.n_imfs() == imfs.n_imfs());
    REQUIRE(hilbert_result.n_samples() == imfs.n_samples());
    REQUIRE(hilbert_result.n_freq_bins() > 0);
}

TEST_CASE("Hilbert instantaneous amplitude spans have correct sizes", "[hilbert]") {
    auto signal = make_sine_signal(100);
    ferromode::EmdConfig emd_config;
    auto imfs = ferromode::emd(signal, emd_config);

    auto hilbert_result = ferromode::hilbert(imfs);
    auto amplitudes = hilbert_result.instantaneous_amplitude();

    REQUIRE(amplitudes.size() == hilbert_result.n_imfs());
    for (const auto& amp : amplitudes) {
        REQUIRE(amp.size() == hilbert_result.n_samples());
    }
}

TEST_CASE("Hilbert marginal spectrum has correct size", "[hilbert]") {
    auto signal = make_sine_signal(100);
    ferromode::EmdConfig emd_config;
    auto imfs = ferromode::emd(signal, emd_config);

    auto hilbert_result = ferromode::hilbert(imfs);
    auto marginal = hilbert_result.marginal_spectrum();

    REQUIRE(marginal.size() == hilbert_result.n_freq_bins());
}

// =========================================================================
// Error handling tests
// =========================================================================

TEST_CASE("EMD throws on empty signal", "[error]") {
    std::vector<double> empty_signal;
    ferromode::EmdConfig config;

    REQUIRE_THROWS_AS(ferromode::emd(empty_signal, config), ferromode::FerromodeError);
}

TEST_CASE("EMD throws on signal with NaN", "[error]") {
    std::vector<double> signal = {1.0, std::nan(""), 3.0, 4.0, 5.0};
    ferromode::EmdConfig config;

    REQUIRE_THROWS_AS(ferromode::emd(signal, config), ferromode::FerromodeError);
}

TEST_CASE("EMD throws on signal with Inf", "[error]") {
    std::vector<double> signal = {1.0, std::numeric_limits<double>::infinity(), 3.0};
    ferromode::EmdConfig config;

    REQUIRE_THROWS_AS(ferromode::emd(signal, config), ferromode::FerromodeError);
}

TEST_CASE("VMD throws on empty signal", "[error]") {
    std::vector<double> empty_signal;
    ferromode::VmdConfig config;

    REQUIRE_THROWS_AS(ferromode::vmd(empty_signal, config), ferromode::FerromodeError);
}

// =========================================================================
// Config default values tests
// =========================================================================

TEST_CASE("EmdConfig has sensible defaults", "[config]") {
    ferromode::EmdConfig config;
    REQUIRE(config.max_imfs == 0);
    REQUIRE(config.sd_threshold == 0.2);
    REQUIRE(config.s_number == 5);
    REQUIRE(config.max_sifting_iterations == 100);
    REQUIRE(config.boundary_condition == 0);
}

TEST_CASE("EnsembleConfig has sensible defaults", "[config]") {
    ferromode::EnsembleConfig config;
    REQUIRE(config.num_ensembles == 10);
    REQUIRE(config.noise_std == 0.2);
    REQUIRE(config.seed == 0);
    REQUIRE(config.use_seed == false);
}

TEST_CASE("VmdConfig has sensible defaults", "[config]") {
    ferromode::VmdConfig config;
    REQUIRE(config.n_modes == 3);
    REQUIRE(config.alpha == 2000.0);
    REQUIRE(config.tau == 0.0);
    REQUIRE(config.tol == 1e-7);
    REQUIRE(config.max_iterations == 500);
}

// =========================================================================
// Move semantics tests
// =========================================================================

TEST_CASE("ImfCollection supports move construction", "[move]") {
    auto signal = make_sine_signal(100);
    ferromode::EmdConfig config;

    auto result1 = ferromode::emd(signal, config);
    size_t n_imfs = result1.n_imfs();
    size_t n_samples = result1.n_samples();

    auto result2 = std::move(result1);

    REQUIRE(result2.n_imfs() == n_imfs);
    REQUIRE(result2.n_samples() == n_samples);
    REQUIRE(result2.imfs().size() == n_imfs);
}

TEST_CASE("ImfCollection supports move assignment", "[move]") {
    auto signal = make_sine_signal(100);
    ferromode::EmdConfig config;

    auto result1 = ferromode::emd(signal, config);
    ferromode::ImfCollection result2;
    result2 = std::move(result1);

    REQUIRE(result2.n_imfs() > 0);
    REQUIRE(result2.n_samples() == 100);
}

TEST_CASE("HilbertResult supports move construction", "[move]") {
    auto signal = make_sine_signal(100);
    ferromode::EmdConfig emd_config;
    auto imfs = ferromode::emd(signal, emd_config);

    auto result1 = ferromode::hilbert(imfs);
    size_t n_imfs = result1.n_imfs();

    auto result2 = std::move(result1);

    REQUIRE(result2.n_imfs() == n_imfs);
    REQUIRE(result2.instantaneous_amplitude().size() == n_imfs);
}
