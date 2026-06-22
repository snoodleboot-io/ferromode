// Catch2 test suite for the Ferromode C++ bindings (wrapping the C ABI).

#include <catch2/catch_test_macros.hpp>
#include "ferromode.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
constexpr double PI = 3.14159265358979323846;

std::vector<double> sine(size_t n, double freq = 5.0) {
    std::vector<double> s(n);
    for (size_t i = 0; i < n; ++i)
        s[i] = std::sin(2.0 * PI * freq * static_cast<double>(i) / static_cast<double>(n));
    return s;
}
}  // namespace

TEST_CASE("EMD decomposes and reconstructs", "[emd]") {
    auto signal = sine(200);
    ferromode::EmdConfig cfg;
    cfg.max_imfs = 4;
    auto r = ferromode::emd(signal, cfg);
    REQUIRE(r.n_imfs() >= 1);
    REQUIRE(r.n_samples() == 200);
    REQUIRE(r.imf(0).size() == 200);
    REQUIRE(r.reconstruct().size() == 200);
}

TEST_CASE("EMD honors palindrome boundary + spline + knobs", "[emd][config]") {
    auto signal = sine(128);
    ferromode::EmdConfig cfg;
    cfg.max_imfs = 4;
    cfg.boundary_condition = static_cast<int32_t>(ferromode::Boundary::PalindromeCyclic);
    cfg.spline_type = static_cast<int32_t>(ferromode::Spline::Periodic);
    cfg.energy_threshold = 1e-7;
    auto r = ferromode::emd(signal, cfg);
    REQUIRE(r.n_imfs() >= 1);
}

TEST_CASE("Ensemble methods run", "[ensemble]") {
    auto signal = sine(200);
    ferromode::EnsembleConfig ens;
    ens.num_ensembles = 5;
    ens.noise_std = 0.2;
    ens.seed = 42;
    ens.use_seed = 1;
    ferromode::EmdConfig emd;
    emd.max_imfs = 4;
    REQUIRE(ferromode::eemd(signal, ens, emd).n_imfs() >= 1);
    REQUIRE(ferromode::ceemd(signal, ens, emd).n_imfs() >= 1);
    REQUIRE(ferromode::ceemdan(signal, ens, emd).n_imfs() >= 1);
    REQUIRE(ferromode::iceemdan(signal, ens, emd).n_imfs() >= 1);
}

TEST_CASE("VMD runs", "[vmd]") {
    auto signal = sine(200);
    ferromode::VmdConfig cfg;
    cfg.n_modes = 2;
    auto r = ferromode::vmd(signal, cfg);
    REQUIRE(r.n_imfs() == 2);
}

TEST_CASE("MEMD and NA-MEMD run", "[mv]") {
    std::vector<std::vector<double>> ch = {sine(128, 5.0), sine(128, 7.0)};
    ferromode::MemdConfig mc;
    mc.num_directions = 16;
    mc.max_imfs = 3;
    REQUIRE(ferromode::memd(ch, mc).n_imfs() >= 1);

    ferromode::NaMemdConfig nc;
    nc.base = mc;
    nc.n_noise_channels = 2;
    REQUIRE(ferromode::namemd(ch, nc).n_imfs() >= 1);
}

TEST_CASE("Hilbert spectrum", "[hilbert]") {
    auto signal = sine(128);
    ferromode::EmdConfig cfg;
    cfg.max_imfs = 4;
    auto r = ferromode::emd(signal, cfg);
    // Flatten IMFs row-major.
    std::vector<double> flat;
    for (size_t i = 0; i < r.n_imfs(); ++i) {
        auto imf = r.imf(i);
        flat.insert(flat.end(), imf.begin(), imf.end());
    }
    auto h = ferromode::hilbert(flat, r.n_imfs(), r.n_samples(), 100.0);
    REQUIRE(h.n_imfs() == r.n_imfs());
    REQUIRE(h.marginal_spectrum().size() > 0);
}

TEST_CASE("Streaming decomposes chunks", "[streaming]") {
    ferromode::EmdConfig cfg;
    cfg.max_imfs = 4;
    ferromode::StreamingDecomposer dec(cfg, 2048, 3);
    auto chunk = sine(256);
    auto out = dec.decompose_chunk(chunk);
    REQUIRE(out.imfs.size() >= 1);
    dec.reset();
}

TEST_CASE("Differentiable forward + backward", "[diff]") {
    auto signal = sine(200);
    ferromode::EmdConfig cfg;
    cfg.max_imfs = 4;
    auto fwd = ferromode::emd_forward(signal, cfg);
    REQUIRE(fwd.n_imfs() >= 1);
    REQUIRE(std::isfinite(fwd.reconstruction_error()));
    REQUIRE(fwd.imf(0).size() == 200);

    size_t n_imfs = fwd.n_imfs();
    std::vector<double> grads(n_imfs * 200, 1.0);
    auto grad = fwd.backward(grads);  // exact VJP via the saved forward context
    REQUIRE(grad.size() == 200);
    REQUIRE(std::all_of(grad.begin(), grad.end(), [](double v) { return std::isfinite(v); }));
}
