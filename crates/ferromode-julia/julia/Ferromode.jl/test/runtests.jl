using Test
using Ferromode

@testset "Ferromode.jl" begin
    @testset "EMD" begin
        n = 100
        signal = sin.(range(0, 2π, length=n))
        result = emd(signal)
        @test result.n_imfs >= 1
        @test result.n_samples == n
        @test result.elapsed_ms >= 0.0
        @test algorithm_name(result) == "emd"

        # Test IMF access
        imf1 = get_imf(result, 0)
        @test length(imf1) == n

        # Test residue
        residue = get_residue(result)
        @test length(residue) == n

        # Test reconstruction
        reconstructed = reconstruct(result)
        @test length(reconstructed) == n
        @test maximum(abs.(signal .- reconstructed)) < 1e-6
    end

    @testset "EEMD" begin
        n = 100
        signal = sin.(range(0, 2π, length=n))
        result = eemd(signal; num_ensembles=5, noise_std=0.2, seed=42)
        @test result.n_imfs >= 1
        @test result.n_samples == n
        @test algorithm_name(result) == "eemd"
    end

    @testset "CEEMD" begin
        n = 100
        signal = sin.(range(0, 2π, length=n))
        result = ceemd(signal; num_ensembles=5, noise_std=0.2, seed=42)
        @test result.n_imfs >= 1
        @test algorithm_name(result) == "ceemd"
    end

    @testset "CEEMDAN" begin
        n = 100
        signal = sin.(range(0, 2π, length=n))
        result = ceemdan(signal; num_ensembles=5, noise_std=0.2, seed=42)
        @test result.n_imfs >= 1
        @test algorithm_name(result) == "ceemdan"
    end

    @testset "ICEEMDAN" begin
        n = 100
        signal = sin.(range(0, 2π, length=n))
        result = iceemdan(signal; num_ensembles=5, noise_std=0.2, seed=42)
        @test result.n_imfs >= 1
        @test algorithm_name(result) == "iceemdan"
    end

    @testset "VMD" begin
        n = 200
        t = range(0, 1, length=n)
        signal = sin.(2π * 5 .* t) .+ 0.5 .* sin.(2π * 20 .* t)
        result = vmd(signal; n_modes=2, alpha=2000.0, tol=1e-7, max_iterations=500)
        @test result.n_imfs == 2
        @test result.n_samples == n
        @test algorithm_name(result) == "vmd"
    end

    @testset "MEMD" begin
        n = 100
        t = range(0, 2π, length=n)
        signal = hcat(sin.(t), sin.(t))  # 2-channel bivariate signal
        result = memd(signal; num_directions=16, direction_seed=42, max_imfs=3, max_sifting_iterations=50)
        @test result.n_imfs >= 1
        @test algorithm_name(result) == "memd"
    end

    @testset "NA-MEMD" begin
        n = 100
        t = range(0, 2π, length=n)
        signal = hcat(sin.(t), sin.(t))
        result = namemd(signal; num_directions=16, direction_seed=42, n_noise_channels=2, noise_std=0.1, seed=42, max_sifting_iterations=50)
        @test result.n_imfs >= 1
        @test algorithm_name(result) == "namemd"
    end

    @testset "Config structs" begin
        emd_cfg = EmdConfig(max_imfs=5, sd_threshold=0.1)
        @test emd_cfg.max_imfs == 5
        @test emd_cfg.sd_threshold == 0.1

        ens_cfg = EnsembleConfig(num_ensembles=50, seed=123)
        @test ens_cfg.num_ensembles == 50
        @test ens_cfg.use_seed == 1

        ens_cfg_no_seed = EnsembleConfig(num_ensembles=50)
        @test ens_cfg_no_seed.use_seed == 0

        vmd_cfg = VmdConfig(n_modes=5, alpha=1000.0)
        @test vmd_cfg.n_modes == 5
        @test vmd_cfg.alpha == 1000.0

        memd_cfg = MemdConfig(num_directions=32, max_imfs=10)
        @test memd_cfg.num_directions == 32
        @test memd_cfg.max_imfs == 10

        namemd_cfg = NaMemdConfig(n_noise_channels=4, noise_std=0.2, seed=99)
        @test namemd_cfg.n_noise_channels == 4
        @test namemd_cfg.use_seed == 1
    end

    @testset "Boundary + full config" begin
        n = 128
        signal = sin.(range(0, 4π, length=n))
        # palindrome_cyclic boundary (code 7) + periodic spline + extra knobs
        result = emd(signal; max_imfs=4, boundary_condition=7, spline_type=1,
                     energy_threshold=1e-7)
        @test result.n_imfs >= 1
        @test algorithm_name(result) == "emd"
    end

    @testset "Hilbert" begin
        n = 128
        signal = sin.(range(0, 8π, length=n))
        result = emd(signal; max_imfs=4)
        imfs = [get_imf(result, i) for i in 0:(result.n_imfs - 1)]
        h = hilbert(imfs, 100.0)
        @test size(h.instantaneous_amplitude, 1) == result.n_imfs
        @test size(h.instantaneous_amplitude, 2) == n
        @test length(h.marginal_spectrum) > 0
    end

    @testset "Streaming" begin
        dec = StreamingDecomposer(; max_imfs=4, buffer_size=2048, ar_order=3)
        chunk = sin.(range(0, 4π, length=256))
        out = decompose_chunk(dec, chunk)
        @test length(out.imfs) >= 1
        @test haskey(out.metrics, :spectral_entropy)
        reset!(dec)
    end

    @testset "Differentiable" begin
        n = 200
        signal = sin.(range(0, 4π, length=n)) .+ 0.3 .* range(0, 1, length=n)
        ctx = emd_forward(signal; max_imfs=4)
        @test ctx.n_imfs >= 1
        @test length(get_imf(ctx, 0)) == n
        @test isfinite(reconstruction_error(ctx))
        grads = [ones(n) for _ in 1:ctx.n_imfs]
        grad = emd_backward(grads, signal)
        @test length(grad) == n
        @test abs(grad[1] - 1.0) < 1e-12
    end

    @testset "Version" begin
        @test version() == "0.1.0"
    end
end
