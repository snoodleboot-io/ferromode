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

    @testset "Version" begin
        @test version() == "0.1.0"
    end
end
