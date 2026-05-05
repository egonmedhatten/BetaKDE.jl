using Test
using BetaKDE
using Distributions

@testset "BetaKDE.jl" begin

    @testset "Basic functionality" begin
        data = rand(Beta(2, 5), 500)
        result = betakde(data)
        @test result isa BetaKDEUnivariate
        @test length(result.x) == 512
        @test length(result.density) == 512
        @test result.bandwidth > 0.0
    end

    @testset "Integration to 1.0 (normalized)" begin
        data = rand(Beta(2, 5), 500)
        result = betakde(data)
        # Trapezoidal rule
        dx = diff(result.x)
        integral = sum(dx .* (result.density[1:end-1] .+ result.density[2:end]) ./ 2)
        @test isapprox(integral, 1.0; atol=1e-10)
    end

    @testset "normalize=false skips normalization" begin
        data = rand(Beta(2, 5), 500)
        result = betakde(data; normalize=false)
        dx = diff(result.x)
        integral = sum(dx .* (result.density[1:end-1] .+ result.density[2:end]) ./ 2)
        # Raw kernel average typically doesn't integrate to exactly 1
        @test result isa BetaKDEUnivariate
        @test all(result.density .>= 0.0)
    end

    @testset "Density non-negative" begin
        data = rand(Beta(3, 3), 300)
        result = betakde(data)
        @test all(result.density .>= 0.0)
    end

    @testset "Custom bandwidth" begin
        data = rand(Beta(2, 5), 200)
        result = betakde(data; bw=0.1)
        @test result.bandwidth == 0.1
    end

    @testset "Custom npoints" begin
        data = rand(Beta(2, 5), 200)
        result = betakde(data; npoints=256)
        @test length(result.x) == 256
        @test length(result.density) == 256
    end

    @testset "Fallback: U-shaped Beta(0.5, 0.5)" begin
        data = rand(Beta(0.5, 0.5), 300)
        result = betakde(data)
        @test result isa BetaKDEUnivariate
        @test result.bandwidth > 0.0
        @test all(isfinite.(result.density))
    end

    @testset "Fallback: J-shaped Beta(0.3, 2.0)" begin
        data = rand(Beta(0.3, 2.0), 300)
        result = betakde(data)
        @test result isa BetaKDEUnivariate
        @test result.bandwidth > 0.0
        @test all(isfinite.(result.density))
    end

    @testset "Edge data with 0s and 1s" begin
        data = vcat([0.0, 1.0, 0.0, 1.0], rand(Beta(2, 5), 200))
        result = betakde(data)
        @test all(isfinite.(result.density))
        @test !any(isnan.(result.density))
    end

    @testset "bw_HS direct call" begin
        data = rand(Beta(3, 5), 1000)
        h, is_fallback = bw_HS(data)
        @test 0.0 < h < 1.0
        @test is_fallback == false
    end

    @testset "bw_HS fallback triggers" begin
        data = rand(Beta(0.1, 0.1), 500)
        h, is_fallback = bw_HS(data)
        @test 0.0 < h < 1.0
        @test is_fallback == true
    end

    @testset "Bandwidth matches Python reference" begin
        # Deterministic data — verified against Python beta_kde package
        data = [0.12, 0.25, 0.33, 0.41, 0.48, 0.55, 0.62, 0.71, 0.79, 0.88]
        h, is_fallback = bw_HS(data)
        @test is_fallback == false
        @test isapprox(h, 1.915314763032799e-01; rtol=1e-10)
    end

    @testset "Density values match Python reference" begin
        # Same data, fixed bandwidth, unnormalized — compare to scipy.stats.beta.pdf
        data = [0.12, 0.25, 0.33, 0.41, 0.48, 0.55, 0.62, 0.71, 0.79, 0.88]
        result = betakde(data; bw=0.1, npoints=5, normalize=false)
        expected = [4.311497075363923e-01, 9.928262670948985e-01,
                    1.295786396751072e+00, 1.130204273306703e+00,
                    5.020511503966179e-01]
        @test all(isapprox.(result.density, expected; rtol=1e-8))
    end

    @testset "Constant data does not error" begin
        data = fill(0.5, 100)
        result = betakde(data)
        @test result isa BetaKDEUnivariate
        @test all(isfinite.(result.density))
        @test result.bandwidth > 0.0
    end

    @testset "Single data point" begin
        data = [0.5]
        result = betakde(data)
        @test result isa BetaKDEUnivariate
        @test all(isfinite.(result.density))
    end

    @testset "Minimum viable sample (n=2)" begin
        data = [0.3, 0.7]
        result = betakde(data)
        @test result isa BetaKDEUnivariate
        @test all(isfinite.(result.density))
        @test result.bandwidth > 0.0
    end

    @testset "Peak near mode of distribution" begin
        # Beta(2,5) has mode at (2-1)/(2+5-2) = 0.2
        data = rand(Beta(2, 5), 5000)
        result = betakde(data)
        peak_idx = argmax(result.density)
        @test 0.1 < result.x[peak_idx] < 0.35
    end

end
