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

    @testset "Arbitrary bounds [a, b]" begin
        # Generate data on [2, 7] from a scaled Beta
        raw = rand(Beta(2, 5), 500)
        data = raw .* 5.0 .+ 2.0  # [2, 7]
        result = betakde(data; lower=2.0, upper=7.0)

        @test result.lower == 2.0
        @test result.upper == 7.0
        @test result.x[1] ≈ 2.0
        @test result.x[end] ≈ 7.0

        # Density integrates to 1
        dx = diff(result.x)
        integral = sum(dx .* (result.density[1:end-1] .+ result.density[2:end]) ./ 2)
        @test isapprox(integral, 1.0; atol=1e-10)

        # All density values non-negative
        @test all(result.density .>= 0.0)
    end

    @testset "Bounds default matches original" begin
        data = rand(Beta(3, 3), 300)
        r1 = betakde(data)
        r2 = betakde(data; lower=0.0, upper=1.0)
        @test r1.x ≈ r2.x
        @test r1.density ≈ r2.density
        @test r1.bandwidth == r2.bandwidth
    end

    @testset "Invalid bounds error" begin
        @test_throws ArgumentError betakde([0.5]; lower=1.0, upper=0.0)
        @test_throws ArgumentError betakde([0.5]; lower=1.0, upper=1.0)
    end

    @testset "Bounds with clamping" begin
        # Data outside [0, 10] should be clamped
        data = [-1.0, 0.5, 5.0, 9.5, 11.0]
        result = betakde(data; lower=0.0, upper=10.0)
        @test all(isfinite.(result.density))
        @test result.x[1] ≈ 0.0
        @test result.x[end] ≈ 10.0
    end

end

@testset "StatsBase extension" begin
    using StatsBase

    data = rand(Beta(2, 5), 500)
    result = fit(BetaKDEUnivariate, data)

    @test result isa BetaKDEUnivariate
    @test length(result.x) == 512
    @test result.bandwidth > 0.0

    # Keyword arguments pass through
    result2 = fit(BetaKDEUnivariate, data; bw=0.1, npoints=256)
    @test result2.bandwidth == 0.1
    @test length(result2.x) == 256

    # Bounds pass through
    data_scaled = data .* 5.0
    result3 = fit(BetaKDEUnivariate, data_scaled; lower=0.0, upper=5.0)
    @test result3.x[end] ≈ 5.0
end

@testset "Point evaluation (pdf / logpdf)" begin
    using Distributions: pdf, logpdf

    data = rand(Beta(2, 5), 1000)
    result = betakde(data)

    @testset "pdf at grid points matches stored density" begin
        for i in [1, 100, 256, 512]
            @test pdf(result, result.x[i]) ≈ result.density[i] rtol=1e-10
        end
    end

    @testset "pdf at off-grid points is exact" begin
        mid_x = (result.x[100] + result.x[101]) / 2
        @test pdf(result, mid_x) > 0.0
        # Callable syntax gives same result
        @test result(mid_x) == pdf(result, mid_x)
    end

    @testset "pdf outside support returns 0" begin
        @test pdf(result, -0.1) == 0.0
        @test pdf(result, 1.1) == 0.0
    end

    @testset "logpdf consistent with pdf" begin
        x_test = 0.3
        @test logpdf(result, x_test) ≈ log(pdf(result, x_test))
    end

    @testset "logpdf outside support returns -Inf" begin
        @test logpdf(result, -0.1) == -Inf
    end

    @testset "pdf on arbitrary bounds" begin
        data_ab = rand(Beta(2, 5), 500) .* 10.0
        result_ab = betakde(data_ab; lower=0.0, upper=10.0)
        @test pdf(result_ab, 5.0) > 0.0
        @test pdf(result_ab, -1.0) == 0.0
        @test pdf(result_ab, 11.0) == 0.0
    end
end

@testset "Summary statistics (mean, var, quantile)" begin
    using Statistics: mean, var, quantile

    # Beta(2,5): theoretical mean = 2/7 ≈ 0.2857, var = 10/343 ≈ 0.0292
    data = rand(Beta(2, 5), 10000)
    result = betakde(data)

    @testset "mean close to theoretical" begin
        @test isapprox(mean(result), 2 / 7; atol=0.02)
    end

    @testset "var close to theoretical" begin
        @test isapprox(var(result), 10 / 343; atol=0.005)
    end

    @testset "median (quantile 0.5) reasonable" begin
        q50 = quantile(result, 0.5)
        @test 0.1 < q50 < 0.5  # Beta(2,5) median ≈ 0.257
    end

    @testset "quantile monotone" begin
        q25 = quantile(result, 0.25)
        q50 = quantile(result, 0.50)
        q75 = quantile(result, 0.75)
        @test q25 < q50 < q75
    end

    @testset "quantile boundary values" begin
        @test quantile(result, 0.0) ≈ result.x[1]
        @test quantile(result, 1.0) ≈ result.x[end] atol=1e-4
    end

    @testset "quantile invalid p" begin
        @test_throws ArgumentError quantile(result, -0.1)
        @test_throws ArgumentError quantile(result, 1.1)
    end
end

@testset "DensityInterface extension" begin
    using DensityInterface

    data = rand(Beta(2, 5), 500)
    result = betakde(data)

    @test DensityKind(result) === DensityInterface.IsDensity()
    @test densityof(result, 0.3) == pdf(result, 0.3)
    @test logdensityof(result, 0.3) == logpdf(result, 0.3)
    @test logdensityof(result, 0.3) ≈ log(densityof(result, 0.3))
    @test densityof(result, -0.1) == 0.0
    @test logdensityof(result, -0.1) == -Inf
end
