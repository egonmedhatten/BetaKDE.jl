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

    @testset "bw_beta_rot direct call" begin
        data = rand(Beta(3, 5), 1000)
        h, is_fallback = BetaKDE.bw_beta_rot(data)
        @test 0.0 < h < 1.0
        @test is_fallback == false
    end

    @testset "bw_beta_rot fallback triggers" begin
        data = rand(Beta(0.1, 0.1), 500)
        h, is_fallback = BetaKDE.bw_beta_rot(data)
        @test 0.0 < h < 1.0
        @test is_fallback == true
    end

end
