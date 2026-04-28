module BetaKDE

using Distributions
using SpecialFunctions: loggamma
using Statistics: mean, var
using RecipesBase

export betakde, BetaKDEUnivariate

"""
    BetaKDEUnivariate

Result type for univariate Beta kernel density estimation.
Fields mirror `KernelDensity.jl`'s `UnivariateKDE`:
- `x`: evaluation grid points
- `density`: estimated density values
- `bandwidth`: bandwidth used
"""
struct BetaKDEUnivariate
    x::Vector{Float64}
    density::Vector{Float64}
    bandwidth::Float64
end

# --------------------------------------------------------------------------
# Internal helpers: Beta distribution moments
# --------------------------------------------------------------------------

_beta_variance(a::Real, b::Real) = a * b / ((a + b)^2 * (a + b + 1))

function _beta_skewness(a::Real, b::Real)
    return 2(b - a) * sqrt(a + b + 1) / ((a + b + 2) * sqrt(a * b))
end

function _beta_kurtosis(a::Real, b::Real)
    return 6 * ((a - b)^2 * (a + b + 1) - a * b * (a + b + 2)) /
           (a * b * (a + b + 2) * (a + b + 3))
end

# --------------------------------------------------------------------------
# Boundary correction (Chen 1999)
# --------------------------------------------------------------------------

function _rho(x::Real, h::Real)
    term = max(4h^4 + 6h^2 + 2.25 - x^2 - x / h, 0.0)
    return (2h^2 + 2.5) - sqrt(term)
end

# --------------------------------------------------------------------------
# O(1) Rule-of-Thumb Bandwidth Selector
# --------------------------------------------------------------------------

"""
    bw_beta_rot(data::AbstractVector{<:Real}) -> (h::Float64, is_fallback::Bool)

Closed-form O(1) rule-of-thumb bandwidth selector for Beta KDE.
Returns the bandwidth `h` and a flag indicating whether the fallback heuristic was used.
"""
function bw_beta_rot(data::AbstractVector{<:Real})
    data_filtered = filter(xi -> 0.0 < xi < 1.0, data)
    n = length(data_filtered)
    n < 2 && return (1e-5, true)

    μ = mean(data_filtered)
    σ² = var(data_filtered; corrected=false)

    # Guard: degenerate cases
    if σ² == 0.0 || σ² >= μ * (1.0 - μ)
        return _fallback_bandwidth(2.0, 2.0, n)
    end

    # Method of moments
    λ = (μ * (1.0 - μ)) / σ² - 1.0
    a = μ * λ
    b = (1.0 - μ) * λ

    # Constraint check for MISE rule validity
    if !(a > 1.5 && b > 1.5 && a + b > 3.0)
        return _fallback_bandwidth(a, b, n)
    end

    # Denominator factor checks
    denom_t1 = (a - 1.0) * (b - 1.0)
    denom_t2 = 6.0 - 4.0 * b + a * (3.0 * b - 4.0)
    if denom_t1 <= 0.0 || denom_t2 <= 0.0
        return _fallback_bandwidth(a, b, n)
    end

    # MISE formula in log-space
    log_num = (log(2a + 2b - 5) + log(2a + 2b - 3) +
               loggamma(2a + 2b - 6) +
               loggamma(a) + loggamma(b) +
               loggamma(a - 0.5) + loggamma(b - 0.5))

    log_denom = (log(denom_t1) + log(denom_t2) +
                 loggamma(2a - 3) + loggamma(2b - 3) +
                 loggamma(a + b) + loggamma(a + b - 1))

    log_factor = log(2) + log(n) + 0.5 * log(π)

    log_h = (2.0 / 5.0) * (log_num - log_denom - log_factor)
    h = exp(log_h)

    # Final validity check
    if !isfinite(h) || h <= 0.0 || h >= 1.0
        return _fallback_bandwidth(a, b, n)
    end

    return (h, false)
end

function _fallback_bandwidth(a::Real, b::Real, n::Int)
    s = sqrt(_beta_variance(a, b))
    if s <= 0.0
        return (1e-5, true)
    end
    correction = 1.0 + abs(_beta_skewness(a, b)) + abs(_beta_kurtosis(a, b))
    h = (s / correction) * n^(-0.4)
    h = clamp(h, 1e-5, 1.0 - 1e-5)
    return (h, true)
end

# --------------------------------------------------------------------------
# Main Estimator
# --------------------------------------------------------------------------

"""
    betakde(data; bw=nothing, npoints=512, normalize=true) -> BetaKDEUnivariate

Compute a Beta kernel density estimate of `data` on [0, 1].

# Arguments
- `data`: vector of observations (values outside (0,1) are clamped).
- `bw`: bandwidth (Float64). If `nothing`, selected automatically via `bw_beta_rot`.
- `npoints`: number of equally-spaced evaluation grid points (default 512).
- `normalize`: if `true` (default), rescale the density so it integrates to 1.

# Returns
A `BetaKDEUnivariate` with fields `x`, `density`, and `bandwidth`.
"""
function betakde(data::AbstractVector{<:Real}; bw=nothing, npoints::Int=512, normalize::Bool=true)
    ε = 1e-10
    data_c = clamp.(Float64.(data), ε, 1.0 - ε)

    # Bandwidth selection
    h::Float64 = if bw === nothing
        bw_beta_rot(data_c)[1]
    else
        Float64(bw)
    end

    # Evaluation grid
    x_grid = range(0.0, 1.0; length=npoints)
    density = Vector{Float64}(undef, npoints)
    n = length(data_c)

    # Kernel evaluation
    @inbounds for i in eachindex(x_grid)
        x = x_grid[i]

        # Compute kernel shape parameters with boundary correction
        α = x / h
        β_p = (1.0 - x) / h

        if x < 2h
            α = _rho(x, h)
        end
        if x > 1.0 - 2h
            β_p = _rho(1.0 - x, h)
        end

        # Ensure valid Beta parameters
        α = max(α, 1e-10)
        β_p = max(β_p, 1e-10)

        # Evaluate kernel at all data points and average
        kern = Beta(α, β_p)
        s = 0.0
        for j in eachindex(data_c)
            s += pdf(kern, data_c[j])
        end
        density[i] = s / n
    end

    # Post-hoc normalization via trapezoidal rule
    if normalize
        dx = step(x_grid)
        integral = dx * ((density[1] + density[end]) / 2 + sum(@view density[2:end-1]))
        if integral > 0.0
            density ./= integral
        end
    end

    return BetaKDEUnivariate(collect(x_grid), density, h)
end

# --------------------------------------------------------------------------
# Plotting Recipe
# --------------------------------------------------------------------------

@recipe function f(k::BetaKDEUnivariate)
    xguide --> "x"
    yguide --> "Density"
    label --> "Beta KDE"
    seriestype --> :path
    k.x, k.density
end

end # module BetaKDE
