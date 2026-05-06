module BetaKDE

using Distributions
using SpecialFunctions: loggamma, logbeta
using Statistics: mean, var
using RecipesBase

export betakde, BetaKDEUnivariate, bw_HS

"""
    BetaKDEUnivariate

Result type for univariate Beta kernel density estimation.
Fields mirror `KernelDensity.jl`'s `UnivariateKDE`:
- `x`: evaluation grid points
- `density`: estimated density values
- `bandwidth`: bandwidth used
- `lower`: lower bound of the support
- `upper`: upper bound of the support
"""
struct BetaKDEUnivariate
    x::Vector{Float64}
    density::Vector{Float64}
    bandwidth::Float64
    lower::Float64
    upper::Float64
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
# O(1) Rule-of-Thumb Bandwidth Selector (Hallberg Szabadváry, 2026)
# --------------------------------------------------------------------------

"""
    bw_HS(data::AbstractVector{<:Real}) -> (h::Float64, is_fallback::Bool)

Closed-form O(1) MISE-optimal bandwidth selector for Beta KDE
(Hallberg Szabadváry, 2026). Named `HS` to match the R `kdensity` package convention.

Returns the bandwidth `h` and a flag indicating whether the fallback heuristic was used.
"""
function bw_HS(data::AbstractVector{<:Real})
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

# Backward-compatible alias
const bw_beta_rot = bw_HS

# --------------------------------------------------------------------------
# Main Estimator
# --------------------------------------------------------------------------

"""
    betakde(data; bw=:HS, npoints=512, normalize=true, lower=0.0, upper=1.0) -> BetaKDEUnivariate

Compute a Beta kernel density estimate of `data` on [`lower`, `upper`].

# Arguments
- `data`: vector of observations (values outside (lower, upper) are clamped).
- `bw`: bandwidth. Either a numeric value, or `:HS` (default) for automatic
  selection via the Hallberg Szabadváry (2026) rule-of-thumb.
- `npoints`: number of equally-spaced evaluation grid points (default 512).
- `normalize`: if `true` (default), rescale the density so it integrates to 1.
- `lower`: lower bound of the support (default 0.0).
- `upper`: upper bound of the support (default 1.0).

# Returns
A `BetaKDEUnivariate` with fields `x`, `density`, `bandwidth`, `lower`, `upper`.
"""
function betakde(data::AbstractVector{<:Real}; bw::Union{Symbol,Real}=:HS, npoints::Int=512, normalize::Bool=true, lower::Real=0.0, upper::Real=1.0)
    lower < upper || throw(ArgumentError("lower must be less than upper"))
    lo = Float64(lower)
    hi = Float64(upper)
    span = hi - lo

    ε = 1e-10
    # Transform data to [0, 1]
    data_c = clamp.((Float64.(data) .- lo) ./ span, ε, 1.0 - ε)

    # Bandwidth selection
    h::Float64 = if bw === :HS
        bw_HS(data_c)[1]
    else
        Float64(bw)
    end

    # Evaluation grid
    x_grid = range(0.0, 1.0; length=npoints)
    density = Vector{Float64}(undef, npoints)
    n = length(data_c)

    # Pre-compute log(x) and log(1-x) for all data points (reused at every grid point)
    log_data = log.(data_c)
    log_1mdata = log.(1.0 .- data_c)

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

        # Evaluate kernel at all data points and average.
        # Compute logbeta once per grid point (constant across data points).
        lb = logbeta(α, β_p)
        am1 = α - 1.0
        bm1 = β_p - 1.0
        s = 0.0
        @simd for j in eachindex(data_c)
            s += exp(am1 * log_data[j] + bm1 * log_1mdata[j] - lb)
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

    # Transform grid and density back to [lower, upper]
    x_out = collect(x_grid) .* span .+ lo
    density ./= span

    return BetaKDEUnivariate(x_out, density, h, lo, hi)
end

# --------------------------------------------------------------------------
# Point Evaluation (pdf / logpdf via linear interpolation)
# --------------------------------------------------------------------------

"""
    _interp_density(k::BetaKDEUnivariate, x::Real) -> Float64

Linear interpolation of the stored gridded density at point `x`.
Returns 0.0 for `x` outside `[k.lower, k.upper]`.
"""
function _interp_density(k::BetaKDEUnivariate, x::Real)
    xf = Float64(x)
    (xf < k.lower || xf > k.upper) && return 0.0
    n = length(k.x)
    # Fractional index
    t = (xf - k.lower) / (k.upper - k.lower) * (n - 1)
    i = floor(Int, t)
    i = clamp(i, 0, n - 2)
    w = t - i
    return (1.0 - w) * k.density[i+1] + w * k.density[i+2]
end

import Distributions: pdf, logpdf

"""
    pdf(k::BetaKDEUnivariate, x::Real)

Evaluate the estimated density at point `x` via linear interpolation on the grid.
"""
pdf(k::BetaKDEUnivariate, x::Real) = _interp_density(k, x)

"""
    logpdf(k::BetaKDEUnivariate, x::Real)

Log-density at point `x`. Returns `-Inf` for `x` outside the support.
"""
function logpdf(k::BetaKDEUnivariate, x::Real)
    d = _interp_density(k, x)
    return d > 0.0 ? log(d) : -Inf
end

# --------------------------------------------------------------------------
# Summary Statistics (computed from gridded density)
# --------------------------------------------------------------------------

import Statistics: mean, var, quantile

"""
    mean(k::BetaKDEUnivariate)

Mean of the estimated density, computed via trapezoidal integration.
"""
function mean(k::BetaKDEUnivariate)
    dx = k.x[2] - k.x[1]
    xf = k.x .* k.density
    return dx * ((xf[1] + xf[end]) / 2 + sum(@view xf[2:end-1]))
end

"""
    var(k::BetaKDEUnivariate)

Variance of the estimated density, computed via trapezoidal integration.
"""
function var(k::BetaKDEUnivariate)
    μ = mean(k)
    dx = k.x[2] - k.x[1]
    x2f = ((k.x .- μ) .^ 2) .* k.density
    return dx * ((x2f[1] + x2f[end]) / 2 + sum(@view x2f[2:end-1]))
end

"""
    quantile(k::BetaKDEUnivariate, p::Real)

`p`-th quantile of the estimated density via cumulative trapezoidal integration.
"""
function quantile(k::BetaKDEUnivariate, p::Real)
    (0.0 <= p <= 1.0) || throw(ArgumentError("p must be in [0, 1]"))
    n = length(k.x)
    dx = k.x[2] - k.x[1]
    # Build cumulative distribution
    cum = 0.0
    @inbounds for i in 2:n
        cum += dx * (k.density[i-1] + k.density[i]) / 2
        if cum >= p
            # Linear interpolation between x[i-1] and x[i]
            prev_cum = cum - dx * (k.density[i-1] + k.density[i]) / 2
            frac = (p - prev_cum) / (cum - prev_cum)
            return k.x[i-1] + frac * dx
        end
    end
    return k.x[end]
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
