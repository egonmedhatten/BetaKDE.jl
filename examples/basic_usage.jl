using BetaKDE
using Distributions

# ==========================================================================
# Example 1: Basic usage — automatic bandwidth
# ==========================================================================
data = rand(Beta(2, 5), 500)
result = betakde(data)

println("=== Beta(2, 5) — Automatic bandwidth ===")
println("Bandwidth (auto): ", round(result.bandwidth; digits=4))
println("Grid points:      ", length(result.x))
println("Peak density:     ", round(maximum(result.density); digits=3))

# Verify integration to 1 (guaranteed by normalize=true default)
dx = diff(result.x)
integral = sum(dx .* (result.density[1:end-1] .+ result.density[2:end]) ./ 2)
println("Integral ≈        ", round(integral; digits=6))

# ==========================================================================
# Example 2: Custom bandwidth and grid size
# ==========================================================================
println("\n=== Custom bandwidth (h=0.05) and npoints=256 ===")
result2 = betakde(data; bw=0.05, npoints=256)
println("Bandwidth:  ", result2.bandwidth)
println("Grid points: ", length(result2.x))

# ==========================================================================
# Example 3: Challenging distributions (fallback heuristic)
# ==========================================================================
println("\n=== U-shaped: Beta(0.5, 0.5) ===")
data_u = rand(Beta(0.5, 0.5), 300)
result_u = betakde(data_u)
println("Bandwidth: ", round(result_u.bandwidth; digits=4))

println("\n=== J-shaped: Beta(0.3, 2.0) ===")
data_j = rand(Beta(0.3, 2.0), 300)
result_j = betakde(data_j)
println("Bandwidth: ", round(result_j.bandwidth; digits=4))

# ==========================================================================
# Example 4: Accessing the bandwidth selector directly
# ==========================================================================
println("\n=== Direct bandwidth selection ===")
data_nice = rand(Beta(3, 5), 1000)
h, is_fallback = BetaKDE.bw_beta_rot(data_nice)
println("h = ", round(h; digits=5), "  fallback = ", is_fallback)

# ==========================================================================
# Example 5: Normalized vs raw output
# ==========================================================================
println("\n=== normalize=true vs normalize=false ===")
r_norm = betakde(data; normalize=true)
r_raw  = betakde(data; normalize=false)
dx = diff(r_raw.x)
raw_integral = sum(dx .* (r_raw.density[1:end-1] .+ r_raw.density[2:end]) ./ 2)
println("Normalized integral: ", round(sum(diff(r_norm.x) .* (r_norm.density[1:end-1] .+ r_norm.density[2:end]) ./ 2); digits=6))
println("Raw integral:        ", round(raw_integral; digits=6))
