using BetaKDE
using Distributions
using Plots

# ==========================================================================
# Example 1: Simple Beta(2, 5) — right-skewed
# ==========================================================================
data = rand(Beta(2, 5), 500)
result = betakde(data)

p1 = plot(result; linewidth=2)
plot!(p1, x -> pdf(Beta(2, 5), x), 0, 1;
      label="True Beta(2,5)", ls=:dash, color=:red)
histogram!(p1, data; normalize=:pdf, alpha=0.3, label="Histogram", color=:grey)
title!(p1, "Beta(2, 5) — Right-skewed")

# ==========================================================================
# Example 2: Symmetric Beta(5, 5)
# ==========================================================================
data_sym = rand(Beta(5, 5), 500)
result_sym = betakde(data_sym)

p2 = plot(result_sym; linewidth=2)
plot!(p2, x -> pdf(Beta(5, 5), x), 0, 1;
      label="True Beta(5,5)", ls=:dash, color=:red)
histogram!(p2, data_sym; normalize=:pdf, alpha=0.3, label="Histogram", color=:grey)
title!(p2, "Beta(5, 5) — Symmetric")

# ==========================================================================
# Example 3: U-shaped Beta(0.5, 0.5) — tests fallback heuristic
# ==========================================================================
data_u = rand(Beta(0.5, 0.5), 500)
result_u = betakde(data_u)

p3 = plot(result_u; linewidth=2)
plot!(p3, x -> pdf(Beta(0.5, 0.5), x), 0, 1;
      label="True Beta(0.5,0.5)", ls=:dash, color=:red)
histogram!(p3, data_u; normalize=:pdf, alpha=0.3, label="Histogram", color=:grey)
title!(p3, "Beta(0.5, 0.5) — U-shaped (fallback)")

# ==========================================================================
# Example 4: J-shaped Beta(0.3, 2)
# ==========================================================================
data_j = rand(Beta(0.3, 2), 500)
result_j = betakde(data_j)

p4 = plot(result_j; linewidth=2)
plot!(p4, x -> pdf(Beta(0.3, 2), x), 0, 1;
      label="True Beta(0.3,2)", ls=:dash, color=:red)
histogram!(p4, data_j; normalize=:pdf, alpha=0.3, label="Histogram", color=:grey)
title!(p4, "Beta(0.3, 2) — J-shaped (fallback)")

# ==========================================================================
# Combined 2×2 panel
# ==========================================================================
p = plot(p1, p2, p3, p4; layout=(2, 2), size=(900, 700))
savefig(p, joinpath(@__DIR__, "betakde_examples.png"))
println("Saved: examples/betakde_examples.png")

# ==========================================================================
# Example 5: Effect of bandwidth on smoothness
# ==========================================================================
data_bw = rand(Beta(2, 5), 300)

p5 = histogram(data_bw; normalize=:pdf, alpha=0.2, label="Histogram", color=:grey)
for h in [0.01, 0.05, 0.1, 0.2]
    r = betakde(data_bw; bw=h)
    plot!(p5, r.x, r.density; label="h = $h", linewidth=2)
end
plot!(p5, x -> pdf(Beta(2, 5), x), 0, 1;
      label="True", ls=:dash, color=:black, linewidth=2)
title!(p5, "Effect of bandwidth")
xlabel!(p5, "x")
ylabel!(p5, "Density")
savefig(p5, joinpath(@__DIR__, "bandwidth_comparison.png"))
println("Saved: examples/bandwidth_comparison.png")

# ==========================================================================
# Example 6: Normalized vs unnormalized
# ==========================================================================
data_norm = rand(Beta(2, 5), 500)
r_norm = betakde(data_norm; normalize=true)
r_raw = betakde(data_norm; normalize=false)

p6 = plot(r_norm.x, r_norm.density; label="normalize=true", linewidth=2)
plot!(p6, r_raw.x, r_raw.density; label="normalize=false", linewidth=2, ls=:dash)
title!(p6, "Normalized vs Raw density")
xlabel!(p6, "x")
ylabel!(p6, "Density")
savefig(p6, joinpath(@__DIR__, "normalization_comparison.png"))
println("Saved: examples/normalization_comparison.png")
