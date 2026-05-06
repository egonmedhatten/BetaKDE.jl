module BetaKDEStatsBaseExt

using BetaKDE
import StatsBase

function StatsBase.fit(::Type{BetaKDEUnivariate}, data::AbstractVector{<:Real}; kw...)
    betakde(data; kw...)
end

end # module
