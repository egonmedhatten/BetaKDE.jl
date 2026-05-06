module BetaKDEDensityInterfaceExt

using BetaKDE
using DensityInterface
import Distributions: pdf, logpdf

@inline DensityInterface.DensityKind(::BetaKDEUnivariate) = DensityInterface.IsDensity()

DensityInterface.logdensityof(d::BetaKDEUnivariate, x) = logpdf(d, x)
DensityInterface.densityof(d::BetaKDEUnivariate, x) = pdf(d, x)

end # module
