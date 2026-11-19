| distribution and quantities | largest relative difference from SciPy |
|---|---:|
| `Rice`: pdf, cdf, sf (ν/σ = 0, 1.7, 6) | 1e-13 |
| `Rice`: sf at x = 6 (10⁻⁵ to 10⁻⁹) against direct integration of the density (SciPy: 2e-7) | 9e-15 |
| `Rice`: ppf; mean, variance, skewness, kurtosis | 1e-9 |
| `Rice`, ν/σ = 40: cdf and sf in the tails (SciPy's moments are NaN here) | 4e-11 |
| `Rice::fit` (location 0): log-likelihood 5e-9 above SciPy's, whose optimiser stops earlier | 1e-5 |
| `Rice`, ν/σ = 40: mean against the limit ν + σ²/(2ν) | 5e-8 |
| `MultivariateNormal`: logpdf, entropy | 3e-16 |
| `MultivariateT`: logpdf, entropy | 3e-16 |
| `Dirichlet`: logpdf, mean, variance, entropy | 1e-15 |
| `Wishart`: logpdf, variance, entropy | 1e-16 |
| `InvWishart`: logpdf, mean, mode | 1e-16 |
| `Multinomial`: logpmf, covariance | 1e-15 |
