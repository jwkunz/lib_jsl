| integrand | values (relative difference) | evaluations (library / SciPy) | subintervals | error estimate (relative difference) |
|---|---:|---:|---:|---:|
| 5 components with a √\|x − 0.3\| kink, [0, 3], max norm | 0 | 819 / 819 | 20 / 20 | 0 |
| the same, 2-norm (the default) | 0 | 819 / 819 | 20 / 20 | 6e-13 |
| the same, with the kink as a break point | 0 | 1218 / 1218 | 30 / 30 | 1e-12 |
| the same, trapezoid rule, epsrel = 10⁻⁵ | 0 | 5073 / 5073 | 1015 / 1015 | 3e-12 |
| 3 Gaussians, [0, ∞) | 0 | 165 / 165 | 6 / 6 | 6e-9 |
| the same, (−∞, 0] | 0 | 195 / 195 | 7 / 7 | 5e-10 |
| the same, (−∞, ∞) | 0 | 330 / 330 | 12 / 12 | 8e-10 |
| Bessel integrals J₀ … J₇ at x = n, epsrel = 10⁻¹² | 0 | 189 / 189 | 5 / 5 | 5e-13 |
