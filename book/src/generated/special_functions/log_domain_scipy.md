| computation (SciPy reference) | largest difference (relative) |
|---|---:|
| `logsumexp` (overflowing, underflowing and ordinary terms; signed weights) | 0 |
| `log_ndtr` on −50, −49, …, 10 | 3e-15 |
| `expit` and `log_expit` on ±800, ±40, ±5, … | 1e-16 |
| `logit` from 10⁻¹² to 1 − 10⁻⁹ | 1e-16 |
| `polygamma`, orders 1–4, x from 0.1 to 55.5 | 5e-16 |
| `polygamma`, orders 1–4, x = −2.5 and −0.3 (cancellation in the shift; see the text) | 8e-12 |
| `multigammaln`, dimensions 1, 2, 3, 10 | 1e-15 |
| `bernoulli`, B₀ … B₃₀, against exact rationals (SciPy's own error: 2e-12) | 8e-16 |
| `euler`, E₀ … E₃₀, against exact integers (SciPy's own error: 1e-15) | 2e-15 |
| `gammaincinv`, a from 0.1 to 100, p from 10⁻¹⁰ to 1 − 10⁻⁶ | 5e-15 |
| `gammainccinv`, the same grid | 1e-14 |
| `betaincinv`, six (a, b) pairs, the same p | 4e-13 |
| `boxcox` and `boxcox1p` | 0 |
