| function | test points | median relative error | largest relative error | at arguments |
|---|---:|---:|---:|---|
| `normal_cdf` | 6 | 0 | 1e-15 | (-3) |
| `normal_quantile` | 5 | 6e-16 | 2e-12 | (0.999999) |
| `student_t_cdf` | 30 | 2e-16 | 5e-12 | (-1, 10000) |
| `student_t_quantile` | 20 | 1e-15 | 2e-14 | (0.3, 100) |
| `chi_square_quantile` | 24 | 1e-15 | 4e-12 | (0.999999, 1) |
| `chi_square_survival` | 4 | 0 | 7e-16 | (30, 4) |
| `f_cdf` | 12 | 1e-15 | 2e-14 | (1, 20, 50) |
| `poisson_pmf` | 5 | 2e-15 | 1e-13 | (1000, 1000) |
| `poisson_cdf` | 5 | 6e-16 | 6e-14 | (1000, 1000) |
| `binomial_pmf` | 5 | 1e-15 | 3e-13 | (400, 1000, 0.42) |
| `binomial_cdf` | 5 | 3e-15 | 2e-13 | (400, 1000, 0.42) |
