| problem | method | iterations (library / SciPy) | function evaluations | status | largest difference in x |
|---|---:|---:|---:|---:|---:|
| Rosenbrock, n = 4 | `trust-ncg` | 53 / 53 | 54 / 54 | 0 / 0 | 0 |
| Rosenbrock, n = 4 | `trust-exact` | 15 / 15 | 16 / 16 | 0 / 0 | 4e-16 |
| Rosenbrock, n = 4 | `dogleg` | 0 / 0 | 1 / 1 | 3 / 3 | 0 |
| Rosenbrock, n = 10 | `trust-ncg` | 86 / 86 | 87 / 87 | 0 / 0 | 0 |
| Rosenbrock, n = 10 | `trust-exact` | 30 / 30 | 31 / 31 | 0 / 0 | 0 |
| Rosenbrock, n = 10 | `dogleg` | 4 / 4 | 5 / 5 | 3 / 3 | 2e-16 |
| Rosenbrock, n = 4, near the minimum | `trust-ncg` | 12 / 12 | 13 / 13 | 0 / 0 | 0 |
| Rosenbrock, n = 4, near the minimum | `trust-exact` | 11 / 11 | 12 / 12 | 0 / 0 | 0 |
| Rosenbrock, n = 4, near the minimum | `dogleg` | 11 / 11 | 12 / 12 | 0 / 0 | 0 |
| x⁴ − 2x² + y² + xy/2 from near the saddle | `trust-ncg` | 5 / 5 | 6 / 6 | 0 / 0 | 0 |
| x⁴ − 2x² + y² + xy/2 from near the saddle | `trust-exact` | 4 / 4 | 5 / 5 | 0 / 0 | 0 |
