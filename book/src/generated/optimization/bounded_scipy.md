| problem (SciPy reference) | largest difference in x | difference in the objective | iterations (library / SciPy) | evaluations (library / SciPy) |
|---|---:|---:|---:|---:|
| `minimize_lbfgsb`: Rosenbrock, n = 10, three upper bounds active | 3e-7 | 3e-11 | 32 / 27 | 58 / 31 |
| `minimize_lbfgsb`: convex quadratic, n = 30, −0.5 ≤ xᵢ ≤ 0.5 | 1e-7 | 2e-15 | 25 / 23 | 26 / 24 |
| `least_squares_bounded`: exponential with outliers, linear loss | 0 | 0 |  | 10 / 10 |
| `least_squares_bounded`: exponential with outliers, soft_l1 loss | 0 | 4e-16 |  | 10 / 10 |
| `least_squares_bounded`: exponential with outliers, huber loss | 2e-15 | 0 |  | 10 / 10 |
| `least_squares_bounded`: exponential with outliers, cauchy loss | 9e-16 | 6e-17 |  | 12 / 12 |
| `least_squares_bounded`: exponential with outliers, arctan loss | 3e-15 | 3e-17 |  | 14 / 14 |
| `least_squares_bounded`: the same, decay rate ≤ 0.4 (active) | 0 | 0 |  | 9 / 9 |
| `lsq_linear` (BVLS): 40 × 12, −0.15 ≤ xᵢ ≤ 0.15 | 4e-16 | 9e-16 |  |  |
| `nnls`: 40 × 12 | 3e-16 | 0 |  |  |
