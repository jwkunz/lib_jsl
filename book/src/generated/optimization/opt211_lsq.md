| Problem | x vs SciPy | nfev: library / SciPy | njev | status / info |
|---|---:|---:|---:|---:|
| dogbox: Rosenbrock, x₁ ≥ 1.5 | 0 | 6 / 6 | 6 / 6 | 1 / 1 |
| dogbox: Rosenbrock, −2 ≤ x ≤ (0.5, 2) | 0 | 12 / 12 | 10 / 10 | 1 / 1 |
| dogbox: exponential fit, rate ≤ 1.2 (active) | 2e-16 | 8 / 8 | 8 / 8 | 2 / 2 |
| dogbox: exponential fit with outliers, soft L1 loss | 4e-16 | 13 / 13 | 11 / 11 | 2 / 2 |
| `leastsq`: Rosenbrock, analytic Jacobian | 0 | 21 / 21 | 16 / 16 | 4 / 4 |
| `leastsq`: Rosenbrock, forward differences | 0 | 54 / 54 | — | 2 / 2 |
| `leastsq`: exponential fit, 30 residuals | 4e-9 | 38 / 38 | — | 1 / 1 |
| `leastsq`: x² + y² = 4, eˣ + y = 1 | 0 | 16 / 16 | — | 2 / 2 |
| `leastsq`: Powell's badly scaled system | 2e-13 | 53 / 53 | — | 2 / 2 |
| `root_lm`: the same system | 0 | 16 / — | — | converged |
