| Function and problem | Library against SciPy |
|---|---:|
| `minimize_trust_krylov`: 4-D Rosenbrock from (−1.2, 1, −0.5, 0.8) | x differs by 2e-11; iterations 48 vs 46, Hessian-vector products 163 vs 146 |
| `minimize_trust_krylov`: 30-D Rosenbrock from (−1.2, 1, …) | x differs by 1e-11; iterations 133 vs 119, Hessian-vector products 1018 vs 714 |
| `least_squares_lsmr`: Broyden tridiagonal system, n = 100, sparse Jacobian | x differs by 1e-16; cost 6.3e-19 vs 6.3e-19; evaluations 5 vs 5 |
| `least_squares_lsmr`: exponential fit, free | x differs by 4e-16; evaluations 7 vs 7 |
| `least_squares_lsmr`: exponential fit, rate bounded by 0.5 | x differs by 0; evaluations 8 vs 8 |
| `least_squares_lsmr`: exponential fit, soft-L1 loss | x differs by 0; evaluations 6 vs 6 |
| `lsmr`, 60 × 20 sparse, damping 0 | x differs by 5e-16; istop 7 vs 7; iterations 20 vs 20 |
| `lsmr`, 60 × 20 sparse, damping 0.5 | x differs by 4e-16; istop 7 vs 7; iterations 20 vs 20 |
