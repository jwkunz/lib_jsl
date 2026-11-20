| Method and problem | Library: distance from the optimum | SciPy: distance from the optimum |
|---|---:|---:|
| COBYLA: x₀x₁ on the half disc, from (1, 1) | 2e-9 (75 evaluations) | 1e0 (21 evaluations) |
| trust-constr: x₀x₁ on the half disc, from (1, 1) | 6e-8 (234 iterations) | 1e-8 (36 iterations) |
| COBYLA: Rosenbrock in the disc x² + y² ≤ 1.5 | 3e-1 (1000 evaluations) | 5e-2 (1000 evaluations) |
| … with up to 10 000 evaluations | 1e-9 (4905 evaluations) | 2e-9 (1304 evaluations) |
| trust-constr: Rosenbrock in the disc x² + y² ≤ 1.5 | 4e-7 (254 iterations) | 5e-7 (137 iterations) |
| trust-constr: HS071 (4 variables, 1 equality, 9 inequalities) | 2e-6 (228 iterations) | 4e-7 (301 iterations) |
