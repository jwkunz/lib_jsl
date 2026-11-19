| Problem | Result | Difference from SciPy | Notes |
|---|---:|---:|---:|
| `isotonic_regression`, increasing | 60 points, 14 blocks | 2e-15 (values), 7e-15 (weights) | identical blocks |
| `isotonic_regression`, decreasing | 60 points, 1 block | 9e-16 (values), 0 (weights) | identical blocks |
| `approx_fprime`, Rosenbrock in 4-D | 4 components | 0 | relative; same steps as SciPy |
| `check_grad`, Rosenbrock in 4-D | 1.615402e-5 | 9e-10 | relative |
| `brute`, Himmelblau, 20 × 20 grid | (2.894737, 1.842105) | 0 | same grid point and value |
| `brute`, then Nelder-Mead | (3.0000000103, 2.0000000129) | 4e-5 | SciPy's `fmin` stops at xtol = 10⁻⁴; the library is 1e-8 from (3, 2) |
