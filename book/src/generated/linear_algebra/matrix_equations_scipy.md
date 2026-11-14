| computation (SciPy reference) | difference | residual, library | residual, SciPy |
|---|---:|---:|---:|
| `expm`, 20 × 20, ‖A‖₁ ≈ 4.6 (`linalg.expm`) | 1e-16 |  |  |
| `expm`, 20 × 20, ‖A‖₁ ≈ 54, entries up to 4 × 10⁵ | 8e-16 |  |  |
| `expm`, complex 20 × 20 | 4e-16 |  |  |
| `solve_sylvester`, 20 × 20 and 15 × 15 | 6e-15 | 3e-15 | 3e-15 |
| `solve_continuous_lyapunov`, 20 × 20 | 3e-15 | 6e-15 | 4e-15 |
| `solve_continuous_lyapunov`, complex | 5e-15 | 4e-15 | 6e-15 |
| `solve_discrete_lyapunov`, 20 × 20, spectral radius 0.8 | 4e-15 | 3e-15 | 2e-15 |
| `solve_continuous_are`, 20 states, 4 inputs | 2e-13 | 1e-13 | 2e-13 |
| `solve_continuous_are`, complex | 4e-13 | 2e-13 | 3e-13 |
| `solve_discrete_are`, 20 states, 4 inputs | 3e-15 | 2e-16 | 3e-15 |
