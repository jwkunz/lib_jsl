| computation (SciPy / NumPy reference) | largest difference (relative) |
|---|---:|
| `sqrtm`, symmetric positive definite 12 × 12 | 2e-15 |
| `logm`, symmetric positive definite 12 × 12 | 4e-15 |
| `fractional_matrix_power` (p = 0.3), symmetric positive definite 12 × 12 | 8e-16 |
| `sqrtm`, general 12 × 12 | 2e-15 |
| `logm`, general 12 × 12 | 4e-15 |
| `fractional_matrix_power` (p = 0.3), general 12 × 12 | 4e-15 |
| `sqrtm`, complex 10 × 10 | 2e-15 |
| `logm`, complex 10 × 10 | 2e-15 |
| `fractional_matrix_power` (p = 0.3), complex 10 × 10 | 2e-15 |
| `pinv`, 15 × 9 of rank 4 | 2e-15 |
| `matrix_rank`, 15 × 9 of rank 4: 4 (NumPy 4) | 0 |
| `null_space`: projector onto the 5-dimensional null space | 2e-15 |
| `orth`: projector onto the 4-dimensional range | 6e-16 |
| `pinv`, 20 × 7 of full rank | 4e-15 |
| `norm`, all eight orders, gen | 2e-15 |
| `norm`, all eight orders, lowrank | 6e-16 |
| `norm`, all eight orders, tall | 8e-16 |
| `cond` in the 2, −2, 1, ∞ and Frobenius norms, gen | 4e-15 |
| `cond` in the 2, −2, 1, ∞ and Frobenius norms, spd | 5e-16 |
