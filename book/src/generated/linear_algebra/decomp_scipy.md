| function | test matrices | largest difference from SciPy (relative to the largest entry) |
|---|---:|---:|
| `qr` (full and economic) | square, tall, wide and complex matrices | 4e-15 |
| `qr` with column pivoting | the same; permutations identical | 4e-15 |
| `rq` (full and economic) | the same | 6e-15 |
| `polar` (right and left) | the same | 6e-15 |
| all four, a 6 × 6 matrix of rank 3 | the parts determined by the matrix: R, P, the first 3 pivots and columns of Q | 1e-15 |
| `hessenberg` | 6 × 6 real, 5 × 5 complex, 30 × 30 real | 8e-13 |
| `schur_real`: eigenvalues of the blocks | 6 × 6 and 30 × 30 real | 3e-14 |
| `schur_real`: ‖Z T Zᵀ − A‖ / max \|A\| | the same (2 × 2 blocks standardised) | 3e-15 |
| `ldl` (lower and upper) | definite, indefinite, zero-diagonal, Hermitian and complex symmetric; 13 2 × 2 pivots | 6e-16 |
| `subspace_angles` (relative) | real and complex subspaces | 2e-15 |
| `subspace_angles`, angles near 10⁻⁷ (relative) | a slightly perturbed subspace | 1e-8 |
| `funm` | exp, sin and x³ of a 6 × 6 real matrix; exp of a 5 × 5 complex one | 5e-15 |
