| computation (SciPy reference) | `lib_jsl_linalg` | `lib_jsl_linalg_reference` |
|---|---:|---:|
| solve, 40 x 40 (`linalg.solve`) | 5e-15 | 2e-14 (pivoted elimination), 2e-14 (Doolittle LU) |
| inverse (`linalg.inv`) | 3e-15 |  |
| determinant (`linalg.det`) | 1e-14 | 2e-14 |
| Cholesky factor, 30 x 30 (`linalg.cholesky`) | 2e-16 | 2e-16 |
| symmetric eigenvalues, 30 x 30 (`linalg.eigh`) | 3e-15 | 7e-15 (Jacobi), 1e-14 (shifted QR) |
| their eigenvectors, up to sign | 8e-15 |  |
| general eigenvalues, 25 x 25, complex pairs (`linalg.eigvals`) | 2e-15 |  |
| QR, R factor up to signs, 50 x 20 (`linalg.qr`) |  | 7e-16 (Householder), 2e-16 (modified Gram-Schmidt) |
| singular values, 50 x 20 (`linalg.svd`) | 4e-15 | 4e-15 (through AᵀA) |
| singular vectors, up to sign | 5e-14 |  |
| least squares, 50 x 20 (`linalg.lstsq`) | 4e-15 | 3e-15 (QR), 4e-15 (normal equations) |
| minimum-norm least squares, rank 12 of 20 (`linalg.lstsq`) | 3e-15, rank 12 (SciPy 12) |  |
| Hermitian eigenvalues, complex 20 x 20 (`linalg.eigh`) | 3e-15 |  |
| complex solve, 20 x 20 (`linalg.solve`) | 2e-15 |  |
