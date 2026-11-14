| computation (n = 900 unless stated) | largest difference from SciPy (relative) |
|---|---:|
| `spsolve`, 2-D Laplacian | 4e-15 |
| `spsolve`, convection-diffusion | 4e-15 |
| `spsolve`, varying diagonal | 1e-15 |
| `lsqr`, random 200 × 80 (5 % filled), 46 / 46 iterations, against SciPy | 8e-12 |
| `lsqr`, the same, against the dense least-squares solution | 1e-9 |
| `eigsh`, the 4 largest eigenvalues of the 2-D Laplacian | 9e-15 |
| `eigsh`, the 4 smallest eigenvalues of the 2-D Laplacian | 1e-13 |
| `eigsh`, the 4 largest in magnitude eigenvalues of the 2-D Laplacian | 2e-15 |
| `svds`, the 4 largest singular values of the 200 × 80 matrix | 3e-15 |
