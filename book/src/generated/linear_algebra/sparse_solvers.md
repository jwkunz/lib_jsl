| solver, n = 900, rtol = 10⁻⁸ | iterations (library / SciPy) | error against the direct solution (library / SciPy) |
|---|---:|---:|
| `cg`, 2-D Laplacian | 87 / 87 | 6e-10 / 6e-10 |
| `minres`, 2-D Laplacian | 86 / 53 | 1e-9 / 8e-6 |
| `cg`, varying diagonal | 31 / 31 | 5e-8 / 5e-8 |
| `minres`, varying diagonal | 30 / 26 | 9e-8 / 4e-7 |
| `cg` with Jacobi preconditioning, varying diagonal | 10 / 10 | 9e-9 / — |
| `bicgstab`, 2-D Laplacian | 63 / 61 | 1e-9 / 5e-9 |
| `gmres` (restart 30), 2-D Laplacian | 118 / 118 | 1e-8 / 1e-8 |
| `bicgstab`, convection-diffusion | 57 / 56 | 6e-9 / 6e-9 |
| `gmres` (restart 30), convection-diffusion | 210 / 210 | 6e-9 / 6e-9 |
