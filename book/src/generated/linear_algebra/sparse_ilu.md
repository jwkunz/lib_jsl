| matrix (n = 1600) | preconditioner | GMRES(30) iterations | nnz of the factors | final relative residual |
|---|---:|---:|---:|---:|
| convection-diffusion (Péclet 0.3, 0.1) | none | 265 | — | 1e-8 |
| convection-diffusion (Péclet 0.3, 0.1) | Jacobi | 265 | 1600 | 1e-8 |
| convection-diffusion (Péclet 0.3, 0.1) | ILU(0) | 31 | 7840 | 8e-9 |
| convection-diffusion (Péclet 0.3, 0.1) | ILUT(τ = 1e-2, p = 10) | 17 | 13770 | 7e-9 |
| convection-diffusion (Péclet 0.3, 0.1) | ILUT(τ = 1e-4, p = 30) | 5 | 63209 | 5e-10 |
| convection-diffusion (Péclet 0.3, 0.1) | SciPy, none | 265 | — | 1e-8 |
| convection-diffusion (Péclet 0.3, 0.1) | SciPy `spilu(drop_tol = 1e-2)` | 9 | 32476 | 3e-9 |
| convection-diffusion (Péclet 0.3, 0.1) | SciPy `spilu(drop_tol = 1e-4)` | 8 | 53358 | 2e-9 |
| convection-dominated (Péclet 1.4, 0.9) | none | 190 | — | 9e-9 |
| convection-dominated (Péclet 1.4, 0.9) | Jacobi | 190 | 1600 | 9e-9 |
| convection-dominated (Péclet 1.4, 0.9) | ILU(0) | 12 | 7840 | 6e-9 |
| convection-dominated (Péclet 1.4, 0.9) | ILUT(τ = 1e-2, p = 10) | 12 | 9400 | 5e-9 |
| convection-dominated (Péclet 1.4, 0.9) | ILUT(τ = 1e-4, p = 30) | 4 | 30000 | 2e-9 |
| convection-dominated (Péclet 1.4, 0.9) | SciPy, none | 190 | — | 9e-9 |
| convection-dominated (Péclet 1.4, 0.9) | SciPy `spilu(drop_tol = 1e-2)` | 5 | 28436 | 1e-8 |
| convection-dominated (Péclet 1.4, 0.9) | SciPy `spilu(drop_tol = 1e-4)` | 3 | 39090 | 1e-11 |
