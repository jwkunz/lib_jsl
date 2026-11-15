| convection-diffusion, n = 900 | augmentation vectors | products with A (library / SciPy) | final relative residual (library / SciPy) | solution, relative difference from SciPy |
|---|---:|---:|---:|---:|
| γ = 0.5 | 3 | 205 / 205 | 8.8e-9 / 8.8e-9 | 5e-16 |
| γ = 0.9 | 3 | 163 / 163 | 9.4e-9 / 9.4e-9 | 2e-16 |
| γ = 0.5 | 0 (restarted GMRES(30)) | 155 / 155 | 9.4e-9 / 9.4e-9 | 3e-16 |
| γ = 0.9 | 0 (restarted GMRES(30)) | 113 / 113 | 9.7e-9 / 9.7e-9 | 2e-16 |
