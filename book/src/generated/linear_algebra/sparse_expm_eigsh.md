| computation (SciPy reference), n = 900 | largest difference (relative) |
|---|---:|
| `expm_multiply`, heat equation e^(−tA) b, t = 0.01 | 1e-16 |
| `expm_multiply`, heat equation e^(−tA) b, t = 1 (dense `expm` times b: 6e-15) | 0 |
| `expm_multiply`, heat equation e^(−tA) b, t = 10 | 2e-15 |
| `expm_multiply`, heat equation e^(−tA) b, t = 40 | 6e-15 |
| `expm_multiply`, convection-diffusion, t = 0.5 | 0 |
| `expm_multiply`, convection-diffusion, t = 3 | 6e-16 |
| `eigsh` shift-invert, the 6 eigenvalues nearest σ = 2 (three double pairs) | 1e-16 |
| `eigsh` shift-invert, the 6 eigenvalues nearest σ = 5.5 (three double pairs) | 0 |
