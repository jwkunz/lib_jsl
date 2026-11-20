| Surface, 400 noisy samples of sin 6x cos 4y (σ = 0.05) | Library against SciPy | Library error on a 21 × 21 grid | SciPy error |
|---|---:|---:|---:|
| `bisplrep_lsq`, knots 0, ¼, ½, ¾, 1 in each direction (SciPy's `task=-1`) | coefficients 3e-14, values 1e-14 | 0.501 | 0.501 |
| `bisplrep`, weights 1/σ, default s = m − √(2m) = 371.7 | fp 371.75 vs 371.82; 11 × 10 knots vs 11 × 11; surfaces differ by 8e-2 | 0.101 | 0.125 |
