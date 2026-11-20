| Function | Difference from SciPy |
|---|---:|
| Singular normal (rank 2 in 4-D): logpdf, entropy; a point off the support | 4e-16 / 0; −∞ in both |
| `dunnett`, two-sided: statistics; p-values (QMC) | 0; 7e-5 (p = 0.6201, 0.0059) |
| `dunnett`, greater: statistics; p-values (QMC) | 0; 5e-5 (p = 0.3251, 0.0029) |
| `dunnett`, less: statistics; p-values (QMC) | 0; 2e-5 (p = 0.9168, 0.9998) |
| `MultivariateNormalQmc`, unscrambled Sobol, 16 samples: inverse transform; Box-Muller | 2e-15; 2e-16 |
| `PoissonDisk`, r = 0.05, `fill_space`: points (mean of 5 seeds), volume; surface | 250 vs 250; 346 vs 347 |
