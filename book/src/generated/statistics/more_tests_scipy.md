| test (SciPy reference) | statistic | p-value | SciPy's p-value |
|---|---:|---:|---:|
| `shapiro`, normal sample (n = 60) | 1e-10 | 1e-8 | 0.0912 |
| `shapiro`, skewed sample (n = 60) | 6e-11 | 3e-9 | 0.0000 |
| `shapiro`, small sample (n = 9) | 9e-11 | 2e-9 | 0.4375 |
| `shapiro`, tiny sample (n = 4) | 2e-10 | 9e-10 | 0.9303 |
| `jarque_bera`, normal | 1e-15 | 6e-15 | 0.0059 |
| `normaltest` (D'Agostino-Pearson), normal | 6e-16 | 2e-15 | 0.0104 |
| `skewtest`, normal | 4e-16 | 4e-15 | 0.0383 |
| `kurtosistest`, normal | 1e-15 | 5e-15 | 0.0278 |
| `jarque_bera`, skewed | 2e-15 | 4e-14 | 0.0000 |
| `normaltest` (D'Agostino-Pearson), skewed | 3e-16 | 4e-15 | 0.0000 |
| `skewtest`, skewed | 4e-16 | 7e-15 | 0.0000 |
| `kurtosistest`, skewed | 5e-16 | 3e-15 | 0.0108 |
| `levene` (median centred), three groups | 5e-16 | 9e-15 | 0.0325 |
| `bartlett`, three groups | 0 | 0 | 0.0009 |
| `fligner`, three groups | 3e-16 | 9e-16 | 0.0370 |
| `kruskal`, three groups | 0 | 0 | 0.0987 |
| `friedman`, 4 treatments × 10 blocks, with ties | 0 | 0 | 0.8451 |
| `wilcoxon`, n = 20, exact, two-sided | 0 | 0 | 0.1327 |
| `wilcoxon`, n = 20, exact, less | 0 | 0 | 0.9385 |
| `wilcoxon`, n = 20, exact, greater | 0 | 0 | 0.0664 |
| `wilcoxon`, n = 12 with ties (all sign changes) | 0 | 0 | 0.0957 |
| `wilcoxon`, n = 40 with ties (normal approximation) | 0 | 1e-16 | 0.7587 |
| `fisher_exact` [[8, 2], [1, 5]], two-sided | 0 | 8e-16 | 0.0350 |
| `fisher_exact` [[8, 2], [1, 5]], less | 0 | 0 | 0.9991 |
| `fisher_exact` [[8, 2], [1, 5]], greater | 0 | 7e-16 | 0.0245 |
| `fisher_exact` [[12, 5], [9, 14]], two-sided | 0 | 3e-15 | 0.0624 |
| `fisher_exact` [[12, 5], [9, 14]], less | 0 | 0 | 0.9897 |
| `fisher_exact` [[12, 5], [9, 14]], greater | 0 | 1e-15 | 0.0488 |
| `fisher_exact` [[0, 7], [6, 3]], two-sided | 0 | 1e-15 | 0.0114 |
| `fisher_exact` [[0, 7], [6, 3]], less | 0 | 1e-15 | 0.0105 |
| `fisher_exact` [[0, 7], [6, 3]], greater | 0 | 0 | 1.0000 |
| `binomtest` k = 7, n = 20, p = 0.5 | 0 | 3e-15 | 0.2632 |
| `binomtest` k = 3, n = 40, p = 0.2 | 0 | 7e-15 | 0.0479 |
| `binomtest` k = 61, n = 100, p = 0.5 | 0 | 6e-14 | 0.0352 |
| `permutation_test`, difference of means, 6435 splits, two-sided | 3e-16 | 0 | 0.2794 |
| `permutation_test`, difference of means, 6435 splits, less | 3e-16 | 0 | 0.1397 |
| `permutation_test`, difference of means, 6435 splits, greater | 3e-16 | 0 | 0.8605 |
| `anderson_normal`, normal: statistic, critical values | 5e-14 | 0 |  |
| `anderson_normal`, skewed: statistic, critical values | 0 | 0 |  |
| `adjust_p_values`, BenjaminiHochberg (`false_discovery_control`) |  | 1e-16 |  |
| `adjust_p_values`, BenjaminiYekutieli (`false_discovery_control`) |  | 2e-16 |  |
