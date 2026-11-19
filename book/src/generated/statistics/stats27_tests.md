| test or estimate | largest relative difference from SciPy |
|---|---:|
| `cramervonmises` (n = 10, 50) | 2e-14 |
| `cramervonmises_2samp`: exact (7 and 9, with ties), asymptotic (7 and 9; 25 and 30) | 2e-14 |
| `brunnermunzel` (three alternatives; t and normal) | 9e-16 |
| `StudentizedRange`: cdf, pdf, 95 % point (k = 2–10, ν = 5–∞) | 3e-13 |
| `tukey_hsd` (4 groups, unequal sizes; Tukey-Kramer and Games-Howell): differences, intervals | 3e-14 |
| `tukey_hsd`: p-values (absolute difference) | 3e-15 |
| `odds_ratio`, conditional and sample, with intervals for three alternatives (one table with a zero) | 4e-15 |
| `relative_risk` with Katz's interval | 8e-16 |
| `association`: Cramér's V, Tschuprow's T, Pearson's C (3 × 4 table) | 2e-16 |
