| computation (SciPy reference) | largest difference (relative) |
|---|---:|
| `linregress`: slope, intercept, r, p-value and both standard errors | 5e-15 |
| `theilslopes` with its 95 % interval | 0 |
| `theilslopes` with its 90 % interval | 0 |
| `theilslopes`, tied x and y values | 0 |
| `siegelslopes` | 0 |
| `trim_mean`, cutting 0, 10, 25 and 40 % | 4e-16 |
| `zscore`, ddof 0 and 1 | 0 |
| `mode` of the rounded values | 0 |
| `circmean`, `circvar`, `circstd` in radians | 4e-16 |
| the same in degrees | 3e-16 |
| `boxcox`: λ = 0.012088181 against SciPy's 0.012088107, absolute difference (log-likelihood 5e-14 higher than at SciPy's λ) | 7e-8 |
| `boxcox`: the transformed data | 9e-8 |
| `pdist`, all 13 metrics, 7 points in 4 dimensions | 7e-15 |
| `cdist`, 3 × 4 points | 0 |
