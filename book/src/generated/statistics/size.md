| test | data (null hypothesis true) | rejection rate at 5% |
|---|---|---:|
| one-sample $t$ | normal, $n = 10$ | 0.0507 |
| one-sample $t$ | exponential, $n = 10$ | 0.1018 |
| one-sample $t$ | exponential, $n = 100$ | 0.0567 |
| Student's two-sample $t$ | normal, $n = 10, 10$, equal variances | 0.0513 |
| Student's two-sample $t$ | normal, $n = 10, 40$, $\sigma = 3, 1$ | 0.2554 |
| Welch's $t$ | normal, $n = 10, 40$, $\sigma = 3, 1$ | 0.0517 |
| paired $t$ | normal pairs, correlation 0.8, $n = 15$ | 0.0514 |
| Mann-Whitney $U$ | normal, $n = 10, 10$ | 0.0440 |
| Mann-Whitney $U$ | Cauchy, $n = 20, 30$ | 0.0476 |
| Kolmogorov-Smirnov, one sample | uniform, $n = 50$ | 0.0476 |
| Kolmogorov-Smirnov, two samples | normal, $n = 50, 50$ | 0.0386 |
| $\chi^2$ goodness of fit | 60 rolls of a fair die | 0.0469 |
| $\chi^2$ independence | $2 \times 3$ table, 200 independent cases | 0.0515 |
| one-way ANOVA | three normal groups of 8 | 0.0508 |
| Pearson $r$ | independent normals, $n = 20$ | 0.0509 |
| Spearman $\rho$ | independent normals, $n = 20$ | 0.0508 |
| Kendall $\tau_b$ | independent normals, $n = 20$ | 0.0471 |

Each rate is over 20000 simulated data sets; its Monte Carlo standard error is about 0.0015.
