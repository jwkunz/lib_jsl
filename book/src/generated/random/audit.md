| distribution | method | sample mean (exact) | sample variance (exact) | KS statistic $D$ |
|---|---|---:|---:|---:|
| uniform $U(-1, 3)$ | scaling | 1.0031 (1.0000) | 1.3314 (1.3333) | 0.0024 |
| exponential, rate 2 | inversion | 0.5006 (0.5000) | 0.2540 (0.2500) | 0.0019 |
| Weibull, shape 1.5, scale 2 | inversion | 1.8151 (1.8055) | 1.5205 (1.5028) | 0.0043 |
| normal $N(1, 4)$ | Box-Muller | 1.0069 (1.0000) | 3.9767 (4.0000) | 0.0038 |
| log-normal, $\mu = 0$, $\sigma = 0.5$ | exp of a normal | 1.1342 (1.1331) | 0.3656 (0.3647) | 0.0018 |
| chi-square, $k = 5$ | sum of squared normals | 4.9938 (5.0000) | 9.9272 (10.0000) | 0.0017 |
| Student's $t$, $\nu = 5$ | normal over $\sqrt{\chi^2/\nu}$ | 0.0028 (0.0000) | 1.6549 (1.6667) | 0.0023 |
| gamma, shape 2.5, scale 2 | Marsaglia-Tsang | 5.0023 (5.0000) | 10.0014 (10.0000) | 0.0024 |
| gamma, shape 0.3 | Marsaglia-Tsang with boosting | 0.2986 (0.3000) | 0.2995 (0.3000) | 0.0020 |
| beta $(2, 5)$ | ratio of gammas | 0.2857 (0.2857) | 0.0257 (0.0255) | 0.0023 |
| Cauchy | inversion | (none) | (none) | 0.0032 |
| Bernoulli, $p = 0.3$ | comparison | 0.2979 (0.3000) | 0.2092 (0.2100) | 0.0021 |
| geometric, $p = 0.2$ (failures) | inversion | 4.0329 (4.0000) | 20.3440 (20.0000) | 0.0034 |
| Poisson, $\lambda = 4$ | Knuth's products | 4.0028 (4.0000) | 4.0114 (4.0000) | 0.0013 |
| Poisson, $\lambda = 300$ | Ahrens-Dieter recursion | 299.9857 (300.0000) | 300.5645 (300.0000) | 0.0014 |
| binomial, $n = 20$, $p = 0.3$ | inversion | 5.9981 (6.0000) | 4.1773 (4.2000) | 0.0014 |
| binomial, $n = 100000$, $p = 0.4$ | beta splitting | 40000.1401 (40000.0000) | 24016.5330 (24000.0000) | 0.0031 |

With $n = 100000$ samples the 5% critical value of $D$ is $1.36/\sqrt n = 0.0043$.
