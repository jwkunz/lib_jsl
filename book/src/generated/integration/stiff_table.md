| problem | solver | steps | rejected | evaluations of $f$ | Jacobians | error in $y_1$ |
|---|---|---:|---:|---:|---:|---:|
| $y' = -1000(y - \cos t)$ on $[0, 1]$ | Dormand-Prince | 381 | 0 | 2287 | 0 | 7.5e-8 |
|  | Rosenbrock 2(3) | 616 | 0 | 2465 | 616 | 1.4e-7 |
|  | BDF, orders 1–5 | 114 | 5 | 477 | 119 | 6.5e-10 |
| Van der Pol, $\mu = 100$, on $[0, 200]$ | Dormand-Prince | 11953 | 591 | 75265 | 0 | 3.3e-7 |
|  | Rosenbrock 2(3) | 5298 | 2 | 26501 | 5300 | 1.8e-5 |
|  | BDF, orders 1–5 | 878 | 42 | 4601 | 920 | 4.7e-6 |
| Robertson kinetics on $[0, 40]$ | Dormand-Prince | 34560 | 1207 | 214603 | 0 | 5.6e-11 |
|  | Rosenbrock 2(3) | 537 | 0 | 3223 | 537 | 2.5e-8 |
|  | BDF, orders 1–5 | 134 | 6 | 841 | 140 | 1.3e-6 |
