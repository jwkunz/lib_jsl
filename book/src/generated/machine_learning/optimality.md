| model | optimality condition | largest violation |
|---|---|---:|
| least squares | $X^{\mathsf T}r = 0$ (relative to $\lVert X\rVert\lVert y\rVert$) | 5.2e-17 |
| ridge, $\alpha = 5$ | $X^{\mathsf T}r = \alpha w$ (relative) | 6.4e-17 |
| lasso, $\alpha = 0.1$ | subgradient: $\tfrac1n x_j^{\mathsf T}r - \alpha(1-\rho)w_j \in \alpha\rho\,\partial\lvert w_j\rvert$ | 9.6e-11 (6 of 8 weights non-zero) |
| elastic net, $\alpha = 0.1$, $\rho = 0.5$ | subgradient: $\tfrac1n x_j^{\mathsf T}r - \alpha(1-\rho)w_j \in \alpha\rho\,\partial\lvert w_j\rvert$ | 9.4e-11 (7 of 8 weights non-zero) |
| logistic regression, $\lambda = 0.01$ | gradient of the penalised log-loss | 6.7e-8 (15 L-BFGS iterations) |
