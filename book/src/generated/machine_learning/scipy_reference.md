| model (SciPy reference) | largest difference |
|---|---:|
| least squares with intercept (`linalg.lstsq`) | 1e-14 |
| ridge, alpha = 10 (`linalg.solve` of the centred normal equations) | 5e-16 |
| lasso, alpha = 0.2 (`optimize.minimize`, L-BFGS-B on split weights) | 1e-10 |
| elastic net, alpha = 0.2, rho = 0.5 (`optimize.minimize`, L-BFGS-B on split weights) | 5e-9 |
| logistic regression, L2 penalty 0.01: log-odds coefficients (`optimize.minimize`, BFGS) | 9e-9; probabilities 9e-11 |
| Gaussian naive Bayes, posterior probabilities (`stats.norm.logpdf`) | 4e-16 |
| linear discriminant analysis: Fisher ratios and unit directions (`linalg.eigh(S_b, S_w)`) | ratios 1e-12, directions 2e-13 |
