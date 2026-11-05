| rule (SciPy reference) | largest difference |
|---|---:|
| Gauss-Legendre, 5 points (`special.roots_legendre`) | nodes 0; weights 2e-16 (relative); against mpmath, library 1e-16 and SciPy 2e-16 |
| Gauss-Legendre, 20 points (`special.roots_legendre`) | nodes 1e-16; weights 3e-15 (relative); against mpmath, library 1e-15 and SciPy 4e-15 |
| Gauss-Legendre, 100 points (`special.roots_legendre`) | nodes 1e-16; weights 2e-13 (relative); against mpmath, library 3e-15 and SciPy 2e-13 |
| Gauss-Hermite, 10 points (`special.roots_hermite`) | nodes 4e-16, weights 1e-15 (relative) |
| Gauss-Hermite, 40 points (`special.roots_hermite`) | nodes 9e-16, weights 6e-16 (relative) |
| Gauss-Laguerre, 10 points, alpha = 0 (`special.roots_genlaguerre`) | nodes 4e-16, weights 7e-15 (relative) |
| Gauss-Laguerre, 20 points, alpha = 1.5 (`special.roots_genlaguerre`) | nodes 4e-15, weights 1e-14 (relative) |
