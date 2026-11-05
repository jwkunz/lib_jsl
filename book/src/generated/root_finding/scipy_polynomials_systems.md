| problem (SciPy reference) | largest difference in the roots |
|---|---:|
| Wilkinson-like, roots 1 to 10 (`numpy.roots`) | 4e-9; from the exact roots, library 3e-10 and NumPy 3e-9, and the library's Newton-polished real roots 1e-10 |
| x^5 - x - 1 (`numpy.roots`) | 3e-15 |
| (x - 1)^2 (x + 2) (x^2 + 1) (`numpy.roots`) | 6e-10 |
| degree 12, coefficients 1 to 13 (`numpy.roots`) | 1e-15 |
| system: circle and exponential (`optimize.fsolve`) | Newton 0, Broyden 2e-16 |
| system: three trigonometric equations (`optimize.fsolve`) | Newton 1e-16, Broyden 1e-16 |
| system: Broyden tridiagonal, 5 variables (`optimize.fsolve`) | Newton 1e-16, Broyden 1e-16 |
