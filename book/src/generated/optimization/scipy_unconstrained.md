| problem | method (library / SciPy) | library: distance from the minimiser | SciPy: distance from the minimiser |
|---|---:|---:|---:|
| Rosenbrock, 2 variables | Newton / Newton-CG | 0 | 7e-10 |
|  | BFGS | 4e-16 | 2e-15 |
|  | L-BFGS (5 pairs) / L-BFGS-B | 0 | 3e-11 |
|  | conjugate gradient (PR+) / CG | 0 | 2e-12 |
|  | Nelder-Mead | 1e-7 | 2e-13 |
|  | Powell | 7e-15 | 4e-14 |
| Rosenbrock, 5 variables | Newton / Newton-CG | 2e0 | 2e0 |
|  | BFGS | 0 | 6e-15 |
|  | L-BFGS (5 pairs) / L-BFGS-B | 0 | 4e-10 |
|  | conjugate gradient (PR+) / CG | 4e-14 | 2e-12 |
|  | Nelder-Mead | 4e-8 | 1e-13 |
|  | Powell | 0 | 1e-11 |
| can of least area, 1 variable | golden section / golden, Brent | 7e-9 | 6e-9, 4e-11 |
